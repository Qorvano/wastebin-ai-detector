"""Detection coordinator and the learning snapshot collector.

The coordinator runs the periodic detection against the learned
profile. Night/IR frames (greyscale suspect) never overwrite the last
daylight result: color detection has nothing to say in the dark, so the
entities hold their state until the next daylight analysis.

The collector is the integration-internal replacement for a manual
snapshot automation: while learning mode is on, it archives a camera
frame at a fixed cadence during daylight. Archived frames become
calibration data as soon as the user labels them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.camera import async_get_image
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.sun import is_up
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BIN_ACTIVE,
    CONF_BINS,
    CONF_CAMERA,
    CONF_CAPTURE_INTERVAL,
    CONF_CONFIRM_SCANS,
    CONF_SCAN_INTERVAL,
    CONF_TRUST_MARKS,
    DEFAULT_CAPTURE_INTERVAL_MIN,
    DEFAULT_CONFIRM_SCANS,
    DEFAULT_SCAN_INTERVAL_MIN,
    DOMAIN,
)
from .core import (
    AutoStamp,
    CalibrationError,
    DetectionResult,
    Profile,
    WastebinError,
    detect,
    learn_color_model,
    load_image_rgb_bytes,
    reference_rects,
    reservoir_decision,
)
from .storage import WastebinStorage, archive_dir, widen_profile_gates

_LOGGER = logging.getLogger(__name__)


class WastebinCoordinator(DataUpdateCoordinator[DetectionResult]):
    """Periodic detection against the learned profile."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, storage: WastebinStorage
    ) -> None:
        scan_minutes = entry.options.get(
            CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL_MIN
        )
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN} {entry.title}",
            update_interval=timedelta(minutes=scan_minutes),
        )
        self.entry = entry
        self.storage = storage
        self.camera_entity: str = entry.data[CONF_CAMERA]
        self.last_daylight_update: datetime | None = None
        self.last_greyscale_skip: datetime | None = None
        self.last_overexposure_skip: datetime | None = None
        self.last_frame_integrity_skip: datetime | None = None
        self.last_confident_update: dict[str, datetime] = {}
        self._active_bin_ids: set[str] = {
            b["id"]
            for b in entry.data.get(CONF_BINS, [])
            if b.get(CONF_BIN_ACTIVE, True)
        }
        self._confirm_scans: int = int(
            entry.options.get(CONF_CONFIRM_SCANS, DEFAULT_CONFIRM_SCANS)
        )
        self._pending_flips: dict[str, int] = {}
        # Single-flight guard: a timed-out executor thread keeps
        # running (asyncio cannot abort it, and its future reads as
        # done once the awaiting task was cancelled), so the worker
        # itself clears this flag in a finally block.
        self._executor_busy: bool = False
        # Why the last analysis ended the way it did, with the measured
        # values against the learned limits. Exposed by the always
        # available status sensor: outcomes must never be invisible.
        self.diagnostics: dict[str, Any] = {"outcome": "no_run_yet"}

    def _set_diagnostics(
        self,
        outcome: str,
        result: DetectionResult | None = None,
        **extra: Any,
    ) -> None:
        diag: dict[str, Any] = {
            "outcome": outcome,
            "at": dt_util.utcnow().isoformat(),
        }
        if result is not None:
            diag["median_sat"] = result.median_sat
            diag["median_val"] = result.median_val
            diag["clip_frac"] = result.clip_frac
            diag["row_dup_frac"] = result.row_dup_frac
        profile = self.storage.profile
        if profile is not None:
            diag["limit_daylight_sat_min"] = profile.daylight_sat_min
            diag["limit_overexposure_clip_max"] = profile.overexposure_clip_max
            diag["limit_daylight_val_max"] = profile.daylight_val_max
            diag["limit_row_dup_max"] = profile.row_dup_max
        diag.update(extra)
        self.diagnostics = diag

    async def _async_update_data(self) -> DetectionResult:
        # Watchdog: one analysis must finish within one scheduled cycle
        # (a coordinator wedged for hours happened in the field). The
        # bound derives from the configured cadence. It covers the
        # awaitable phases (executor detection, persistence); the
        # camera fetch carries its own timeout, and a hang that blocks
        # the event loop itself is beyond any asyncio watchdog. A timed
        # out executor thread keeps running, so a single-flight guard
        # below bounds the leak to one thread.
        if self._executor_busy:
            self._set_diagnostics("previous_run_still_busy")
            raise UpdateFailed(
                "previous analysis is still running in the executor; "
                "skipping this cycle"
            )
        timeout_s = self._scan_seconds()
        try:
            async with asyncio.timeout(timeout_s):
                return await self._async_analyze()
        except TimeoutError as err:
            self._set_diagnostics("watchdog_timeout", timeout_seconds=timeout_s)
            raise UpdateFailed(
                f"analysis exceeded one scan cycle ({timeout_s:.0f} s) and "
                "was aborted by the watchdog"
            ) from err

    async def _async_analyze(self) -> DetectionResult:
        profile = self.storage.profile
        if profile is None:
            self._set_diagnostics("not_calibrated")
            raise UpdateFailed(
                "not calibrated yet: add samples and labels, then call the "
                f"{DOMAIN}.relearn service"
            )
        try:
            image = await async_get_image(self.hass, self.camera_entity)
        except HomeAssistantError as err:
            self._set_diagnostics(
                "camera_error",
                error=str(err),
                hint=(
                    "if this persists although the camera delivers images "
                    "elsewhere, reload this integration entry (observed "
                    "after camera re-registrations)"
                ),
            )
            raise UpdateFailed(f"camera snapshot failed: {err}") from err
        try:
            self._executor_busy = True
            result = await self.hass.async_add_executor_job(
                self._detect_bytes_guarded, image.content, profile
            )
        except WastebinError as err:
            self._set_diagnostics("detect_error", error=str(err))
            raise UpdateFailed(f"detection failed: {err}") from err
        now = dt_util.utcnow()
        # Retired bins may linger in the profile until the next relearn
        # lands; their results have no entity and must not steer the
        # stability logic (e.g. the cold-start ambiguity hold).
        result = replace(
            result,
            bins=[b for b in result.bins if b.id in self._active_bin_ids],
        )
        if result.frame_integrity_suspect:
            # A smeared/truncated frame has PLAUSIBLE color statistics
            # (it repeats real rows), so it must be rejected before the
            # light gates even look at it - and never absorbed, or the
            # gate would learn the pathology as normal.
            self.last_frame_integrity_skip = now
            self._pending_flips.clear()
            self._set_diagnostics(
                "hold_frame_integrity",
                result,
                held_previous_state=self.data is not None,
                hint=(
                    "the camera stream delivered a corrupted frame "
                    "(duplicated rows); check camera/stream health if "
                    "this persists"
                ),
            )
            if self.data is not None:
                return self.data
            raise UpdateFailed(
                "frame integrity suspect (smeared/truncated keyframe) and "
                "no prior result yet; waiting for a clean frame"
            )
        if result.grayscale_suspect or result.overexposure_suspect:
            # Degraded evidence (IR night frame or harsher light than
            # anything calibrated): hold the last accepted state rather
            # than publishing a verdict the color signal cannot carry.
            # Both diagnostics are stamped independently; a frame can
            # legitimately trip both gates.
            reasons = []
            if result.grayscale_suspect:
                reasons.append("greyscale")
            if result.overexposure_suspect:
                reasons.append("overexposure")
            outcome = "hold_" + "_and_".join(reasons)
            if result.grayscale_suspect:
                self.last_greyscale_skip = now
            if result.overexposure_suspect:
                self.last_overexposure_skip = now
            # A gated frame breaks the chain of consecutive confident
            # analyses the confirm_scans option promises.
            self._pending_flips.clear()
            # Diagnostics first, so the shown limits are the ones this
            # frame was actually judged against (absorption right after
            # may widen them for the NEXT frame).
            self._set_diagnostics(
                outcome,
                result,
                held_previous_state=self.data is not None,
                hint=(
                    "label a snapshot of these conditions as calibration "
                    "data to widen the learned quality gates"
                ),
            )
            if not result.grayscale_suspect and is_up(self.hass):
                # Self-heal applies only in the harsh-light direction.
                # Greyscale-suspect frames are exactly the low-sat
                # cluster the night gate exists to reject: absorbing
                # one would widen daylight_sat_min down to IR levels
                # and permanently disable night protection.
                await self._async_absorb_gate_sample(result)
            if self.data is not None:
                return self.data
            raise UpdateFailed(
                "frame quality too low (greyscale/IR or overexposed) and "
                "no prior daylight result yet; waiting for the first "
                "clean analysis"
            )
        self.last_daylight_update = now
        stabilized = self._apply_stability(result, now)
        self._set_diagnostics("ok", stabilized)
        if is_up(self.hass):
            # Clean daylight frames teach the light gates too, so the
            # learned daily range keeps tracking the season, labels not
            # required.
            await self._async_absorb_gate_sample(result)
        return stabilized

    def _apply_stability(self, result: DetectionResult, now: datetime) -> DetectionResult:
        """Per-bin acceptance: learned-uncertainty hold plus optional
        k-confirmation for state flips (clear evidence switches with
        the default of 1 immediately)."""
        if self.data is None:
            ambiguous = [b.id for b in result.bins if b.uncertain]
            confident = [b for b in result.bins if not b.uncertain]
            if not confident:
                # Cold start on a fully ambiguous frame: nothing can be
                # published without risking a flip across the reload
                # boundary on evidence that never confidently
                # established anything.
                self._set_diagnostics(
                    "ambiguous_cold_start", result, ambiguous_bins=ambiguous
                )
                raise UpdateFailed(
                    "first analysis is ambiguous for every bin; waiting "
                    "for a confident frame"
                )
            if ambiguous:
                # Per-bin cold start: confident bins go live NOW; only
                # the ambiguous ones stay unavailable (their sensors
                # show nothing rather than a guess) until their own
                # first confident frame. One chronically ambiguous bin
                # must not blind the others.
                for bin_id in ambiguous:
                    self._pending_flips.pop(bin_id, None)
                self.last_daylight_update = now
                for b in confident:
                    self.last_confident_update[b.id] = now
                return replace(result, bins=confident)
        previous = {b.id: b for b in self.data.bins} if self.data else {}
        bins = []
        for result_bin in result.bins:
            prev_bin = previous.get(result_bin.id)
            if result_bin.uncertain:
                # Ambiguous frame: never flip on it, hold the previous
                # state, and break this bin's confident-flip chain
                # (confirm_scans counts consecutive confident analyses).
                self._pending_flips.pop(result_bin.id, None)
                if prev_bin is not None:
                    result_bin = replace(result_bin, present=prev_bin.present)
                    bins.append(result_bin)
                # No previous state for this bin (it sat out the cold
                # start): publishing its raw threshold verdict now
                # would be exactly the guess the cold-start rule
                # forbids - keep it unpublished until confident.
                continue
            if prev_bin is None or result_bin.present == prev_bin.present:
                self._pending_flips.pop(result_bin.id, None)
                self.last_confident_update[result_bin.id] = now
                bins.append(result_bin)
                continue
            # Confident evidence contradicting the accepted state.
            count = self._pending_flips.get(result_bin.id, 0) + 1
            if count >= self._confirm_scans:
                self._pending_flips.pop(result_bin.id, None)
                self.last_confident_update[result_bin.id] = now
                bins.append(result_bin)
            else:
                self._pending_flips[result_bin.id] = count
                bins.append(
                    replace(result_bin, present=prev_bin.present, uncertain=True)
                )
        return replace(result, bins=bins)

    async def _async_absorb_gate_sample(self, result: DetectionResult) -> None:
        self.storage.gate_samples.append(
            [
                result.median_sat,
                result.median_val,
                result.clip_frac,
                result.row_dup_frac,
            ]
        )
        # Rolling window of one full daily light cycle at the current
        # cadence: older extremes are already baked into the profile by
        # the monotone widening below, so keeping raw samples beyond
        # one day adds nothing and would grow the store without bound.
        # Aging out is also the escape hatch for anomalies: a relearn
        # rebuilds the gates from labels and re-widens only from this
        # window.
        window = max(int(86400.0 / max(self._scan_seconds(), 1.0)), 1)
        if len(self.storage.gate_samples) > window:
            del self.storage.gate_samples[:-window]
        profile = self.storage.profile
        if profile is None:
            return
        # Persist only when the limits actually moved; unsaved window
        # entries regenerate from live frames after a restart.
        if widen_profile_gates(profile, self.storage.gate_samples):
            await self.storage.async_save()

    def _scan_seconds(self) -> float:
        if self.update_interval is not None:
            return self.update_interval.total_seconds()
        return DEFAULT_SCAN_INTERVAL_MIN * 60.0

    def _detect_bytes_guarded(
        self, data: bytes, profile: Profile
    ) -> DetectionResult:
        try:
            return self._detect_bytes(data, profile)
        finally:
            # Runs in the executor thread: clears the single-flight
            # flag even when the awaiting task timed out long ago.
            self._executor_busy = False

    @staticmethod
    def _detect_bytes(data: bytes, profile: Profile) -> DetectionResult:
        return detect(load_image_rgb_bytes(data), profile)


def _declaration_signature(
    present: list[str], absent: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(sorted(present)), tuple(sorted(absent)))


def _entry_signature(entry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if entry is None:
        return ((), ())
    return _declaration_signature(entry.present, entry.absent)


def _evaluate_auto_frame(
    data: bytes,
    profile: Profile,
    references: dict[str, list],
    retained: list[tuple[str, tuple[float, float]]],
    trust_marks: bool = False,
) -> dict[str, Any]:
    """Decide whether this frame earns a slot in the reservoir.

    Pure and HA-free, so it runs in the executor. Two gate classes,
    both MODEL-FREE by design:

    - frame-level light validity, reusing the very gates detection
      already applies (greyscale night, overexposure, broken keyframe);
    - patch coherence, which asks "is this patch ONE colour", never "is
      it the colour I already know".

    The second distinction is the whole point. A guard of the form
    "only sample when it matches the current model" would reject
    exactly the frames carrying new information - a lid in light the
    models have never seen - which is what the collection exists for.

    With trust_marks the two gates that judge the LIGHT of a still
    colourful frame step aside: an overexposed frame and a patch whose
    hue has scattered are precisely the washed-out conditions the mode
    exists to learn. The two that judge whether there is any usable
    colour at all stay: a broken keyframe is corrupted data rather than
    a light condition, and a greyscale night frame carries no hue to
    learn - training on it would drag every saturation floor to zero
    and with it any ability to tell the bins apart.
    """
    img = load_image_rgb_bytes(data)
    result = detect(img, profile)
    if result.frame_integrity_suspect:
        return {"accept": False, "reason": "frame_integrity", "evict": None}
    if result.grayscale_suspect:
        return {"accept": False, "reason": "greyscale", "evict": None}
    if result.overexposure_suspect and not trust_marks:
        return {"accept": False, "reason": "overexposed", "evict": None}
    # Coherence: every referenced patch must show one colour. A rect
    # that has slid half onto the ground (a bin was moved despite the
    # standing precondition) fails here without any model knowledge.
    for bin_id, rects in references.items():
        hue, sat, val = _patch_pixels(img, profile, rects)
        if hue.size == 0:
            return {"accept": False, "reason": "empty_patch", "evict": None}
        try:
            learn_color_model(
                hue, sat, val, bin_id=bin_id, trust_marks=trust_marks
            )
        except CalibrationError:
            return {"accept": False, "reason": "incoherent_patch", "evict": None}
    coord = (result.median_sat, result.median_val)
    decision = reservoir_decision(retained, coord)
    return {
        "accept": decision.accept,
        "reason": decision.reason,
        "evict": decision.evict,
        "coord": coord,
    }


def _patch_pixels(img, profile: Profile, rects: list):
    """Pixels of the reference patches, through each rect's own stored
    extraction grid - exactly how the learner reads them."""
    import numpy as np

    from .core import extract_working_roi, rect_to_pixels, rgb_to_hsv
    from .core.store import image_rect_in_roi

    hues, sats, vals = [], [], []
    for sample in rects:
        grid = image_rect_in_roi(sample.rect, sample.roi)
        if grid is None:
            continue
        arr = extract_working_roi(
            img, sample.roi, profile.working_width, profile.resample
        )
        hue, sat, val = rgb_to_hsv(arr)
        try:
            x0, y0, x1, y1 = rect_to_pixels(
                grid.x, grid.y, grid.w, grid.h, hue.shape[1], hue.shape[0]
            )
        except WastebinError:
            continue
        hues.append(hue[y0:y1, x0:x1].ravel())
        sats.append(sat[y0:y1, x0:x1].ravel())
        vals.append(val[y0:y1, x0:x1].ravel())
    if not hues:
        return np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(hues),
        np.concatenate(sats),
        np.concatenate(vals),
    )


class LearningCollector:
    """Archives daylight snapshots while learning mode is enabled, and
    re-applies the user's own lid marks to the informative ones.

    The unattended half collects COLOUR evidence only. It never writes
    a presence label: the threshold is anchored by the smallest
    positive and largest negative blob a human confirmed, and the only
    machine that could produce such a label is the detector itself.
    """

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, storage: WastebinStorage
    ) -> None:
        self._hass = hass
        self._storage = storage
        self._camera: str = entry.data[CONF_CAMERA]
        self._interval = timedelta(
            minutes=entry.options.get(
                CONF_CAPTURE_INTERVAL, DEFAULT_CAPTURE_INTERVAL_MIN
            )
        )
        self._dir = archive_dir(hass, entry.entry_id)
        self._unsub: CALLBACK_TYPE | None = None
        self._entry = entry
        self._auto_busy = False
        self.auto_diagnostics: dict[str, Any] = {}

    @callback
    def async_start(self) -> None:
        if self._unsub is None:
            self._unsub = async_track_time_interval(
                self._hass, self._async_interval, self._interval
            )

    @callback
    def async_stop(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _async_interval(self, _now: datetime) -> None:
        if self._storage.learning_declaration is None:
            # A declared run IS learning mode: without one there is
            # nothing a captured frame could mean, so nothing is
            # captured (and the archive stops growing for nothing).
            return
        # Computed from the configured location, independent of the sun
        # entity. IR/greyscale night frames carry no color information;
        # archiving them would only pollute the calibration set.
        if not is_up(self._hass):
            return
        try:
            filename, data = await self._async_capture_bytes()
        except HomeAssistantError as err:
            _LOGGER.debug("learning capture skipped: %s", err)
            return
        await self._async_maybe_auto_sample(filename, data)

    async def _async_maybe_auto_sample(
        self, filename: str, data: bytes
    ) -> None:
        """Record this frame as one observation of the DECLARED
        situation, if it adds anything.

        Cheapest checks first; every rejection is silent and leaves no
        store entry behind, so declining costs nothing.
        """
        storage = self._storage
        declaration = storage.learning_declaration
        if declaration is None:
            return
        if storage.auto_paused is not None:
            self._note_skip("paused")
            return
        if self._auto_busy:
            self._note_skip("still_working")
            return
        profile = storage.profile
        if profile is None:
            self._note_skip("no_profile")
            # Without a learned profile there are no light gates, so
            # "is this frame usable" is unanswerable. Refusing is the
            # honest answer; the first manual calibration provides it.
            return
        runtime = getattr(self._entry, "runtime_data", None)
        if runtime is None:
            return
        if runtime.relearn_lock.locked():
            self._note_skip("relearn_running")
            return
        present = [b for b, state in declaration.items() if state == "present"]
        absent = [b for b, state in declaration.items() if state == "absent"]
        # Marks of an ABSENT bin would sample the ground, so only the
        # bins declared present contribute colour.
        references = {
            bin_id: rects
            for bin_id, rects in reference_rects(storage.calibration).items()
            if bin_id in present
        }
        missing_marks = [b for b in present if b not in references]
        if missing_marks:
            # A bin declared present without a current mark cannot be
            # sampled, and labelling it present anyway would assert
            # something no patch ever checked. The run waits until the
            # user marks it (or ends the run and declares it away).
            self._note_skip("no_mark_for:" + ",".join(sorted(missing_marks)))
            return
        # One reservoir per declared SITUATION: absent runs must not be
        # evicted by present runs, and vice versa - they are different
        # evidence, not competing samples of the same thing.
        signature = _declaration_signature(present, absent)
        retained = [
            (path, coord)
            for path, coord in storage.calibration.auto_reservoir()
            if _entry_signature(storage.calibration.get_image(path))
            == signature
        ]
        self._auto_busy = True
        try:
            outcome = await self._hass.async_add_executor_job(
                _evaluate_auto_frame,
                data,
                profile,
                references,
                list(retained),
                bool(self._entry.options.get(CONF_TRUST_MARKS, False)),
            )
        except WastebinError as err:
            _LOGGER.debug("learning run skipped this frame: %s", err)
            return
        except Exception:  # noqa: BLE001 - never wedge the collector
            _LOGGER.exception("learning run failed to evaluate a frame")
            return
        finally:
            self._auto_busy = False
        self.auto_diagnostics = {
            **self.auto_diagnostics,
            "last_reason": outcome["reason"],
        }
        if not outcome["accept"]:
            return
        # The gates were checked before an await; re-check the ones a
        # concurrent relearn or an unload can have changed meanwhile.
        if (
            storage.learning_declaration != declaration
            or storage.auto_paused is not None
            or getattr(self._entry, "runtime_data", None) is None
        ):
            return
        stamp = AutoStamp(*outcome["coord"])
        storage.calibration.record_auto_frame(
            filename,
            stamp,
            samples={
                bin_id: [r.rect for r in rects]
                for bin_id, rects in references.items()
            },
            present=present,
            absent=absent,
        )
        if outcome["evict"]:
            storage.calibration.forget_image(outcome["evict"])
        await storage.async_save()
        from .services import async_relearn  # local: avoids a cycle

        try:
            await async_relearn(self._hass, self._entry)
        except HomeAssistantError as err:
            _LOGGER.warning("relearn after auto sampling failed: %s", err)

    async def async_capture_now(self) -> str:
        """Capture one frame into the archive; returns the filename."""
        filename, _data = await self._async_capture_bytes()
        return filename

    async def _async_capture_bytes(self) -> tuple[str, bytes]:
        """Capture one frame into the archive; returns (name, bytes).

        Raises HomeAssistantError for both camera and filesystem
        failures, so every caller has a single error contract.
        """
        image = await async_get_image(self._hass, self._camera)
        # Microsecond precision: two captures within the same second
        # must not silently overwrite each other (and their labels).
        filename = dt_util.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        try:
            await self._hass.async_add_executor_job(
                self._write, filename, image.content
            )
        except OSError as err:
            raise HomeAssistantError(
                f"cannot write snapshot to {self._dir}: {err}"
            ) from err
        return filename, image.content

    def _note_skip(self, reason: str) -> None:
        """Why the last tick collected nothing - otherwise a run that
        silently does nothing looks identical to one that works."""
        self.auto_diagnostics = {
            **self.auto_diagnostics,
            "last_reason": reason,
        }

    def _write(self, filename: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / filename).write_bytes(data)
