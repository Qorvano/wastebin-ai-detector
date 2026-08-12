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
    CONF_CAMERA,
    CONF_CAPTURE_INTERVAL,
    CONF_CONFIRM_SCANS,
    CONF_SCAN_INTERVAL,
    DEFAULT_CAPTURE_INTERVAL_MIN,
    DEFAULT_CONFIRM_SCANS,
    DEFAULT_SCAN_INTERVAL_MIN,
    DOMAIN,
)
from .core import (
    DetectionResult,
    Profile,
    WastebinError,
    detect,
    load_image_rgb_bytes,
)
from .storage import WastebinStorage, archive_dir

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
        self.last_confident_update: dict[str, datetime] = {}
        self._confirm_scans: int = int(
            entry.options.get(CONF_CONFIRM_SCANS, DEFAULT_CONFIRM_SCANS)
        )
        self._pending_flips: dict[str, int] = {}
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
        profile = self.storage.profile
        if profile is not None:
            diag["limit_daylight_sat_min"] = profile.daylight_sat_min
            diag["limit_overexposure_clip_max"] = profile.overexposure_clip_max
            diag["limit_daylight_val_max"] = profile.daylight_val_max
        diag.update(extra)
        self.diagnostics = diag

    async def _async_update_data(self) -> DetectionResult:
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
            self._set_diagnostics("camera_error", error=str(err))
            raise UpdateFailed(f"camera snapshot failed: {err}") from err
        try:
            result = await self.hass.async_add_executor_job(
                self._detect_bytes, image.content, profile
            )
        except WastebinError as err:
            self._set_diagnostics("detect_error", error=str(err))
            raise UpdateFailed(f"detection failed: {err}") from err
        now = dt_util.utcnow()
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
            self._set_diagnostics(
                outcome,
                result,
                held_previous_state=self.data is not None,
                hint=(
                    "label a snapshot of these conditions as calibration "
                    "data to widen the learned quality gates"
                ),
            )
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
        return stabilized

    def _apply_stability(self, result: DetectionResult, now: datetime) -> DetectionResult:
        """Per-bin acceptance: learned-uncertainty hold plus optional
        k-confirmation for state flips (clear evidence switches with
        the default of 1 immediately)."""
        if self.data is None and any(b.uncertain for b in result.bins):
            # Cold start (fresh setup or entry reload) on an ambiguous
            # frame: publishing its raw threshold verdict would let a
            # sensor flip across the reload boundary on evidence that
            # never confidently established anything. Stay unavailable
            # like the quality gates do; the first confident frame is
            # accepted immediately.
            self._set_diagnostics(
                "ambiguous_cold_start",
                result,
                ambiguous_bins=[b.id for b in result.bins if b.uncertain],
            )
            raise UpdateFailed(
                "first analysis is ambiguous for at least one bin; "
                "waiting for a confident frame"
            )
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

    @staticmethod
    def _detect_bytes(data: bytes, profile: Profile) -> DetectionResult:
        return detect(load_image_rgb_bytes(data), profile)


class LearningCollector:
    """Archives daylight snapshots while learning mode is enabled."""

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
        if not self._storage.learning:
            return
        # Computed from the configured location, independent of the sun
        # entity. IR/greyscale night frames carry no color information;
        # archiving them would only pollute the calibration set.
        if not is_up(self._hass):
            return
        try:
            await self.async_capture_now()
        except HomeAssistantError as err:
            _LOGGER.debug("learning capture skipped: %s", err)

    async def async_capture_now(self) -> str:
        """Capture one frame into the archive; returns the filename.

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
        return filename

    def _write(self, filename: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / filename).write_bytes(data)
