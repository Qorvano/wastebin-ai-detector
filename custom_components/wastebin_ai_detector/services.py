"""Domain services: capture, sample, label, relearn, lifecycle.

These are the scriptable calibration API. The upcoming calibration
card is a thin UI over exactly these services, so everything the card
will do can already be done today from Developer Tools or automations.

Payload SHAPE validation lives in the registered vol schemas: shape
errors then fail in core's schema phase and reach every transport as a
clean invalid-input error. Only STATE-dependent validation (unknown
entry, unknown bin, unknown image) stays inside the handlers as
ServiceValidationError.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import ServiceValidationError
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers import issue_registry as ir

from .const import (
    CONF_ROI_POLYGONS,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_WORKING_WIDTH,
    CONF_CAMERA,
    ATTR_ABSENT,
    ATTR_AUTO_RELEARN,
    ATTR_BIN,
    ATTR_ENTRY_ID,
    ATTR_FILENAME,
    ATTR_FILENAMES,
    ATTR_PRESENT,
    ATTR_RECT,
    ATTR_SPACE,
    DOMAIN,
    SERVICE_ADD_SAMPLE,
    SERVICE_CAPTURE,
    SERVICE_FORGET_IMAGE,
    SERVICE_LABEL_IMAGE,
    SERVICE_MARK_BIN_CHANGED,
    SERVICE_DISCARD_AUTO,
    SERVICE_START_LEARNING,
    SERVICE_STOP_LEARNING,
    SERVICE_RECONFIRM_IMAGES,
    SERVICE_RESTORE_AUTO,
    SERVICE_RELEARN,
    SERVICE_RESTORE_IMAGE,
    SERVICE_SET_ROI,
)
from .core import (
    HUE_TOL_PERCENTILE,
    REL_EPS,
    CalibrationError,
    ProfileError,
    Rect,
    Roi,
    WastebinError,
    AdoptionVerdict,
    adoption_verdict,
    has_auto_evidence,
    misclassified_manual_labels,
    over_capacity_paths,
    reservoir_by_situation,
    without_auto_evidence,
    learn_profile,
    rect_as_rings,
    rings_bbox,
    rings_equal,
    roi_rect_to_image_rect,
    store_from_dict,
    store_to_dict,
)
from .core.region import clamp_rings, validate_rings
from .storage import store_anchor, widen_profile_gates


def _plain_filename(value: Any) -> str:
    """Schema validator: a bare archive file name, no path components.

    Service input reaches filesystem paths via the calibration store;
    a path component or absolute path would escape the archive.
    """
    name = cv.string(value)
    if not name or Path(name).name != name or name in (".", ".."):
        raise vol.Invalid(
            "filename must be a plain file name from the calibration archive"
        )
    return name


def _unit_rect(value: list[float]) -> list[float]:
    """Schema validator: [x, y, w, h] inside the unit frame, non-empty."""
    x, y, w, h = value
    if (
        w <= 0.0
        or h <= 0.0
        or x < -REL_EPS
        or y < -REL_EPS
        or x + w > 1.0 + REL_EPS
        or y + h > 1.0 + REL_EPS
    ):
        raise vol.Invalid(
            "rect must be [x, y, w, h] with positive size inside [0, 1]"
        )
    return value


_ENTRY_SCHEMA = {vol.Optional(ATTR_ENTRY_ID): cv.string}

CAPTURE_SCHEMA = vol.Schema(_ENTRY_SCHEMA)

ADD_SAMPLE_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): _plain_filename,
        vol.Required(ATTR_BIN): cv.string,
        vol.Required(ATTR_RECT): vol.All(
            [vol.Coerce(float)], vol.Length(min=4, max=4), _unit_rect
        ),
        vol.Optional(ATTR_SPACE, default="image"): vol.In(("image", "roi")),
    }
)

LABEL_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): _plain_filename,
        vol.Optional(ATTR_PRESENT, default=[]): [cv.string],
        vol.Optional(ATTR_ABSENT, default=[]): [cv.string],
        vol.Optional(ATTR_AUTO_RELEARN, default=True): cv.boolean,
    }
)

RELEARN_SCHEMA = vol.Schema(_ENTRY_SCHEMA)

FILENAME_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): _plain_filename,
    }
)

RECONFIRM_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAMES): vol.All(
            [_plain_filename], vol.Length(min=1)
        ),
    }
)

START_LEARNING_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Optional(ATTR_PRESENT, default=[]): [cv.string],
        vol.Optional(ATTR_ABSENT, default=[]): [cv.string],
    }
)

MARK_BIN_CHANGED_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_BIN): cv.string,
    }
)


def _valid_roi_payload(data: dict) -> dict:
    x, y = data[CONF_ROI_X], data[CONF_ROI_Y]
    w, h = data[CONF_ROI_W], data[CONF_ROI_H]
    # NaN passes every ordering comparison below (all False) and would
    # be persisted as null by the orjson-based entry store, bricking
    # the entry on the next restart - reject non-finite values first.
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise vol.Invalid("region coordinates must be finite numbers")
    if w <= 0.0 or h <= 0.0 or x < 0.0 or y < 0.0 or x + w > 1.0 or y + h > 1.0:
        raise vol.Invalid(
            "region must have positive size and lie inside [0, 1]"
        )
    return data


def _valid_polygons(value: Any) -> list[list[list[float]]]:
    """Schema validator: a list of rings of [x, y] pairs, all finite
    and inside the unit frame (structural check; even-odd handles any
    geometry beyond that)."""
    try:
        rings = [
            [(float(x), float(y)) for x, y in ring] for ring in value
        ]
    except (TypeError, ValueError) as exc:
        raise vol.Invalid(f"polygons must be [[[x, y], ...], ...]: {exc}")
    for ring in rings:
        for x, y in ring:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise vol.Invalid("polygon coordinates must be finite")
    try:
        validate_rings(rings)
    except ProfileError as exc:
        raise vol.Invalid(str(exc))
    return [[list(v) for v in ring] for ring in rings]


SET_ROI_SCHEMA = vol.Schema(
    vol.All(
        {
            **_ENTRY_SCHEMA,
            # Either a polygon region (bbox derived server-side) or the
            # four rectangle fields; polygons win when both are given.
            vol.Optional("polygons"): _valid_polygons,
            vol.Optional(CONF_ROI_X): vol.Coerce(float),
            vol.Optional(CONF_ROI_Y): vol.Coerce(float),
            vol.Optional(CONF_ROI_W): vol.Coerce(float),
            vol.Optional(CONF_ROI_H): vol.Coerce(float),
            vol.Optional(CONF_WORKING_WIDTH): vol.All(
                vol.Coerce(int), vol.Range(min=1)
            ),
        },
        lambda data: (
            data
            if "polygons" in data
            else _valid_roi_payload_required(data)
        ),
    )
)


def _valid_roi_payload_required(data: dict) -> dict:
    missing = [
        k
        for k in (CONF_ROI_X, CONF_ROI_Y, CONF_ROI_W, CONF_ROI_H)
        if k not in data
    ]
    if missing:
        raise vol.Invalid(
            "either polygons or all four roi_* fields are required"
        )
    return _valid_roi_payload(data)


def _get_entry(hass: HomeAssistant, call: ServiceCall) -> ConfigEntry:
    entries = [
        e
        for e in hass.config_entries.async_entries(DOMAIN)
        if e.state is ConfigEntryState.LOADED
    ]
    entry_id = call.data.get(ATTR_ENTRY_ID)
    if entry_id:
        for entry in entries:
            if entry.entry_id == entry_id:
                return entry
        raise ServiceValidationError(
            f"no loaded {DOMAIN} entry with id {entry_id}"
        )
    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise ServiceValidationError(f"no {DOMAIN} entry is set up and loaded")
    raise ServiceValidationError(
        f"{ATTR_ENTRY_ID} is required when {len(entries)} entries are set up"
    )


def relearn_issue_id(entry: ConfigEntry) -> str:
    """Repairs-issue id flagging a failed relearn for this entry."""
    return f"relearn_failed_{entry.entry_id}"


async def async_relearn(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Recompute the profile from the store; shared by services and setup."""
    runtime = entry.runtime_data
    storage = runtime.storage
    async with runtime.relearn_lock:
        # Deep snapshot: the live store is mutable from concurrent
        # service calls while the executor thread iterates it. The dict
        # round-trip is lossless and hands the learner an isolated copy.
        calibration = store_from_dict(store_to_dict(storage.calibration))
        anchor = store_anchor(hass, entry.entry_id)
        adoption: dict[str, Any] | None = None
        try:
            if has_auto_evidence(calibration):
                # Learn twice: once on the user's own evidence alone,
                # once with the unattended colour samples on top. Both
                # passes see IDENTICAL human labels (auto entries carry
                # none), so any difference is the colour models - and
                # the comparison runs on evidence the collector cannot
                # touch, which is what keeps it from grading its own
                # homework.
                baseline, _base_warnings = await hass.async_add_executor_job(
                    learn_profile, without_auto_evidence(calibration), anchor
                )
                candidate, warnings = await hass.async_add_executor_job(
                    learn_profile, calibration, anchor
                )
                verdict = adoption_verdict(baseline, candidate)
                # Holdout: whatever the run collected, the resulting
                # profile must still reproduce every statement the USER
                # made by hand. That evidence is the one thing an
                # unattended run cannot influence, which is what makes
                # this a check rather than self-assessment.
                manual_only = without_auto_evidence(calibration)
                candidate_mistakes = await hass.async_add_executor_job(
                    misclassified_manual_labels, candidate, manual_only, anchor
                )
                baseline_mistakes = await hass.async_add_executor_job(
                    misclassified_manual_labels, baseline, manual_only, anchor
                )
                # Only NEW disagreements count: a calibration that does
                # not separate already misclassifies some of its own
                # images, and holding that against the collected
                # evidence would make adoption impossible forever.
                known = set(baseline_mistakes)
                mistakes = [m for m in candidate_mistakes if m not in known]
                regressions = [*verdict.regressions, *mistakes]
                adoption = {
                    "adopted": not regressions,
                    "regressions": regressions,
                    "gaps": verdict.gaps,
                }
                verdict = AdoptionVerdict(
                    not regressions, regressions, verdict.gaps
                )
                if verdict.adopt:
                    profile = candidate
                else:
                    # Keep what the human evidence alone says, set the
                    # collected part aside (reversibly) and stop
                    # collecting until the user has looked.
                    profile = baseline
                    storage.calibration.discard_auto_evidence()
                    storage.auto_paused = "; ".join(verdict.regressions)
                    warnings = [
                        *_base_warnings,
                        "what the learning run collected was NOT adopted "
                        "and is set aside; collection stays paused until "
                        "you call restore_auto_evidence (brings it back) "
                        "or start a new run: "
                        + "; ".join(verdict.regressions),
                    ]
            else:
                profile, warnings = await hass.async_add_executor_job(
                    learn_profile, calibration, anchor
                )
        except WastebinError as err:
            # No issue is created here: interactive callers see the
            # error directly, and handle_label even treats it as the
            # normal "not learnable yet" state during first
            # calibration. Only the INVISIBLE background path after a
            # reconfiguration flags it (see __init__).
            raise ServiceValidationError(f"relearn failed: {err}") from err
        # Any successful relearn means detection runs on the current
        # geometry again - a stale-profile flag, wherever it came
        # from, is no longer true.
        ir.async_delete_issue(hass, DOMAIN, relearn_issue_id(entry))
        # The labeled set defines the gates' floor; the unlabeled archive
        # statistics may only widen them (they know the full daily light
        # range, labels usually do not).
        widen_profile_gates(profile, storage.gate_samples)
        storage.profile = profile
        storage.profile_view_epoch = calibration.view_epoch
        await storage.async_save()
    await runtime.coordinator.async_request_refresh()
    trained_ids = {b.id for b in profile.bins}
    return {
        "warnings": warnings,
        **({} if adoption is None else {"auto": adoption}),
        "untrained": sorted(
            set(calibration.active_bin_ids()) - trained_ids
        ),
        "bins": {
            b.id: {
                "min_area_frac": b.min_area_frac,
                **b.learning_stats,
            }
            for b in profile.bins
        },
    }


def async_setup_services(hass: HomeAssistant) -> None:
    """Register the domain services (called from async_setup)."""

    async def handle_capture(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        # Camera/filesystem failures are runtime errors, not user input
        # errors: let the HomeAssistantError propagate as-is.
        filename = await entry.runtime_data.collector.async_capture_now()
        return {ATTR_FILENAME: filename}

    async def handle_add_sample(call: ServiceCall) -> None:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        rect = Rect(*call.data[ATTR_RECT])
        try:
            if call.data[ATTR_SPACE] == "roi":
                rect = roi_rect_to_image_rect(rect, storage.calibration.roi)
            storage.calibration.add_sample(
                call.data[ATTR_FILENAME], call.data[ATTR_BIN], rect
            )
        except CalibrationError as err:
            raise ServiceValidationError(str(err)) from err
        await storage.async_save()

    async def handle_label(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        try:
            storage.calibration.set_labels(
                call.data[ATTR_FILENAME],
                present=call.data[ATTR_PRESENT],
                absent=call.data[ATTR_ABSENT],
            )
        except CalibrationError as err:
            raise ServiceValidationError(str(err)) from err
        await storage.async_save()
        if not call.data[ATTR_AUTO_RELEARN]:
            return {"relearn": "skipped"}
        try:
            result = await async_relearn(hass, entry)
        except ServiceValidationError as err:
            # Labeling must never fail because the set is not learnable
            # yet (e.g. first label, no samples for another bin).
            return {"relearn": f"not possible yet: {err}"}
        return {"relearn": "ok", **result}

    async def handle_start_learning(call: ServiceCall) -> ServiceResponse:
        """Declare the situation and start collecting on its own.

        Everything the run records afterwards rests on this
        declaration, so it is validated like any other label: unknown
        or retired bins and present/absent conflicts fail loudly here
        rather than quietly producing wrong evidence for hours.
        """
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        present = list(call.data[ATTR_PRESENT])
        absent = list(call.data[ATTR_ABSENT])
        store = storage.calibration
        declared = present + absent
        unknown = [b for b in declared if b not in store.bin_ids()]
        if unknown:
            raise ServiceValidationError(
                f"unknown bin ids {unknown}; declared: {store.bin_ids()}"
            )
        retired = [
            b
            for b in declared
            if (decl := store.get_bin(b)) is not None and not decl.active
        ]
        if retired:
            raise ServiceValidationError(
                f"bins {retired} are retired - reactivate them first"
            )
        conflict = sorted(set(present) & set(absent))
        if conflict:
            raise ServiceValidationError(
                f"bins declared both present and absent: {conflict}"
            )
        if not declared:
            raise ServiceValidationError(
                "declare at least one bin as present or absent - the "
                "declaration is what the collected frames will mean"
            )
        missing = [
            b.id for b in store.bins if b.active and b.id not in declared
        ]
        # A run adds one vote per collected frame to the colour pool of
        # every bin declared present. A bin whose existing marks are
        # already near the majority rule will therefore lose its model
        # on the first frame, the adoption test will refuse, and the
        # run will pause before it ever collected anything - which is
        # exactly what happened in the field. Say so BEFORE the run,
        # while re-marking is still cheap.
        fragile = []
        profile = storage.profile
        if profile is not None:
            for model in profile.bins:
                if model.id not in present:
                    continue
                junk = float(model.learning_stats.get("junk_fraction", 0.0))
                if junk > (100.0 - HUE_TOL_PERCENTILE) / 100.0:
                    fragile.append(f"{model.id} ({junk:.0%} junk)")
        storage.learning_declaration = {
            **{b: "present" for b in present},
            **{b: "absent" for b in absent},
        }
        storage.auto_paused = None
        await storage.async_save()
        # The card reads the running run from the status sensor, so it
        # must reflect the new state now, not at the next scan.
        await entry.runtime_data.coordinator.async_request_refresh()
        return {
            "declaration": storage.learning_declaration,
            # Undeclared bins are simply not part of this run: their
            # evidence stays whatever it was.
            "not_declared": missing,
            "warnings": (
                []
                if not fragile
                else [
                    "these bins carry more junk in their existing marks "
                    "than the colour model tolerates, so the first "
                    "collected frame is likely to end the run: "
                    + ", ".join(fragile)
                    + " - re-mark them in softer light first"
                ]
            ),
        }

    async def handle_stop_learning(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        collected = sum(
            len(v)
            for v in reservoir_by_situation(storage.calibration).values()
        )
        storage.learning_declaration = None
        await storage.async_save()
        await entry.runtime_data.coordinator.async_request_refresh()
        return {"collected": collected}

    async def handle_discard_auto(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        touched = storage.calibration.discard_auto_evidence()
        await storage.async_save()
        return {"excluded": touched}

    async def handle_restore_auto(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        touched = storage.calibration.restore_auto_evidence()
        # A bulk restore also brings back everything the reservoir had
        # displaced over the run's history, so the capacity has to be
        # re-established - dropping the most redundant frames first,
        # by the same dispersion rule that admitted them.
        trimmed = over_capacity_paths(storage.calibration)
        for path in trimmed:
            storage.calibration.forget_image(path)
        storage.auto_paused = None
        await storage.async_save()
        return {
            "restored": [p for p in touched if p not in set(trimmed)],
            "still_set_aside": trimmed,
        }

    async def handle_relearn(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        return await async_relearn(hass, entry)

    async def handle_forget(call: ServiceCall) -> None:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        if not storage.calibration.forget_image(call.data[ATTR_FILENAME]):
            raise ServiceValidationError(
                f"no calibration entry for {call.data[ATTR_FILENAME]!r}"
            )
        await storage.async_save()

    async def handle_restore(call: ServiceCall) -> None:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        if not storage.calibration.restore_image(call.data[ATTR_FILENAME]):
            raise ServiceValidationError(
                f"no calibration entry for {call.data[ATTR_FILENAME]!r}"
            )
        await storage.async_save()

    async def handle_reconfirm(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        confirmed: list[str] = []
        skipped: list[str] = []
        unknown: list[str] = []
        for name in call.data[ATTR_FILENAMES]:
            if storage.calibration.get_image(name) is None:
                # A typo must not masquerade as "merely unlabeled".
                unknown.append(name)
            elif storage.calibration.confirm_image_view(name):
                confirmed.append(name)
            else:
                skipped.append(name)
        if confirmed:
            await storage.async_save()
        return {
            "confirmed": confirmed,
            "skipped_unlabeled": skipped,
            "unknown": unknown,
        }

    async def handle_set_roi(call: ServiceCall) -> ServiceResponse:
        """Programmatic ROI edit (the calibration card's write path).

        Runs through entry.data exactly like the reconfigure flow: the
        update listener reloads the entry, reconcile syncs the store
        and the stale profile triggers the background relearn. Stored
        evidence is untouched by design.
        """
        entry = _get_entry(hass, call)
        if "polygons" in call.data:
            rings = clamp_rings(
                [
                    [(float(x), float(y)) for x, y in ring]
                    for ring in call.data["polygons"]
                ]
            )
            try:
                bbox = rings_bbox(rings)
            except ProfileError as err:
                raise ServiceValidationError(str(err)) from err
            # A ring that IS the bbox rectangle normalizes to the rect
            # fast path (None): keeps every pre-polygon fast path and
            # the analytic containment active for effectively
            # rectangular regions.
            if rings_equal(rings, rect_as_rings(bbox)):
                stored_rings = None
            else:
                stored_rings = [[[x, y] for x, y in ring] for ring in rings]
            updates = {
                CONF_ROI_X: bbox.x,
                CONF_ROI_Y: bbox.y,
                CONF_ROI_W: bbox.w,
                CONF_ROI_H: bbox.h,
                CONF_ROI_POLYGONS: stored_rings,
            }
        else:
            updates = {
                CONF_ROI_X: call.data[CONF_ROI_X],
                CONF_ROI_Y: call.data[CONF_ROI_Y],
                CONF_ROI_W: call.data[CONF_ROI_W],
                CONF_ROI_H: call.data[CONF_ROI_H],
                CONF_ROI_POLYGONS: None,
            }
        if CONF_WORKING_WIDTH in call.data:
            updates[CONF_WORKING_WIDTH] = call.data[CONF_WORKING_WIDTH]
        # No-op guard: identical effective configuration must not force
        # a reload (and, for pre-polygon entries, must not rewrite
        # entry.data just to add a roi_polygons: None key).
        if all(
            entry.data.get(k) == v
            for k, v in updates.items()
            if k != CONF_ROI_POLYGONS
        ) and rings_equal(
            entry.data.get(CONF_ROI_POLYGONS),
            updates.get(CONF_ROI_POLYGONS),
        ):
            return {
                "roi": {
                    "x": updates[CONF_ROI_X],
                    "y": updates[CONF_ROI_Y],
                    "w": updates[CONF_ROI_W],
                    "h": updates[CONF_ROI_H],
                },
                "polygons": updates.get(CONF_ROI_POLYGONS),
                "note": "region unchanged; nothing to do",
            }

        def _effective_rings(data: dict):
            rings = data.get(CONF_ROI_POLYGONS)
            if rings is not None:
                return [[(float(x), float(y)) for x, y in r] for r in rings]
            return rect_as_rings(
                Roi(
                    float(data[CONF_ROI_X]),
                    float(data[CONF_ROI_Y]),
                    float(data[CONF_ROI_W]),
                    float(data[CONF_ROI_H]),
                )
            )

        new_rings = _effective_rings(updates)
        for other in hass.config_entries.async_entries(DOMAIN):
            if other.entry_id == entry.entry_id:
                continue
            d = other.data
            if d.get(CONF_CAMERA) == entry.data[CONF_CAMERA] and rings_equal(
                _effective_rings(d), new_rings
            ):
                raise ServiceValidationError(
                    "another entry already watches the same camera and region"
                )
        changed = hass.config_entries.async_update_entry(
            entry, data={**entry.data, **updates}
        )
        return {
            "roi": {
                "x": updates[CONF_ROI_X],
                "y": updates[CONF_ROI_Y],
                "w": updates[CONF_ROI_W],
                "h": updates[CONF_ROI_H],
            },
            "polygons": updates.get(CONF_ROI_POLYGONS),
            "note": (
                "entry reloads now; the profile is relearned under the "
                "new region in the background"
                if changed
                else "region unchanged; nothing to do"
            ),
        }

    async def handle_mark_bin_changed(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        try:
            epoch = storage.calibration.mark_bin_appearance_changed(
                call.data[ATTR_BIN]
            )
        except CalibrationError as err:
            raise ServiceValidationError(str(err)) from err
        await storage.async_save()
        return {
            "bin": call.data[ATTR_BIN],
            "appearance_epoch": epoch,
            "hint": (
                "existing samples and present-labels for this bin are now "
                "dormant (kept in the store); draw new samples on the new "
                "lid, label fresh snapshots, then run relearn"
            ),
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_CAPTURE,
        handle_capture,
        schema=CAPTURE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_ADD_SAMPLE, handle_add_sample, schema=ADD_SAMPLE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LABEL_IMAGE,
        handle_label,
        schema=LABEL_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RELEARN,
        handle_relearn,
        schema=RELEARN_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_START_LEARNING,
        handle_start_learning,
        schema=START_LEARNING_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_STOP_LEARNING,
        handle_stop_learning,
        schema=RELEARN_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DISCARD_AUTO,
        handle_discard_auto,
        schema=RELEARN_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RESTORE_AUTO,
        handle_restore_auto,
        schema=RELEARN_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_FORGET_IMAGE, handle_forget, schema=FILENAME_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_RESTORE_IMAGE, handle_restore, schema=FILENAME_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RECONFIRM_IMAGES,
        handle_reconfirm,
        schema=RECONFIRM_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_ROI,
        handle_set_roi,
        schema=SET_ROI_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MARK_BIN_CHANGED,
        handle_mark_bin_changed,
        schema=MARK_BIN_CHANGED_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
