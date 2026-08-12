"""Domain services: capture, sample, label, relearn.

These are the scriptable calibration API. The upcoming calibration
card is a thin UI over exactly these services, so everything the card
will do can already be done today from Developer Tools or automations.
"""

from __future__ import annotations

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

from .const import (
    ATTR_ABSENT,
    ATTR_AUTO_RELEARN,
    ATTR_BIN,
    ATTR_ENTRY_ID,
    ATTR_FILENAME,
    ATTR_PRESENT,
    ATTR_RECT,
    ATTR_SPACE,
    DOMAIN,
    SERVICE_ADD_SAMPLE,
    SERVICE_CAPTURE,
    SERVICE_FORGET_IMAGE,
    SERVICE_LABEL_IMAGE,
    SERVICE_RELEARN,
)
from pathlib import Path

from .core import (
    CalibrationError,
    Rect,
    WastebinError,
    learn_profile,
    store_from_dict,
    store_to_dict,
)
from .storage import store_anchor, widen_profile_gates

_ENTRY_SCHEMA = {vol.Optional(ATTR_ENTRY_ID): cv.string}

CAPTURE_SCHEMA = vol.Schema(_ENTRY_SCHEMA)

ADD_SAMPLE_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): cv.string,
        vol.Required(ATTR_BIN): cv.string,
        vol.Required(ATTR_RECT): vol.All(
            [vol.Coerce(float)], vol.Length(min=4, max=4)
        ),
        vol.Optional(ATTR_SPACE, default="image"): vol.In(("image", "roi")),
    }
)

LABEL_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): cv.string,
        vol.Optional(ATTR_PRESENT, default=[]): [cv.string],
        vol.Optional(ATTR_ABSENT, default=[]): [cv.string],
        vol.Optional(ATTR_AUTO_RELEARN, default=True): cv.boolean,
    }
)

RELEARN_SCHEMA = vol.Schema(_ENTRY_SCHEMA)

FORGET_SCHEMA = vol.Schema(
    {
        **_ENTRY_SCHEMA,
        vol.Required(ATTR_FILENAME): cv.string,
    }
)


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


def _validated_filename(filename: str) -> str:
    """Reject anything that is not a plain archive file name.

    Service input reaches filesystem paths via the calibration store;
    a path component or absolute path would escape the archive.
    """
    if (
        not filename
        or Path(filename).name != filename
        or filename in (".", "..")
    ):
        raise ServiceValidationError(
            f"filename must be a plain file name from the calibration "
            f"archive, got {filename!r}"
        )
    return filename


async def _async_relearn(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    runtime = entry.runtime_data
    storage = runtime.storage
    # Deep snapshot: the live store is mutable from concurrent service
    # calls while the executor thread iterates it. The dict round-trip
    # is lossless and hands the learner an isolated copy.
    calibration = store_from_dict(store_to_dict(storage.calibration))
    try:
        profile, warnings = await hass.async_add_executor_job(
            learn_profile,
            calibration,
            store_anchor(hass, entry.entry_id),
        )
    except WastebinError as err:
        raise ServiceValidationError(f"relearn failed: {err}") from err
    # The labeled set defines the gates' floor; the unlabeled archive
    # statistics may only widen them (they know the full daily light
    # range, labels usually do not).
    widen_profile_gates(profile, storage.gate_samples)
    storage.profile = profile
    await storage.async_save()
    await runtime.coordinator.async_request_refresh()
    return {
        "warnings": warnings,
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
        filename = _validated_filename(call.data[ATTR_FILENAME])
        rect = Rect(*call.data[ATTR_RECT])
        try:
            if call.data[ATTR_SPACE] == "image":
                rect = storage.calibration.image_rect_to_roi_rect(rect)
            storage.calibration.add_sample(
                filename, call.data[ATTR_BIN], rect
            )
        except CalibrationError as err:
            raise ServiceValidationError(str(err)) from err
        await storage.async_save()

    async def handle_label(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        filename = _validated_filename(call.data[ATTR_FILENAME])
        try:
            storage.calibration.set_labels(
                filename,
                present=call.data[ATTR_PRESENT],
                absent=call.data[ATTR_ABSENT],
            )
        except CalibrationError as err:
            raise ServiceValidationError(str(err)) from err
        await storage.async_save()
        if not call.data[ATTR_AUTO_RELEARN]:
            return {"relearn": "skipped"}
        try:
            result = await _async_relearn(hass, entry)
        except ServiceValidationError as err:
            # Labeling must never fail because the set is not learnable
            # yet (e.g. first label, no samples for another bin).
            return {"relearn": f"not possible yet: {err}"}
        return {"relearn": "ok", **result}

    async def handle_relearn(call: ServiceCall) -> ServiceResponse:
        entry = _get_entry(hass, call)
        return await _async_relearn(hass, entry)

    async def handle_forget(call: ServiceCall) -> None:
        entry = _get_entry(hass, call)
        storage = entry.runtime_data.storage
        filename = _validated_filename(call.data[ATTR_FILENAME])
        if not storage.calibration.forget_image(filename):
            raise ServiceValidationError(
                f"no calibration entry for {filename!r}"
            )
        await storage.async_save()

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
        DOMAIN, SERVICE_FORGET_IMAGE, handle_forget, schema=FORGET_SCHEMA
    )
