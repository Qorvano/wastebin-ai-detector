"""Home Assistant layer: config flow, setup, calibration services.

These tests run only when pytest-homeassistant-custom-component is
installed; the pure-core suite stays independent of Home Assistant.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("pytest_homeassistant_custom_component")

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from scenes import YELLOW, make_scene
from custom_components.wastebin_ai_detector.const import (
    CONF_BINS,
    CONF_CAMERA,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_WORKING_WIDTH,
    DOMAIN,
)

ENTRY_DATA = {
    CONF_CAMERA: "camera.hinterhof",
    CONF_ROI_X: 0.0,
    CONF_ROI_Y: 0.0,
    CONF_ROI_W: 1.0,
    CONF_ROI_H: 1.0,
    CONF_WORKING_WIDTH: 160,
    CONF_BINS: [{"id": "gelbe_tonne", "name": "Gelbe Tonne"}],
}


# Saturated backdrop covering the whole frame: keeps the median
# saturation high in every test scene, so the greyscale gate stays
# quiet and the overexposure gate can be tested in isolation.
_BACKDROP = ((0.45, 0.55, 0.35), 0.0, 0.0, 1.0, 1.0)


def _scene_jpeg(
    with_yellow: bool, blown: bool = False, yellow_scale: float = 1.0
) -> bytes:
    rects = [_BACKDROP]
    if with_yellow:
        rects.append(
            (YELLOW, 0.30, 0.30, 0.20 * yellow_scale, 0.20 * yellow_scale)
        )
    if blown:
        # Small enough that the median saturation stays on the backdrop
        # (pure overexposure, no greyscale co-trigger), large enough to
        # clip well beyond any learned limit.
        rects.append(((1.0, 1.0, 1.0), 0.05, 0.62, 0.45, 0.20))
    scene = make_scene(size=(320, 200), rects=rects, seed=5)
    buffer = io.BytesIO()
    scene.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _CameraFeed:
    """Mutable stand-in for async_get_image in tests."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(content=self.content)


async def _calibrate_yellow(hass: HomeAssistant) -> str:
    response = await hass.services.async_call(
        DOMAIN, "capture_snapshot", {}, blocking=True, return_response=True
    )
    filename = response["filename"]
    await hass.services.async_call(
        DOMAIN,
        "add_sample",
        {
            "filename": filename,
            "bin": "gelbe_tonne",
            "rect": [0.35, 0.35, 0.10, 0.10],
            "space": "image",
        },
        blocking=True,
    )
    await hass.services.async_call(
        DOMAIN,
        "label_image",
        {"filename": filename, "present": ["gelbe_tonne"]},
        blocking=True,
        return_response=True,
    )
    await hass.async_block_till_done()
    return filename


async def test_full_config_flow(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CAMERA: "camera.hinterhof"}
    )
    assert result["step_id"] == "area"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_ROI_X: 0.2,
            CONF_ROI_Y: 0.2,
            CONF_ROI_W: 0.6,
            CONF_ROI_H: 0.6,
            CONF_WORKING_WIDTH: 480,
        },
    )
    assert result["step_id"] == "bin"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"name": "Gelbe Tonne", "add_another": True}
    )
    assert result["step_id"] == "bin"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"name": "Blaue Tonne", "add_another": False}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_BINS] == [
        {"id": "gelbe_tonne", "name": "Gelbe Tonne"},
        {"id": "blaue_tonne", "name": "Blaue Tonne"},
    ]


async def test_config_flow_rejects_bad_roi(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_CAMERA: "camera.hinterhof"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_ROI_X: 0.8,
            CONF_ROI_Y: 0.0,
            CONF_ROI_W: 0.6,
            CONF_ROI_H: 0.5,
            CONF_WORKING_WIDTH: 480,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_roi"}


async def test_setup_before_calibration(hass: HomeAssistant) -> None:
    """Entities exist but presence is unavailable until first relearn."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    registry = er.async_get(hass)
    sensor_id = registry.async_get_entity_id(
        "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
    )
    switch_id = registry.async_get_entity_id(
        "switch", DOMAIN, f"{entry.entry_id}_learning"
    )
    status_id = registry.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_status"
    )
    assert sensor_id and switch_id and status_id
    assert hass.states.get(sensor_id).state == "unavailable"
    assert hass.states.get(switch_id).state == "on"
    # The status sensor must explain WHY presence is unavailable.
    assert hass.states.get(status_id).state == "not_calibrated"


async def test_calibration_services_full_cycle(hass: HomeAssistant) -> None:
    """capture -> add_sample -> label -> auto relearn -> sensor is on."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)

    image = SimpleNamespace(content=_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        return_value=image,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        response = await hass.services.async_call(
            DOMAIN,
            "capture_snapshot",
            {},
            blocking=True,
            return_response=True,
        )
        filename = response["filename"]

        await hass.services.async_call(
            DOMAIN,
            "add_sample",
            {
                "filename": filename,
                "bin": "gelbe_tonne",
                "rect": [0.35, 0.35, 0.10, 0.10],
                "space": "image",
            },
            blocking=True,
        )
        response = await hass.services.async_call(
            DOMAIN,
            "label_image",
            {"filename": filename, "present": ["gelbe_tonne"]},
            blocking=True,
            return_response=True,
        )
        assert response["relearn"] == "ok"
        assert response["bins"]["gelbe_tonne"]["provisional"] is True
        await hass.async_block_till_done()

        registry = er.async_get(hass)
        sensor_id = registry.async_get_entity_id(
            "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
        )
        state = hass.states.get(sensor_id)
        assert state.state == "on"
        assert state.attributes["margin"] > 1.0


async def test_overexposed_frame_holds_state(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)

        registry = er.async_get(hass)
        sensor_id = registry.async_get_entity_id(
            "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
        )
        assert hass.states.get(sensor_id).state == "on"

        # A blown-out frame must not flip the sensor to off.
        feed.content = _scene_jpeg(with_yellow=False, blown=True)
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(sensor_id).state == "on"
        assert entry.runtime_data.coordinator.last_overexposure_skip is not None

        # ... and the status sensor names the reason with its numbers.
        status_id = registry.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_status"
        )
        status = hass.states.get(status_id)
        assert status.state.startswith("hold_")
        assert "overexposure" in status.state
        assert status.attributes["clip_frac"] > status.attributes[
            "limit_overexposure_clip_max"
        ]
        assert status.attributes["held_previous_state"] is True


async def test_confirm_scans_requires_consecutive_evidence(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=ENTRY_DATA,
        options={"confirm_scans": 2},
        title="Test",
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)

        registry = er.async_get(hass)
        sensor_id = registry.async_get_entity_id(
            "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
        )
        assert hass.states.get(sensor_id).state == "on"

        # Confident absence, first analysis: pending, state held.
        feed.content = _scene_jpeg(with_yellow=False)
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(sensor_id).state == "on"

        # Second consecutive confident absence: flip accepted.
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(sensor_id).state == "off"


async def test_reload_on_ambiguous_frame_stays_unavailable(
    hass: HomeAssistant,
) -> None:
    """An uncertain first frame after a reload must not publish a raw
    verdict; the sensor stays unavailable until a confident frame."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        filename = await _calibrate_yellow(hass)
        # A real absent example gives the profile an observed negative,
        # which activates the ambiguity interval.
        feed.content = _scene_jpeg(with_yellow=False)
        response = await hass.services.async_call(
            DOMAIN, "capture_snapshot", {}, blocking=True, return_response=True
        )
        response = await hass.services.async_call(
            DOMAIN,
            "label_image",
            {"filename": response["filename"], "absent": ["gelbe_tonne"]},
            blocking=True,
            return_response=True,
        )
        assert response["relearn"] == "ok", response
        assert response["bins"]["gelbe_tonne"]["max_neg_area_frac"] == 0.0
        await hass.async_block_till_done()

        registry = er.async_get(hass)
        sensor_id = registry.async_get_entity_id(
            "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
        )

        # Reload with a shrunken (ambiguous) lid in view.
        feed.content = _scene_jpeg(with_yellow=True, yellow_scale=0.6)
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert hass.states.get(sensor_id).state == "unavailable"

        # First confident frame brings the sensor back.
        feed.content = _scene_jpeg(with_yellow=True)
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(sensor_id).state == "on"


async def test_gates_self_heal_after_one_scan(hass: HomeAssistant) -> None:
    """First encounter with harsher light holds one scan; the frame's
    own statistics widen the gates, the next scan is admitted."""
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ), patch(
        "custom_components.wastebin_ai_detector.coordinator.is_up",
        return_value=True,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)

        registry = er.async_get(hass)
        sensor_id = registry.async_get_entity_id(
            "binary_sensor", DOMAIN, f"{entry.entry_id}_gelbe_tonne"
        )
        status_id = registry.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_status"
        )
        assert hass.states.get(sensor_id).state == "on"

        # Harsher light than calibrated (a clipped strip away from the
        # lid): scan 1 holds ...
        feed.content = _scene_jpeg(with_yellow=True, blown=True)
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(status_id).state.startswith("hold_")
        assert hass.states.get(sensor_id).state == "on"

        # ... scan 2 with the same light is admitted and analyzed.
        await entry.runtime_data.coordinator.async_refresh()
        await hass.async_block_till_done()
        assert hass.states.get(status_id).state == "ok"
        assert hass.states.get(sensor_id).state == "on"


async def test_watchdog_aborts_hanging_analysis(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)

    import time as time_mod
    from datetime import timedelta

    release = asyncio.Event()

    def blocking_detect(*_args, **_kwargs):
        # Hang the executor phase, the one the watchdog actually
        # bounds in production; release at test end to free the thread.
        while not release.is_set():
            time_mod.sleep(0.05)
        raise RuntimeError("released")

    coordinator = entry.runtime_data.coordinator
    coordinator.update_interval = timedelta(seconds=0.3)
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ), patch.object(
        type(coordinator), "_detect_bytes", staticmethod(blocking_detect)
    ):
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        assert coordinator.last_update_success is False
        assert coordinator.diagnostics["outcome"] == "watchdog_timeout"

        # The leaked executor thread triggers the single-flight guard.
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        assert coordinator.diagnostics["outcome"] == "previous_run_still_busy"
    release.set()
    await hass.async_block_till_done()


async def test_forget_image_removes_calibration_data(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    image = SimpleNamespace(content=_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        return_value=image,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        response = await hass.services.async_call(
            DOMAIN, "capture_snapshot", {}, blocking=True, return_response=True
        )
        filename = response["filename"]
        await hass.services.async_call(
            DOMAIN,
            "add_sample",
            {
                "filename": filename,
                "bin": "gelbe_tonne",
                "rect": [0.35, 0.35, 0.10, 0.10],
                "space": "image",
            },
            blocking=True,
        )
        await hass.services.async_call(
            DOMAIN, "forget_image", {"filename": filename}, blocking=True
        )
        # Soft semantics: nothing is deleted, the entry is excluded
        # from learning and can be restored.
        entry_after = entry.runtime_data.storage.calibration.get_image(
            filename
        )
        assert entry_after is not None and entry_after.excluded is True
        assert entry_after.samples["gelbe_tonne"]
        await hass.services.async_call(
            DOMAIN, "restore_image", {"filename": filename}, blocking=True
        )
        assert (
            entry.runtime_data.storage.calibration.get_image(filename).excluded
            is False
        )


async def test_relearn_without_data_fails_cleanly(hass: HomeAssistant) -> None:
    from homeassistant.exceptions import ServiceValidationError

    entry = MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA, title="Test")
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "relearn", {}, blocking=True, return_response=True
        )


ENTRY_DATA_TWO_BINS = {
    **ENTRY_DATA,
    CONF_BINS: [
        {"id": "gelbe_tonne", "name": "Gelbe Tonne", "active": True},
        {"id": "blaue_tonne", "name": "Blaue Tonne", "active": True},
    ],
}


async def _start_reconfigure(hass: HomeAssistant, entry) -> dict:
    return await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "reconfigure", "entry_id": entry.entry_id},
    )


async def test_reconfigure_area_updates_store_and_relearns(
    hass: HomeAssistant,
) -> None:
    """ROI edit: entry.data -> store reconcile, samples survive, the
    profile is recomputed under the new region in the background."""
    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)
        storage = entry.runtime_data.storage
        sample_before = storage.calibration.images[0].samples["gelbe_tonne"][0]
        profile_roi_before = storage.profile.roi

        result = await _start_reconfigure(hass, entry)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "reconf_area"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ROI_X: 0.1,
                CONF_ROI_Y: 0.1,
                CONF_ROI_W: 0.8,
                CONF_ROI_H: 0.8,
                CONF_WORKING_WIDTH: 160,
            },
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"
        await hass.async_block_till_done(wait_background_tasks=True)

        storage = entry.runtime_data.storage
        assert storage.calibration.roi.x == pytest.approx(0.1)
        # The sample rect is image-anchored and untouched by the edit.
        entry_after = storage.calibration.images[0]
        assert entry_after.samples["gelbe_tonne"][0] == sample_before
        # Background relearn recomputed the profile under the new ROI.
        assert storage.profile.roi != profile_roi_before
        assert storage.profile.roi.x == pytest.approx(0.1)
        state = hass.states.get(
            "binary_sensor.test_gelbe_tonne"
        )
        assert state is not None and state.state == "on"


async def test_reconfigure_retire_and_reactivate_bin(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=ENTRY_DATA_TWO_BINS,
        title="Test",
        minor_version=2,
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)
        assert hass.states.get("binary_sensor.test_blaue_tonne") is not None

        result = await _start_reconfigure(hass, entry)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "reconf_retire_bin"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"bin": "blaue_tonne"}
        )
        assert result["reason"] == "reconfigure_successful"
        await hass.async_block_till_done()

        # Entity gone, evidence and declaration kept.
        registry = er.async_get(hass)
        assert (
            registry.async_get_entity_id(
                "binary_sensor", DOMAIN, f"{entry.entry_id}_blaue_tonne"
            )
            is None
        )
        storage = entry.runtime_data.storage
        blau = storage.calibration.get_bin("blaue_tonne")
        assert blau is not None and blau.active is False
        # The yellow sensor keeps working.
        assert hass.states.get("binary_sensor.test_gelbe_tonne").state == "on"

        result = await _start_reconfigure(hass, entry)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "reconf_reactivate_bin"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"bin": "blaue_tonne"}
        )
        assert result["reason"] == "reconfigure_successful"
        await hass.async_block_till_done()
        assert entry.runtime_data.storage.calibration.get_bin(
            "blaue_tonne"
        ).active
        assert hass.states.get("binary_sensor.test_blaue_tonne") is not None


async def test_reconfigure_add_bin_does_not_break_relearn(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await _calibrate_yellow(hass)

        result = await _start_reconfigure(hass, entry)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "reconf_add_bin"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"name": "Schwarze Tonne"}
        )
        assert result["reason"] == "reconfigure_successful"
        await hass.async_block_till_done()

        # The new bin has no samples: its sensor exists but is
        # unavailable; relearn still works for the calibrated bin.
        assert (
            hass.states.get("binary_sensor.test_schwarze_tonne").state
            == "unavailable"
        )
        response = await hass.services.async_call(
            DOMAIN, "relearn", {}, blocking=True, return_response=True
        )
        assert "schwarze_tonne" in response["untrained"]
        assert hass.states.get("binary_sensor.test_gelbe_tonne").state == "on"


async def test_v1_storage_data_migrates_with_backup(
    hass: HomeAssistant, tmp_path
) -> None:
    """A stored v1 calibration dict loads via migration; the original
    is backed up next to the images before the first migrated save."""
    from custom_components.wastebin_ai_detector.storage import (
        V1_BACKUP_NAME,
        archive_dir,
    )

    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    v1_calibration = {
        "schema_version": 1,
        "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "working_width": 160,
        "resample": "bilinear",
        "bins": [{"id": "gelbe_tonne", "name": "Gelbe Tonne"}],
        "images": [
            {
                "path": "old.jpg",
                "samples": {
                    "gelbe_tonne": [{"x": 0.35, "y": 0.35, "w": 0.1, "h": 0.1}]
                },
                "present": ["gelbe_tonne"],
                "absent": [],
            }
        ],
    }
    hass_storage_key = f"{DOMAIN}.{entry.entry_id}"
    from homeassistant.helpers.storage import Store

    store = Store(hass, 1, hass_storage_key)
    await store.async_save(
        {
            "calibration": v1_calibration,
            "profile": None,
            "learning": True,
            "gate_samples": [],
        }
    )
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    calibration = entry.runtime_data.storage.calibration
    assert calibration.schema_version == 2
    sample = calibration.get_image("old.jpg").samples["gelbe_tonne"][0]
    assert sample.rect.x == pytest.approx(0.35)
    assert calibration.get_image("old.jpg").label_roi is not None
    backup = archive_dir(hass, entry.entry_id) / V1_BACKUP_NAME
    assert backup.exists()


async def test_reconfirm_images_service(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        filename = await _calibrate_yellow(hass)
        storage = entry.runtime_data.storage
        # Simulate a view bump: labels go stale, reconfirm revives them.
        storage.calibration.bump_view_epoch([])
        response = await hass.services.async_call(
            DOMAIN,
            "reconfirm_images",
            {"filenames": [filename, "does_not_exist.jpg"]},
            blocking=True,
            return_response=True,
        )
        assert response["confirmed"] == [filename]
        assert response["skipped_unlabeled"] == []
        assert response["unknown"] == ["does_not_exist.jpg"]
        assert (
            storage.calibration.get_image(filename).view_epoch
            == storage.calibration.view_epoch
        )


async def test_reconf_view_changed_end_to_end(hass: HomeAssistant) -> None:
    """The 'camera re-aimed' flow must bump the store's view epoch via
    reconcile, clear the gate samples, keep non-snapshot files out of
    capture_epochs, and set aside the old area evidence."""
    from custom_components.wastebin_ai_detector.storage import (
        STORE_MIRROR_NAME,
        archive_dir,
    )

    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        filename = await _calibrate_yellow(hass)
        storage = entry.runtime_data.storage
        storage.gate_samples = [[0.4, 0.5, 0.01]]
        assert storage.calibration.view_epoch == 0
        # The mirror exists after the first save and must NOT be
        # treated as an unmaterialized snapshot below.
        assert (archive_dir(hass, entry.entry_id) / STORE_MIRROR_NAME).exists()

        result = await _start_reconfigure(hass, entry)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": "reconf_view_changed"}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {}
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"
        await hass.async_block_till_done()

        storage = entry.runtime_data.storage
        store = storage.calibration
        assert store.view_epoch == 1
        assert storage.gate_samples == []
        assert STORE_MIRROR_NAME not in store.capture_epochs
        # The labeled image was materialized before the bump: its own
        # epoch stays 0 and its labels are now set aside by the view.
        assert store.get_image(filename).view_epoch == 0
        from custom_components.wastebin_ai_detector.core import (
            learning_view,
        )

        view, _warnings = learning_view(store)
        assert view.get_image(filename).present == []
        # The profile is scene-stale now and marked as such.
        from custom_components.wastebin_ai_detector import _profile_stale

        assert _profile_stale(storage) is True


async def test_closed_storage_never_writes_and_removal_exports(
    hass: HomeAssistant,
) -> None:
    from custom_components.wastebin_ai_detector.storage import (
        STORE_MIRROR_NAME,
        archive_dir,
    )

    entry = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, title="Test", minor_version=2
    )
    entry.add_to_hass(hass)
    feed = _CameraFeed(_scene_jpeg(with_yellow=True))
    with patch(
        "custom_components.wastebin_ai_detector.coordinator.async_get_image",
        new=feed,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        filename = await _calibrate_yellow(hass)
        storage = entry.runtime_data.storage

        # A fenced (superseded) instance must not write: poison the
        # in-memory store, mark closed, save - the persisted state must
        # keep the calibration entry.
        storage.mark_closed()
        storage.calibration.images.clear()
        await storage.async_save()
        from homeassistant.helpers.storage import Store

        raw = await Store(
            hass, 1, f"{DOMAIN}.{entry.entry_id}"
        ).async_load()
        assert raw["calibration"]["images"], "fenced write must be discarded"

        # Removal exports the store next to the images first.
        assert await hass.config_entries.async_remove(entry.entry_id)
        await hass.async_block_till_done()
        mirror = archive_dir(hass, entry.entry_id) / STORE_MIRROR_NAME
        assert mirror.exists()
        import json as _json

        exported = _json.loads(mirror.read_text(encoding="utf-8"))
        assert any(e["path"] == filename for e in exported["images"])
