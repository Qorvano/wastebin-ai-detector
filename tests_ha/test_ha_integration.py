"""Home Assistant layer: config flow, setup, calibration services.

These tests run only when pytest-homeassistant-custom-component is
installed; the pure-core suite stays independent of Home Assistant.
"""

from __future__ import annotations

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


def _scene_jpeg(with_yellow: bool) -> bytes:
    rects = [(YELLOW, 0.30, 0.30, 0.20, 0.20)] if with_yellow else []
    scene = make_scene(size=(320, 200), rects=rects, seed=5)
    buffer = io.BytesIO()
    scene.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


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
    assert sensor_id and switch_id
    assert hass.states.get(sensor_id).state == "unavailable"
    assert hass.states.get(switch_id).state == "on"


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
        assert (
            entry.runtime_data.storage.calibration.get_image(filename) is None
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
