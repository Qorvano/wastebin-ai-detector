"""Button: trigger a detection run right now."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

if TYPE_CHECKING:
    from . import WastebinConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: WastebinConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([WastebinDetectNowButton(entry)])


class WastebinDetectNowButton(ButtonEntity):
    """Requests an immediate coordinator refresh."""

    _attr_has_entity_name = True
    _attr_translation_key = "detect_now"
    _attr_icon = "mdi:magnify-scan"

    def __init__(self, entry: WastebinConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_detect_now"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Qorvano",
            model="Wastebin AI Detector",
        )

    async def async_press(self) -> None:
        await self._entry.runtime_data.coordinator.async_request_refresh()
