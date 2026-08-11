"""Learning-mode switch: controls the background snapshot collector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.switch import SwitchEntity
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
    async_add_entities([WastebinLearningSwitch(entry)])


class WastebinLearningSwitch(SwitchEntity):
    """On: daylight snapshots are archived as future calibration data."""

    _attr_has_entity_name = True
    _attr_translation_key = "learning"
    _attr_icon = "mdi:school"

    def __init__(self, entry: WastebinConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_learning"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Qorvano",
            model="Wastebin AI Detector",
        )

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.storage.learning

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._async_set(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._async_set(False)

    async def _async_set(self, value: bool) -> None:
        storage = self._entry.runtime_data.storage
        storage.learning = value
        await storage.async_save()
        self.async_write_ha_state()
