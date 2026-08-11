"""One presence binary_sensor per configured bin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_BINS, DOMAIN
from .coordinator import WastebinCoordinator

if TYPE_CHECKING:
    from . import WastebinConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: WastebinConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    async_add_entities(
        WastebinPresenceSensor(coordinator, entry, b["id"], b["name"])
        for b in entry.data[CONF_BINS]
    )


class WastebinPresenceSensor(
    CoordinatorEntity[WastebinCoordinator], BinarySensorEntity
):
    """Is this bin currently visible in the region of interest?"""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: WastebinCoordinator,
        entry: WastebinConfigEntry,
        bin_id: str,
        bin_name: str,
    ) -> None:
        super().__init__(coordinator)
        self._bin_id = bin_id
        self._attr_name = bin_name
        self._attr_unique_id = f"{entry.entry_id}_{bin_id}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Qorvano",
            model="Wastebin AI Detector",
        )

    def _bin_result(self) -> Any | None:
        if self.coordinator.data is None:
            return None
        for result in self.coordinator.data.bins:
            if result.id == self._bin_id:
                return result
        return None

    @property
    def available(self) -> bool:
        return super().available and self._bin_result() is not None

    @property
    def is_on(self) -> bool | None:
        result = self._bin_result()
        return result.present if result else None

    @property
    def icon(self) -> str:
        return "mdi:trash-can" if self.is_on else "mdi:trash-can-outline"

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        result = self._bin_result()
        if result is None:
            return None
        attributes: dict[str, Any] = {
            "area_frac": result.area_frac,
            "min_area_frac": result.min_area_frac,
            "margin": result.margin,
            "median_sat": self.coordinator.data.median_sat,
            "grayscale_suspect": self.coordinator.data.grayscale_suspect,
        }
        profile = self.coordinator.storage.profile
        if profile is not None:
            for model in profile.bins:
                if model.id == self._bin_id:
                    stats = model.learning_stats
                    attributes["provisional"] = stats.get("provisional")
                    attributes["separable"] = stats.get("separable")
                    break
        if self.coordinator.last_daylight_update is not None:
            attributes["last_daylight_update"] = (
                self.coordinator.last_daylight_update.isoformat()
            )
        return attributes
