"""Always-available status sensor: why the last analysis ended as it did.

The presence sensors go unavailable while quality gates or errors are
active, which is correct but must never be a mystery. This sensor stays
available in every situation and carries the outcome plus the measured
values against the learned limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_ROI_H,
    CONF_ROI_POLYGONS,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    DOMAIN,
    STATUS_OUTCOMES,
)
from .coordinator import WastebinCoordinator

if TYPE_CHECKING:
    from . import WastebinConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: WastebinConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([WastebinStatusSensor(entry.runtime_data.coordinator, entry)])


class WastebinStatusSensor(
    CoordinatorEntity[WastebinCoordinator], SensorEntity
):
    """Outcome of the most recent analysis attempt."""

    _attr_has_entity_name = True
    _attr_translation_key = "status"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:list-status"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = list(STATUS_OUTCOMES)

    def __init__(
        self, coordinator: WastebinCoordinator, entry: WastebinConfigEntry
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Qorvano",
            model="Wastebin AI Detector",
        )

    @property
    def available(self) -> bool:
        # The whole point of this sensor is to explain failures, so it
        # must not share the presence sensors' availability.
        return True

    @property
    def native_value(self) -> str:
        outcome = str(self.coordinator.diagnostics.get("outcome", "no_run_yet"))
        # Enum sensors reject values outside their options; an unknown
        # outcome (future code drift) degrades to the neutral state.
        return outcome if outcome in STATUS_OUTCOMES else "no_run_yet"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attributes = dict(self.coordinator.diagnostics)
        # The configured region (authoritative entry.data): the
        # calibration card prefills its polygon editor from this.
        attributes["region"] = {
            "bbox": {
                "x": self._entry.data[CONF_ROI_X],
                "y": self._entry.data[CONF_ROI_Y],
                "w": self._entry.data[CONF_ROI_W],
                "h": self._entry.data[CONF_ROI_H],
            },
            "polygons": self._entry.data.get(CONF_ROI_POLYGONS),
        }
        return attributes
