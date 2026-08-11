"""Wastebin AI Detector: camera-based waste-bin presence detection.

Phase 2 wiring: config flow, one binary_sensor per bin, a learning-mode
switch with an internal daylight snapshot collector, and calibration
services (capture / add_sample / label_image / relearn). The detection
core in ``core/`` stays HA-free and is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .coordinator import LearningCollector, WastebinCoordinator
from .services import async_setup_services
from .storage import WastebinStorage, archive_dir

PLATFORMS = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.SWITCH]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the domain services (independent of config entries)."""
    async_setup_services(hass)
    return True


@dataclass
class WastebinRuntime:
    """Per-entry runtime objects."""

    storage: WastebinStorage
    coordinator: WastebinCoordinator
    collector: LearningCollector


type WastebinConfigEntry = ConfigEntry[WastebinRuntime]


async def async_setup_entry(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> bool:
    storage = WastebinStorage(hass, entry)
    await storage.async_load()
    target = archive_dir(hass, entry.entry_id)
    await hass.async_add_executor_job(
        lambda: target.mkdir(parents=True, exist_ok=True)
    )

    coordinator = WastebinCoordinator(hass, entry, storage)
    # A plain refresh, not first_refresh: before the first calibration
    # there is no profile yet, and that must not block the setup. The
    # entities simply stay unavailable until relearn succeeds.
    await coordinator.async_refresh()

    collector = LearningCollector(hass, entry, storage)
    collector.async_start()

    entry.runtime_data = WastebinRuntime(
        storage=storage, coordinator=coordinator, collector=collector
    )
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    entry.async_on_unload(collector.async_stop)
    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    # Remove the persisted calibration state. Archived snapshot files
    # are intentionally kept: they are user data (photos) and deleting
    # them silently would be destructive.
    await WastebinStorage(hass, entry).async_remove()
