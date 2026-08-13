"""Wastebin AI Detector: camera-based waste-bin presence detection.

Phase 2 wiring: config flow, one binary_sensor per bin, a learning-mode
switch with an internal daylight snapshot collector, and calibration
services (capture / add_sample / label_image / relearn). The detection
core in ``core/`` stays HA-free and is reused unchanged.

Phase 2.3: entry.data is the authority for the ACTIVE configuration
(camera, ROI, bin lifecycle) and is reconciled into the calibration
store on every setup; the store keeps full history (retired bins,
dormant evidence) and is never truncated by reconfiguration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path as _Path

from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_integration

from .const import CONF_BIN_ACTIVE, CONF_BINS, DOMAIN
from .coordinator import LearningCollector, WastebinCoordinator
from .core import rings_equal, roi_equal, store_to_dict
from .services import async_relearn, async_setup_services
from .storage import (
    WastebinStorage,
    archive_dir,
    reconcile_store_with_entry,
    store_anchor,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.SENSOR,
    Platform.SWITCH,
]


# The calibration card ships with the integration and is registered as
# a frontend module automatically - no manual resource step for users.
CARD_URL = f"/{DOMAIN}-card/wastebin-calibration-card.js"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the domain services and the bundled calibration card."""
    async_setup_services(hass)
    await hass.http.async_register_static_paths(
        [
            StaticPathConfig(
                CARD_URL,
                str(_Path(__file__).parent / "www" / "wastebin-calibration-card.js"),
                True,
            )
        ]
    )
    if "frontend" in hass.config.components:
        # Every real installation has the frontend; the guard only
        # matters for headless test harnesses without hass_frontend.
        # The manifest version busts the browser cache on updates (a
        # bare URL would be served from heuristic cache for days), so
        # long cache headers above are safe.
        integration = await async_get_integration(hass, DOMAIN)
        add_extra_js_url(hass, f"{CARD_URL}?v={integration.version}")
    return True


@dataclass
class WastebinRuntime:
    """Per-entry runtime objects."""

    storage: WastebinStorage
    coordinator: WastebinCoordinator
    collector: LearningCollector
    # Serializes relearns and lets unload wait out an in-flight one, so
    # a superseded storage instance can never write stale state back.
    relearn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


type WastebinConfigEntry = ConfigEntry[WastebinRuntime]


async def async_migrate_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> bool:
    """Config-entry migration (minor bumps are downgrade-safe)."""
    if entry.version > 1:
        return False
    if entry.minor_version < 2:
        bins = [
            {**b, CONF_BIN_ACTIVE: bool(b.get(CONF_BIN_ACTIVE, True))}
            for b in entry.data[CONF_BINS]
        ]
        hass.config_entries.async_update_entry(
            entry, data={**entry.data, CONF_BINS: bins}, minor_version=2
        )
    return True


# The collector writes exactly this suffix (coordinator snapshot
# archiving); everything else in the folder (store.json mirror, v1
# backup, stray OS files) is not a camera capture and must never be
# stamped into capture_epochs.
SNAPSHOT_SUFFIX = ".jpg"


def _list_archive_filenames(target) -> list[str]:
    if not target.is_dir():
        return []
    return [
        p.name
        for p in target.iterdir()
        if p.is_file() and p.suffix.lower() == SNAPSHOT_SUFFIX
    ]


async def async_setup_entry(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> bool:
    storage = WastebinStorage(hass, entry)
    await storage.async_load()
    target = archive_dir(hass, entry.entry_id)
    await hass.async_add_executor_job(
        lambda: target.mkdir(parents=True, exist_ok=True)
    )

    # entry.data -> store: ROI/working-width updates, view-generation
    # bumps, bin lifecycle flags. Never deletes evidence.
    known = {e.path for e in storage.calibration.images}
    archive_names = await hass.async_add_executor_job(
        _list_archive_filenames, target
    )
    unmaterialized = [n for n in archive_names if n not in known]
    if reconcile_store_with_entry(storage, entry, unmaterialized):
        await storage.async_save()

    _prune_retired_entities(hass, entry)

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

    if _profile_stale(storage):
        # The old profile keeps detecting on its own (consistent old)
        # geometry until this succeeds; a failed relearn is reported,
        # never a half-migrated profile.
        entry.async_create_background_task(
            hass, _async_relearn_after_reconfigure(hass, entry), "relearn"
        )
    return True


def _profile_stale(storage: WastebinStorage) -> bool:
    profile = storage.profile
    if profile is None:
        return False
    store = storage.calibration
    if not roi_equal(profile.roi, store.roi):
        return True
    if not rings_equal(profile.roi_polygons, store.roi_polygons):
        return True
    if profile.working_width != store.working_width:
        return True
    if storage.profile_view_epoch != store.view_epoch:
        # Learned under a different scene mapping: the relearn attempt
        # below will fail loudly until fresh labels exist, which is the
        # visible signal the silent-stale case was missing.
        return True
    active = set(store.active_bin_ids())
    # A retired bin still present in the profile must be relearned
    # away; a NEW active bin without samples is expected to be missing
    # (it trains after its first samples), so it does not count.
    return bool({b.id for b in profile.bins} - active)


async def _async_relearn_after_reconfigure(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> None:
    try:
        result = await async_relearn(hass, entry)
    except ServiceValidationError as err:
        _LOGGER.warning(
            "relearn after reconfiguration failed; the previous profile "
            "stays active on its old geometry: %s",
            err,
        )
        return
    if result["warnings"]:
        _LOGGER.warning(
            "relearn after reconfiguration finished with warnings: %s",
            "; ".join(result["warnings"]),
        )


def _prune_retired_entities(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> None:
    """Remove registry entries of retired bins (exact unique_id match).

    Documented tradeoff: pruning drops per-entity customizations, but a
    permanently unavailable ghost entity would be worse; the stable
    unique_id guarantees the identity returns on reactivation.
    """
    retired = {
        f"{entry.entry_id}_{b['id']}"
        for b in entry.data[CONF_BINS]
        if not b.get(CONF_BIN_ACTIVE, True)
    }
    if not retired:
        return
    registry = er.async_get(hass)
    for reg_entry in er.async_entries_for_config_entry(
        registry, entry.entry_id
    ):
        # Domain restriction: the helper entities (switch/sensor/button)
        # share the unique_id namespace, so a bin slugged e.g.
        # "learning" must only ever prune its own binary_sensor.
        if reg_entry.domain != Platform.BINARY_SENSOR:
            continue
        if reg_entry.unique_id in retired:
            _LOGGER.info(
                "removing entity %s of retired bin (all training data "
                "is kept; reactivating the bin restores the entity)",
                reg_entry.entity_id,
            )
            registry.async_remove(reg_entry.entity_id)


async def _async_update_listener(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: WastebinConfigEntry
) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    )
    if unloaded:
        runtime = entry.runtime_data
        # Wait out an in-flight relearn, then fence this storage
        # instance: a reload creates a fresh instance on the same store
        # key, and a stale last-write-wins save must never overwrite
        # the reconciled state.
        async with runtime.relearn_lock:
            runtime.storage.mark_closed()
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Entry removal must not destroy evidence.

    The calibration store (labels, samples, epochs) is exported next to
    the archived images before the HA-side storage is removed; the
    archive folder is then fully self-contained (usable with the
    offline CLI, re-importable later). The snapshot files themselves
    are user data and are never deleted.
    """
    storage = WastebinStorage(hass, entry)
    await storage.async_load()
    if storage.calibration.images:
        anchor = store_anchor(hass, entry.entry_id)
        payload = store_to_dict(storage.calibration)

        def _export() -> None:
            import json

            anchor.parent.mkdir(parents=True, exist_ok=True)
            anchor.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        await hass.async_add_executor_job(_export)
        _LOGGER.warning(
            "calibration data exported to %s before entry removal; the "
            "archive folder is self-contained and can be re-used",
            anchor,
        )
    await storage.async_remove()
