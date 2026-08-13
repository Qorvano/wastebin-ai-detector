"""Per-entry persistence and file locations.

Three things live here, per config entry:
- the calibration store (samples + labels), the single source of truth
  for learning,
- the learned profile (a derived artifact, recomputed by relearn),
- the learning flag (whether the background snapshot collector runs).

Snapshots are archived as plain JPEG files in the media directory so
they do not bloat `/config` backups; image paths inside the calibration
store are relative to that archive directory. Every save also mirrors
the calibration store as ``store.json`` into that archive directory,
which keeps the folder self-contained (movable, re-importable, usable
with the offline CLI) even if the HA-side storage is ever lost.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    CONF_BIN_ACTIVE,
    CONF_BINS,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_VIEW_GENERATION,
    CONF_WORKING_WIDTH,
    DOMAIN,
    STORAGE_VERSION,
)
from .core import (
    BinDecl,
    CalibrationStore,
    Profile,
    Roi,
    derive_quality_gates,
    profile_from_dict,
    profile_to_dict,
    roi_equal,
    store_from_dict,
    store_to_dict,
)

_LOGGER = logging.getLogger(__name__)

V1_BACKUP_NAME = "calibration_v1_backup.json"
STORE_MIRROR_NAME = "store.json"


def widen_profile_gates(
    profile: Profile, gate_samples: list[list[float]]
) -> bool:
    """Widen the profile's light gates from unlabeled frame statistics.

    Widen only: labeled calibration stays the floor, the archive can
    only extend what counts as known light. Returns True if anything
    changed.
    """
    gates = derive_quality_gates(gate_samples)
    if gates is None:
        return False
    widened = False
    if gates["daylight_sat_min"] < profile.daylight_sat_min:
        profile.daylight_sat_min = gates["daylight_sat_min"]
        widened = True
    if gates["overexposure_clip_max"] > profile.overexposure_clip_max:
        profile.overexposure_clip_max = gates["overexposure_clip_max"]
        widened = True
    if gates["daylight_val_max"] > profile.daylight_val_max:
        profile.daylight_val_max = gates["daylight_val_max"]
        widened = True
    return widened


def archive_dir(hass: HomeAssistant, entry_id: str) -> Path:
    """Snapshot archive location for one entry.

    Prefers the local media dir (excluded from typical backups); falls
    back to the config dir when no media dir is configured.
    """
    media_dirs = getattr(hass.config, "media_dirs", None) or {}
    if "local" in media_dirs:
        # HA's own default media key; deterministic across restarts.
        base = Path(media_dirs["local"])
    elif media_dirs:
        # No "local" key configured: pick the alphabetically first key
        # (a stable, documented rule instead of dict iteration order).
        base = Path(media_dirs[sorted(media_dirs)[0]])
    else:
        base = Path(hass.config.path("media"))
    return base / DOMAIN / entry_id


def store_anchor(hass: HomeAssistant, entry_id: str) -> Path:
    """File anchor used to resolve relative image paths (also the
    mirror location that makes the archive folder self-contained)."""
    return archive_dir(hass, entry_id) / STORE_MIRROR_NAME


def empty_store_from_entry(entry: ConfigEntry) -> CalibrationStore:
    """Build a fresh calibration store from the config-entry setup data."""
    return CalibrationStore(
        roi=Roi(
            x=float(entry.data[CONF_ROI_X]),
            y=float(entry.data[CONF_ROI_Y]),
            w=float(entry.data[CONF_ROI_W]),
            h=float(entry.data[CONF_ROI_H]),
        ),
        working_width=int(entry.data[CONF_WORKING_WIDTH]),
        resample="bilinear",
        bins=[
            BinDecl(
                id=b["id"],
                name=b["name"],
                active=bool(b.get(CONF_BIN_ACTIVE, True)),
            )
            for b in entry.data[CONF_BINS]
        ],
        view_epoch=int(entry.data.get(CONF_VIEW_GENERATION, 0)),
    )


def reconcile_store_with_entry(
    storage: WastebinStorage,
    entry: ConfigEntry,
    unmaterialized: list[str],
) -> bool:
    """Sync the ACTIVE configuration from entry.data into the store.

    entry.data is the authority for what is configured right now; the
    store additionally carries history (retired bins, old epochs) and
    is therefore never truncated - only flags and current values move.
    ``unmaterialized`` lists archive files without a store entry yet,
    needed to stamp their capture epoch on a view bump. Returns True
    when anything changed (caller saves).
    """
    store = storage.calibration
    changed = False
    entry_roi = Roi(
        x=float(entry.data[CONF_ROI_X]),
        y=float(entry.data[CONF_ROI_Y]),
        w=float(entry.data[CONF_ROI_W]),
        h=float(entry.data[CONF_ROI_H]),
    )
    if not roi_equal(store.roi, entry_roi):
        store.roi = entry_roi
        # Gate samples are per-crop frame statistics with a documented
        # one-day rolling window - ephemeral, not training data. Under
        # a new crop they would widen the next profile's gates with
        # numbers from the old view.
        storage.gate_samples = []
        changed = True
    working_width = int(entry.data[CONF_WORKING_WIDTH])
    if store.working_width != working_width:
        store.working_width = working_width
        storage.gate_samples = []
        changed = True
    view_generation = int(entry.data.get(CONF_VIEW_GENERATION, 0))
    while store.view_epoch < view_generation:
        store.bump_view_epoch(unmaterialized)
        storage.gate_samples = []
        changed = True
    declared = {b["id"]: b for b in entry.data[CONF_BINS]}
    ids_in_store = set()
    for i, decl in enumerate(list(store.bins)):
        ids_in_store.add(decl.id)
        wanted = declared.get(decl.id)
        active = wanted is not None and bool(
            wanted.get(CONF_BIN_ACTIVE, True)
        )
        name = wanted["name"] if wanted is not None else decl.name
        if decl.active != active or decl.name != name:
            store.bins[i] = replace(decl, active=active, name=name)
            changed = True
    for bin_id, b in declared.items():
        if bin_id not in ids_in_store:
            store.bins.append(
                BinDecl(
                    id=bin_id,
                    name=b["name"],
                    active=bool(b.get(CONF_BIN_ACTIVE, True)),
                )
            )
            changed = True
    return changed


class WastebinStorage:
    """Loads and persists the per-entry state via the HA Store helper."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._store = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}")
        self.calibration: CalibrationStore = empty_store_from_entry(entry)
        self.profile: Profile | None = None
        # Learning starts enabled: a fresh installation is exactly the
        # phase in which snapshots must be collected.
        self.learning: bool = True
        # [median_sat, median_val, clip_frac] per analyzed daylight
        # frame, in capture order; feeds derive_quality_gates so the
        # light gates widen automatically from unlabeled frames.
        self.gate_samples: list[list[float]] = []
        # View epoch the current profile was learned under: a view
        # bump makes the profile scene-stale even though its ROI still
        # matches (the profile schema itself stays unversioned).
        self.profile_view_epoch: int = 0
        # Set on unload: a superseded instance (config entry reloaded
        # while a relearn was still running) must never write again,
        # or last-write-wins would resurrect pre-reload state.
        self._closed = False

    def mark_closed(self) -> None:
        self._closed = True

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if not data:
            # Missing/empty HA-side store with a populated mirror is
            # the classic backup-restore hole: preserve before the
            # first save can clobber the only surviving copy.
            await self._preserve_divergent_mirror()
            return
        if data.get("calibration"):
            raw = data["calibration"]
            if int(raw.get("schema_version", 0)) == 1:
                # One-time insurance before the first migrated save: the
                # original v1 dict lands next to the images, so even a
                # buggy migration can never destroy evidence.
                await self._hass.async_add_executor_job(
                    self._write_v1_backup, raw
                )
            self.calibration = store_from_dict(raw)
        if data.get("profile"):
            self.profile = profile_from_dict(data["profile"])
        self.learning = bool(data.get("learning", True))
        self.gate_samples = [
            [float(v) for v in sample]
            for sample in data.get("gate_samples", [])
        ]
        # Legacy stores (pre-2.3) carry no marker: assume the profile
        # matches the loaded view so the upgrade itself never flags a
        # stale profile; only future bumps do.
        raw_epoch = data.get("profile_view_epoch")
        self.profile_view_epoch = (
            self.calibration.view_epoch if raw_epoch is None else int(raw_epoch)
        )
        await self._preserve_divergent_mirror()

    def _write_v1_backup(self, raw: dict) -> None:
        target = archive_dir(self._hass, self._entry.entry_id) / V1_BACKUP_NAME
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _LOGGER.info("v1 calibration backup written to %s", target)

    def _write_mirror(self, payload: str) -> None:
        target = store_anchor(self._hass, self._entry.entry_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")

    async def _preserve_divergent_mirror(self) -> None:
        """Never silently destroy a mirror that knows more than we do.

        After a backup restore, the HA-side store may roll back while
        the archive mirror still holds newer evidence; the very first
        save would then overwrite the only surviving copy. If the
        mirror references any image path the loaded store does not
        know, it is preserved as ``store.json.recovered`` (kept until a
        human decides; existing recovery files are never overwritten).
        """
        known = {e.path for e in self.calibration.images}

        def _preserve() -> str | None:
            source = store_anchor(self._hass, self._entry.entry_id)
            if not source.exists():
                return None
            recovery = source.with_suffix(source.suffix + ".recovered")
            if recovery.exists():
                return None
            try:
                data = json.loads(source.read_text(encoding="utf-8"))
                mirror_paths = {
                    str(e.get("path")) for e in data.get("images", [])
                }
            except (OSError, json.JSONDecodeError, AttributeError):
                return None
            if mirror_paths - known:
                recovery.write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8"
                )
                return str(recovery)
            return None

        preserved = await self._hass.async_add_executor_job(_preserve)
        if preserved:
            _LOGGER.warning(
                "the archive mirror references calibration images unknown "
                "to the loaded store (restored from an older backup?); the "
                "mirror was preserved as %s before any overwrite",
                preserved,
            )

    async def async_save(self) -> None:
        if self._closed:
            _LOGGER.debug(
                "discarding save from superseded storage instance (%s)",
                self._entry.entry_id,
            )
            return
        calibration = store_to_dict(self.calibration)
        await self._store.async_save(
            {
                "calibration": calibration,
                "profile": (
                    profile_to_dict(self.profile) if self.profile else None
                ),
                "learning": self.learning,
                "gate_samples": self.gate_samples,
                "profile_view_epoch": self.profile_view_epoch,
            }
        )
        # Mirror next to the images: the archive folder stays
        # self-contained even when .storage is lost or restored from a
        # backup that does not include the media directory.
        await self._hass.async_add_executor_job(
            self._write_mirror,
            json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        )

    async def async_remove(self) -> None:
        await self._store.async_remove()
