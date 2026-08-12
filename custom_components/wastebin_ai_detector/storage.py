"""Per-entry persistence and file locations.

Three things live here, per config entry:
- the calibration store (samples + labels), the single source of truth
  for learning,
- the learned profile (a derived artifact, recomputed by relearn),
- the learning flag (whether the background snapshot collector runs).

Snapshots are archived as plain JPEG files in the media directory so
they do not bloat `/config` backups; image paths inside the calibration
store are relative to that archive directory.
"""

from __future__ import annotations

from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    CONF_BINS,
    CONF_ROI_H,
    CONF_ROI_W,
    CONF_ROI_X,
    CONF_ROI_Y,
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
    store_from_dict,
    store_to_dict,
)


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
    """Virtual file anchor used to resolve relative image paths."""
    return archive_dir(hass, entry_id) / "store.json"


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
            BinDecl(id=b["id"], name=b["name"]) for b in entry.data[CONF_BINS]
        ],
    )


class WastebinStorage:
    """Loads and persists the per-entry state via the HA Store helper."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._store = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}")
        self._entry = entry
        self.calibration: CalibrationStore = empty_store_from_entry(entry)
        self.profile: Profile | None = None
        # Learning starts enabled: a fresh installation is exactly the
        # phase in which snapshots must be collected.
        self.learning: bool = True
        # [median_sat, median_val, clip_frac] per analyzed daylight
        # frame, in capture order; feeds derive_quality_gates so the
        # light gates widen automatically from unlabeled frames.
        self.gate_samples: list[list[float]] = []

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if not data:
            return
        if data.get("calibration"):
            self.calibration = store_from_dict(data["calibration"])
        if data.get("profile"):
            self.profile = profile_from_dict(data["profile"])
        self.learning = bool(data.get("learning", True))
        self.gate_samples = [
            [float(v) for v in sample]
            for sample in data.get("gate_samples", [])
        ]

    async def async_save(self) -> None:
        await self._store.async_save(
            {
                "calibration": store_to_dict(self.calibration),
                "profile": (
                    profile_to_dict(self.profile) if self.profile else None
                ),
                "learning": self.learning,
                "gate_samples": self.gate_samples,
            }
        )

    async def async_remove(self) -> None:
        await self._store.async_remove()
