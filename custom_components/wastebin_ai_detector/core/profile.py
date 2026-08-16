"""Profile data model: the learned, per-installation detection parameters.

A profile is a *derived artifact* - it is always recomputed in full from
the calibration store by :func:`..learn.learn_profile` and carries every
learned threshold plus diagnostic learning statistics. It contains no
hand-tuned numbers: everything in here is user setup data (ROI, bin
list, working width) or learned from the user's calibration images.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ProfileError

SCHEMA_VERSION = 1

# Categorical resampling choice, stored in the profile so calibration
# and detection always run the identical pipeline. Bilinear is the
# default: it approximates an area average without Lanczos overshoot,
# so no out-of-gamut colors are invented at lid edges.
KNOWN_RESAMPLE = ("bilinear", "nearest")

# Shared floating-point slack for validating relative coordinates
# (e.g. x + w == 1.0 arriving as 1.0000000000000002 after division).
# Used identically by every bounds check in the pipeline - one policy,
# not per-module copies.
REL_EPS = 1e-9


@dataclass(frozen=True)
class Roi:
    """Region of interest in image-relative coordinates (0..1)."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in ROI-relative coordinates (0..1)."""

    x: float
    y: float
    w: float
    h: float


@dataclass
class BinModel:
    """Learned color and area model for one bin."""

    id: str
    name: str
    hue_center_deg: float
    hue_tol_deg: float
    sat_min: float
    val_min: float
    min_area_frac: float
    learning_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Profile:
    """Complete learned detection profile for one installation."""

    roi: Roi
    working_width: int | None
    resample: str
    daylight_sat_min: float
    bins: list[BinModel]
    daylight_stats: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    # Learned evidence-quality gates. The permissive defaults (1.0)
    # keep profiles from older versions valid; the gates then simply
    # never fire until the next relearn fills in real learned maxima.
    overexposure_clip_max: float = 1.0
    daylight_val_max: float = 1.0
    # Frame-integrity gate: maximum observed fraction of duplicated
    # adjacent rows (encoder error concealment repeats rows on broken
    # keyframes). Same permissive-default pattern as above.
    row_dup_max: float = 1.0
    # Optional polygon region (rings, image-relative) refining the roi
    # bbox; None = whole bbox (pre-polygon semantics).
    roi_polygons: list | None = None
    # Were this profile's thresholds measured with mutual exclusion
    # between bins? Default False so a profile learned before the rule
    # keeps being detected exactly as it was measured; learn_profile
    # sets it True.
    mutual_exclusion: bool = False


def validate_profile(profile: Profile) -> None:
    """Raise :class:`ProfileError` if the profile is structurally invalid."""
    if profile.schema_version != SCHEMA_VERSION:
        raise ProfileError(
            f"unsupported profile schema_version {profile.schema_version} "
            f"(supported: {SCHEMA_VERSION}) - re-run 'learn' on this "
            "installation or migrate the profile"
        )
    if profile.resample not in KNOWN_RESAMPLE:
        raise ProfileError(
            f"unknown resample {profile.resample!r}; known: {KNOWN_RESAMPLE}"
        )
    if profile.working_width is not None and profile.working_width <= 0:
        raise ProfileError(f"working_width must be positive, got {profile.working_width}")
    if not 0.0 <= profile.daylight_sat_min <= 1.0:
        raise ProfileError(f"daylight_sat_min outside [0, 1]: {profile.daylight_sat_min}")
    for name, value in (
        ("overexposure_clip_max", profile.overexposure_clip_max),
        ("daylight_val_max", profile.daylight_val_max),
        ("row_dup_max", profile.row_dup_max),
    ):
        if not 0.0 <= value <= 1.0:
            raise ProfileError(f"{name} outside [0, 1]: {value}")
    if not profile.bins:
        raise ProfileError("profile contains no bins")
    seen: set[str] = set()
    for b in profile.bins:
        if not b.id:
            raise ProfileError("bin with empty id")
        if b.id in seen:
            raise ProfileError(f"duplicate bin id {b.id!r}")
        seen.add(b.id)
        if not 0.0 <= b.hue_center_deg < 360.0:
            raise ProfileError(f"bin {b.id}: hue_center_deg outside [0, 360): {b.hue_center_deg}")
        # 2·tol ≥ 180° would accept at least half the color circle -
        # geometrically no discriminative power left (same bound the
        # learner enforces; re-checked here against hand-edited files).
        if not 0.0 < b.hue_tol_deg or 2.0 * b.hue_tol_deg >= 180.0:
            raise ProfileError(f"bin {b.id}: hue_tol_deg out of range (0, 90): {b.hue_tol_deg}")
        for name, value in (("sat_min", b.sat_min), ("val_min", b.val_min)):
            if not 0.0 <= value <= 1.0:
                raise ProfileError(f"bin {b.id}: {name} outside [0, 1]: {value}")
        if not 0.0 < b.min_area_frac <= 1.0:
            raise ProfileError(f"bin {b.id}: min_area_frac outside (0, 1]: {b.min_area_frac}")
        # Shape stats feed the plausibility filter in detection: a
        # hand-edited profile must not smuggle in values that crash or
        # silently blind a bin.
        shape_n = b.learning_stats.get("shape_n")
        if shape_n is not None:
            if isinstance(shape_n, bool) or not isinstance(shape_n, int):
                raise ProfileError(
                    f"bin {b.id}: learning_stats.shape_n is not an int"
                )
            if shape_n < 0:
                raise ProfileError(f"bin {b.id}: negative shape_n")
            if shape_n >= 2:
                required = (
                    "shape_log_aspect_min",
                    "shape_log_aspect_max",
                    "shape_fill_min",
                )
                for key in required:
                    value = b.learning_stats.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ProfileError(
                            f"bin {b.id}: learning_stats.{key} missing or "
                            f"not a finite number: {value!r}"
                        )
                if (
                    b.learning_stats["shape_log_aspect_min"]
                    > b.learning_stats["shape_log_aspect_max"]
                ):
                    raise ProfileError(
                        f"bin {b.id}: shape aspect bounds inverted"
                    )
                if not 0.0 <= float(b.learning_stats["shape_fill_min"]) <= 1.0:
                    raise ProfileError(
                        f"bin {b.id}: shape_fill_min outside [0, 1]"
                    )
        # Edge-band stats feed the boundary filter in detection: a
        # hand-edited profile must not smuggle in values that crash or
        # silently blind a bin (same policy as the shape stats).
        edge_n = b.learning_stats.get("region_edge_depth_n")
        if edge_n is not None:
            if isinstance(edge_n, bool) or not isinstance(edge_n, int):
                raise ProfileError(
                    f"bin {b.id}: learning_stats.region_edge_depth_n is "
                    "not an int"
                )
            if edge_n < 0:
                raise ProfileError(
                    f"bin {b.id}: negative region_edge_depth_n"
                )
            if edge_n >= 2:
                band = b.learning_stats.get("region_edge_depth_min_frac")
                if (
                    isinstance(band, bool)
                    or not isinstance(band, (int, float))
                    or not math.isfinite(float(band))
                    or float(band) < 0.0
                ):
                    raise ProfileError(
                        f"bin {b.id}: learning_stats."
                        "region_edge_depth_min_frac missing or not a "
                        f"finite number >= 0: {band!r}"
                    )
        # The veto-qualification stats decide whether this bin may
        # erase another bin's evidence: a hand-edited profile must not
        # smuggle in values that crash detection or silently blind a
        # bin (same policy as the shape and edge-band stats).
        bar = b.learning_stats.get("veto_qualify_min_area_frac")
        if bar is not None:
            if (
                isinstance(bar, bool)
                or not isinstance(bar, (int, float))
                or not math.isfinite(float(bar))
                or not 0.0 <= float(bar) <= 1.0
            ):
                raise ProfileError(
                    f"bin {b.id}: learning_stats.veto_qualify_min_area_frac "
                    f"is not a finite fraction in [0, 1]: {bar!r}"
                )
        for key in ("veto_qualify_separable", "veto_qualify_provisional"):
            flag = b.learning_stats.get(key)
            if flag is not None and not isinstance(flag, bool):
                raise ProfileError(
                    f"bin {b.id}: learning_stats.{key} is not a bool: {flag!r}"
                )
        # These two stats feed the ambiguity interval in detection, so
        # hand-edited profiles must not smuggle in broken values.
        for key in ("min_pos_area_frac", "max_neg_area_frac"):
            value = b.learning_stats.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProfileError(
                    f"bin {b.id}: learning_stats.{key} is not numeric: {value!r}"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ProfileError(
                    f"bin {b.id}: learning_stats.{key} outside [0, 1]: {value}"
                )


def profile_to_dict(profile: Profile) -> dict[str, Any]:
    data = asdict(profile)
    # Stable, human-scannable key order for the JSON file.
    return {
        "schema_version": data["schema_version"],
        "roi": data["roi"],
        "working_width": data["working_width"],
        "resample": data["resample"],
        "daylight_sat_min": data["daylight_sat_min"],
        "overexposure_clip_max": data["overexposure_clip_max"],
        "daylight_val_max": data["daylight_val_max"],
        "row_dup_max": data["row_dup_max"],
        "roi_polygons": data["roi_polygons"],
        "mutual_exclusion": data["mutual_exclusion"],
        "daylight_stats": data["daylight_stats"],
        "bins": data["bins"],
    }


def profile_from_dict(data: dict[str, Any]) -> Profile:
    try:
        profile = Profile(
            schema_version=int(data["schema_version"]),
            roi=Roi(**{k: float(data["roi"][k]) for k in ("x", "y", "w", "h")}),
            working_width=(
                None if data["working_width"] is None else int(data["working_width"])
            ),
            resample=str(data["resample"]),
            daylight_sat_min=float(data["daylight_sat_min"]),
            overexposure_clip_max=float(data.get("overexposure_clip_max", 1.0)),
            daylight_val_max=float(data.get("daylight_val_max", 1.0)),
            row_dup_max=float(data.get("row_dup_max", 1.0)),
            mutual_exclusion=bool(data.get("mutual_exclusion", False)),
            roi_polygons=(
                None
                if data.get("roi_polygons") is None
                else [
                    [(float(x), float(y)) for x, y in ring]
                    for ring in data["roi_polygons"]
                ]
            ),
            daylight_stats=dict(data.get("daylight_stats", {})),
            bins=[
                BinModel(
                    id=str(b["id"]),
                    name=str(b["name"]),
                    hue_center_deg=float(b["hue_center_deg"]),
                    hue_tol_deg=float(b["hue_tol_deg"]),
                    sat_min=float(b["sat_min"]),
                    val_min=float(b["val_min"]),
                    min_area_frac=float(b["min_area_frac"]),
                    learning_stats=dict(b.get("learning_stats", {})),
                )
                for b in data["bins"]
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError(f"malformed profile data: {exc!r}") from exc
    validate_profile(profile)
    return profile


def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read profile {path}: {exc}") from exc
    return profile_from_dict(data)


def save_profile(profile: Profile, path: str | Path) -> None:
    validate_profile(profile)
    Path(path).write_text(
        json.dumps(profile_to_dict(profile), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
