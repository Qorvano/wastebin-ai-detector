"""Detection: apply a learned profile to an image.

Every threshold used here comes from the profile (learned per
installation); this module contains no tunable numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .ccl import largest_component_area
from .color import RGB_8BIT_LEVELS, circular_dist_deg, rgb_to_hsv
from .imageio import extract_working_roi, load_image_rgb
from .profile import BinModel, Profile, validate_profile


def is_uncertain(area_frac: float, learning_stats: dict[str, Any]) -> bool:
    """Is this area fraction inside the learned ambiguity interval?

    The interval between the smallest observed positive and the largest
    observed negative blob is ambiguous by construction: nothing in the
    calibration data says which side such a frame belongs to. For
    non-separable profiles the interval is inverted; taking min/max of
    the two bounds covers both orientations conservatively. Profiles
    without learning stats (hand-written) yield False: no information,
    no claim of uncertainty.
    """
    min_pos = learning_stats.get("min_pos_area_frac")
    max_neg = learning_stats.get("max_neg_area_frac")
    if min_pos is None or max_neg is None:
        return False
    lo = min(min_pos, max_neg)
    hi = max(min_pos, max_neg)
    return lo < area_frac < hi


def row_duplicate_fraction(arr: np.ndarray) -> float:
    """Fraction of adjacent working-image row pairs that are duplicates.

    Broken video keyframes are concealed by repeating the last decoded
    row downward (vertical smear); such frames have plausible color
    statistics but carry no scene content below the break. Two rows
    whose mean absolute channel difference is at most one 8-bit quantum
    are indistinguishable from encoder row repetition - the same
    quantization derivation as the clip floor, not a chosen value.

    A pair counts as duplicated only when EVERY signal column matches
    within one quantum (maximum, not mean): true row repetition is
    columnwise identical, while merely smooth image regions always
    carry at least one column of residual texture above a quantum - a
    mean would blur that distinction on heavily compressed frames.

    Clipped pixels are excluded from the comparison: saturation makes
    them identical by construction of the sensor limit, so a blown
    highlight region says nothing about smear (and an overexposed but
    genuine frame must stay a case for the overexposure gate, not this
    one). A row pair with no unclipped column carries no signal and is
    excluded from the statistic entirely.
    """
    if arr.shape[0] < 2:
        return 0.0
    quantum = 1.0 / RGB_8BIT_LEVELS
    clip_floor = 1.0 - quantum
    upper, lower = arr[:-1], arr[1:]
    # A pixel is clipped when its brightest channel sits at the sensor
    # ceiling (same definition as clip_frac).
    signal = (upper.max(axis=2) < clip_floor) | (lower.max(axis=2) < clip_floor)
    pair_has_signal = signal.any(axis=1)
    if not bool(pair_has_signal.any()):
        return 0.0
    diff = np.abs(upper - lower).mean(axis=2)
    col_max = np.where(signal, diff, 0.0).max(axis=1)
    dup = (col_max <= quantum) & pair_has_signal
    return float(dup.sum() / pair_has_signal.sum())


def bin_mask(
    hue: np.ndarray, sat: np.ndarray, val: np.ndarray, model: BinModel
) -> np.ndarray:
    """Boolean mask of pixels matching one bin's learned color model."""
    dist = circular_dist_deg(hue, model.hue_center_deg)
    # Undefined hue (grey pixel) can never match a color model.
    dist = np.where(np.isnan(dist), np.inf, dist)
    return (dist <= model.hue_tol_deg) & (sat >= model.sat_min) & (val >= model.val_min)


@dataclass
class BinResult:
    id: str
    name: str
    present: bool
    area_frac: float
    min_area_frac: float
    margin: float  # area_frac / min_area_frac - confidence on a ratio scale
    uncertain: bool = False  # inside the learned ambiguity interval

    def to_dict(self) -> dict[str, Any]:
        # Values are serialized unrounded: any display rounding could
        # contradict `present` near the threshold (e.g. a margin of
        # 0.9996 rounding to 1.0 while present is False).
        return {
            "id": self.id,
            "name": self.name,
            "present": self.present,
            "area_frac": self.area_frac,
            "min_area_frac": self.min_area_frac,
            "margin": self.margin,
            "uncertain": self.uncertain,
        }


@dataclass
class DetectionResult:
    bins: list[BinResult]
    median_sat: float
    grayscale_suspect: bool
    working_size: tuple[int, int]  # (width, height) of the analyzed ROI
    median_val: float = 0.0
    clip_frac: float = 0.0
    overexposure_suspect: bool = False
    row_dup_frac: float = 0.0
    frame_integrity_suspect: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": [b.to_dict() for b in self.bins],
            "median_sat": self.median_sat,
            "median_val": self.median_val,
            "clip_frac": self.clip_frac,
            "row_dup_frac": self.row_dup_frac,
            "grayscale_suspect": self.grayscale_suspect,
            "overexposure_suspect": self.overexposure_suspect,
            "frame_integrity_suspect": self.frame_integrity_suspect,
            "working_size": list(self.working_size),
        }


def detect(img: Image.Image, profile: Profile) -> DetectionResult:
    """Run detection on an already-loaded PIL image."""
    validate_profile(profile)
    arr = extract_working_roi(img, profile.roi, profile.working_width, profile.resample)
    hue, sat, val = rgb_to_hsv(arr)
    total = arr.shape[0] * arr.shape[1]
    results: list[BinResult] = []
    for model in profile.bins:
        area = largest_component_area(bin_mask(hue, sat, val, model))
        frac = area / total
        results.append(
            BinResult(
                id=model.id,
                name=model.name,
                present=frac >= model.min_area_frac,
                area_frac=frac,
                min_area_frac=model.min_area_frac,
                margin=frac / model.min_area_frac,
                uncertain=is_uncertain(frac, model.learning_stats),
            )
        )
    median_sat = float(np.median(sat))
    median_val = float(np.median(val))
    # Same derivation as in the learner: one 8-bit quantum below full
    # brightness marks a pixel as clipped or about to clip.
    clip_frac = float(np.mean(val >= 1.0 - 1.0 / RGB_8BIT_LEVELS))
    row_dup_frac = row_duplicate_fraction(arr)
    return DetectionResult(
        bins=results,
        median_sat=median_sat,
        median_val=median_val,
        clip_frac=clip_frac,
        row_dup_frac=row_dup_frac,
        # More duplicated rows than any calibrated frame ever showed:
        # the frame is likely a smeared/truncated keyframe, its color
        # statistics describe garbage, not the scene.
        frame_integrity_suspect=row_dup_frac > profile.row_dup_max,
        # Below the least-saturated daylight calibration image → likely
        # an IR/greyscale night frame; color detection is unreliable.
        grayscale_suspect=median_sat < profile.daylight_sat_min,
        # More clipping or a brighter frame than anything the learner
        # ever saw → harsh-light frame, color evidence is degraded.
        overexposure_suspect=(
            clip_frac > profile.overexposure_clip_max
            or median_val > profile.daylight_val_max
        ),
        working_size=(arr.shape[1], arr.shape[0]),
    )


def detect_file(path: str | Path, profile: Profile) -> DetectionResult:
    """Convenience wrapper: load an image file and run detection."""
    return detect(load_image_rgb(path), profile)
