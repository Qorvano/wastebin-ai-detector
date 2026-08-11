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
from .color import circular_dist_deg, rgb_to_hsv
from .imageio import extract_working_roi, load_image_rgb
from .profile import BinModel, Profile, validate_profile


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
        }


@dataclass
class DetectionResult:
    bins: list[BinResult]
    median_sat: float
    grayscale_suspect: bool
    working_size: tuple[int, int]  # (width, height) of the analyzed ROI

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": [b.to_dict() for b in self.bins],
            "median_sat": self.median_sat,
            "grayscale_suspect": self.grayscale_suspect,
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
            )
        )
    median_sat = float(np.median(sat))
    return DetectionResult(
        bins=results,
        median_sat=median_sat,
        # Below the least-saturated daylight calibration image → likely
        # an IR/greyscale night frame; color detection is unreliable.
        grayscale_suspect=median_sat < profile.daylight_sat_min,
        working_size=(arr.shape[1], arr.shape[0]),
    )


def detect_file(path: str | Path, profile: Profile) -> DetectionResult:
    """Convenience wrapper: load an image file and run detection."""
    return detect(load_image_rgb(path), profile)
