"""Learning: turn the calibration store into a detection profile.

``learn_profile`` always recomputes everything from the store - first
the color models from the pooled lid samples, then (on top of those)
the per-bin area thresholds from the labeled images, then the daylight
saturation floor. The profile is a derived artifact; there is no
incremental patching, so stale thresholds cannot survive a sample edit.

Named constants in this module (rule: no magic numbers - every value is
either learned, an FP epsilon, or named and justified here):

- ``HUE_TOL_PERCENTILE = 95`` / ``SV_MIN_PERCENTILE = 5``: two-sided
  ≈2σ coverage of the sampled lid-pixel distribution, robust against
  outlier pixels (dirt, JPEG edge artifacts). The ≈5% of genuine lid
  pixels that fall outside are compensated by the *subsequently* learned
  area threshold, which is computed with exactly these cuts in place.
- ``PROVISIONAL_AREA_SAFETY = 0.5``: only used while no negative
  evidence exists. A factor-2 margin below the smallest observed
  positive blob is the information-free midpoint (the geometric mean
  with a 0/absent negative is undefined); the profile is flagged
  ``provisional`` and the value is replaced on the first ``learn`` run
  that sees a negative sample.
- ``DEGENERATE_HUE_BAND_DEG = 180``: an acceptance band of
  ``2·tol ≥ 180°`` covers at least half the color circle - geometrically
  no discriminative power left, so learning fails loudly instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ccl import largest_component_area
from .color import (
    HUE_DEG_PER_SEXTANT,
    RGB_8BIT_LEVELS,
    circular_dist_deg,
    circular_mean_deg,
    rgb_to_hsv,
)
from .detect import bin_mask
from .errors import CalibrationError, ImageLoadError
from .imageio import extract_working_roi, load_image_rgb, rect_to_pixels
from .profile import BinModel, Profile
from .store import CalibrationStore, resolve_image_path, validate_store

HUE_TOL_PERCENTILE = 95.0
SV_MIN_PERCENTILE = 5.0
PROVISIONAL_AREA_SAFETY = 0.5
DEGENERATE_HUE_BAND_DEG = 180.0


@dataclass
class ColorLearnResult:
    hue_center_deg: float
    hue_tol_deg: float
    sat_min: float
    val_min: float
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def learn_color_model(
    hue: np.ndarray, sat: np.ndarray, val: np.ndarray, *, bin_id: str = "?"
) -> ColorLearnResult:
    """Learn one bin's color model from pooled sample pixels (1-D arrays)."""
    warnings: list[str] = []
    n_px = int(sat.size)
    if n_px == 0:
        raise CalibrationError(f"bin {bin_id}: no sample pixels")
    center, resultant = circular_mean_deg(hue)  # raises if all hues NaN
    valid_hue = hue[~np.isnan(hue)]
    dist = circular_dist_deg(valid_hue, center)
    tol = float(np.percentile(dist, HUE_TOL_PERCENTILE))
    if tol <= 0.0:
        # All pooled pixels share one exact float hue (uniform synthetic
        # patches). Derive the smallest band that stays non-empty
        # without admitting any neighboring 8-bit-representable hue:
        # neighbors of a pixel with chroma c sit ≥ 60°/(255·c) away, so
        # half the finest quantum among the samples admits none of them.
        # chroma = sat · val, since sat = c/maxc and val = maxc.
        max_chroma = float(np.max(sat * val))
        tol = HUE_DEG_PER_SEXTANT / (RGB_8BIT_LEVELS * max_chroma) / 2.0
    if 2.0 * tol >= DEGENERATE_HUE_BAND_DEG:
        raise CalibrationError(
            f"bin {bin_id}: learned hue band ±{tol:.1f}° covers at least half "
            f"the color circle (resultant R={resultant:.3f}) - the samples "
            "have no consistent color. Re-draw them on a colored lid area; "
            "grey/black lids cannot be color-calibrated - attach a small "
            "colored marker to the lid and sample that instead"
        )
    sat_min = float(np.percentile(sat, SV_MIN_PERCENTILE))
    val_min = float(np.percentile(val, SV_MIN_PERCENTILE))
    # Order-statistics-derived warning (no chosen cutoff): if
    # q/100·(n−1) < 1, the q-th percentile IS the sample minimum, i.e.
    # the sample is too small for the percentile to differ from min.
    if SV_MIN_PERCENTILE / 100.0 * (n_px - 1) < 1.0:
        warnings.append(
            f"bin {bin_id}: only {n_px} sample pixels - the "
            f"{SV_MIN_PERCENTILE:g}th percentile equals the sample minimum; "
            "draw larger sample rectangles"
        )
    return ColorLearnResult(
        hue_center_deg=center,
        hue_tol_deg=tol,
        sat_min=sat_min,
        val_min=val_min,
        stats={
            "n_sample_px": n_px,
            "n_valid_hue_px": int(valid_hue.size),
            "resultant_r": resultant,
        },
        warnings=warnings,
    )


@dataclass
class AreaLearnResult:
    min_area_frac: float
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def learn_area_threshold(
    pos_areas: list[float], neg_areas: list[float], *, bin_id: str = "?"
) -> AreaLearnResult:
    """Learn the presence threshold from labeled blob-area fractions.

    Separable case: geometric mean of (smallest positive, largest
    negative) - the midpoint on the ratio scale areas live on.
    """
    warnings: list[str] = []
    if not pos_areas:
        raise CalibrationError(
            f"bin {bin_id}: no image is labeled 'present' - at least one "
            "positive example is required"
        )
    min_pos = min(pos_areas)
    if min_pos <= 0.0:
        raise CalibrationError(
            f"bin {bin_id}: a 'present'-labeled image yields a zero-area "
            "color blob - the color model misses the lid there entirely; "
            "check the labels and sample rectangles"
        )
    max_neg = max(neg_areas) if neg_areas else 0.0
    separable = max_neg < min_pos
    provisional = False
    if not neg_areas or max_neg <= 0.0:
        threshold = min_pos * PROVISIONAL_AREA_SAFETY
        provisional = not neg_areas
        if not neg_areas:
            warnings.append(
                f"bin {bin_id}: no 'absent'-labeled image yet - provisional "
                f"threshold {threshold:.4f} (= {PROVISIONAL_AREA_SAFETY:g} × "
                f"smallest positive {min_pos:.4f}); label an image without "
                "this bin to replace it"
            )
    elif separable:
        threshold = math.sqrt(min_pos * max_neg)
    else:
        threshold = math.sqrt(min_pos * max_neg)
        warnings.append(
            f"bin {bin_id}: NOT separable - largest negative blob "
            f"{max_neg:.4f} ≥ smallest positive {min_pos:.4f}; threshold "
            f"{threshold:.4f} lies inside the overlap and WILL misclassify "
            "some calibration images; improve samples or labels"
        )
    return AreaLearnResult(
        min_area_frac=threshold,
        stats={
            "n_pos": len(pos_areas),
            "n_neg": len(neg_areas),
            "min_pos_area_frac": min_pos,
            # None (not 0.0) without any negative image: 0.0 would be an
            # unobserved claim, and the ambiguity interval downstream
            # would then treat everything below min_pos as uncertain.
            "max_neg_area_frac": max_neg if neg_areas else None,
            "provisional": provisional,
            "separable": separable,
        },
        warnings=warnings,
    )


def _hue_bands_overlap(a: BinModel, b: BinModel) -> bool:
    """Circular interval overlap of two learned hue bands (pure geometry)."""
    centers = abs(((a.hue_center_deg - b.hue_center_deg + 180.0) % 360.0) - 180.0)
    return centers <= a.hue_tol_deg + b.hue_tol_deg


def learn_profile(
    store: CalibrationStore, store_path: str | Path
) -> tuple[Profile, list[str]]:
    """Full recomputation: store → profile. Returns (profile, warnings)."""
    validate_store(store)
    warnings: list[str] = []

    # One pass through the single shared pipeline per image. Images
    # whose file vanished from the archive are skipped with a loud
    # warning instead of poisoning every future relearn: the store may
    # legitimately outlive individual snapshot files.
    hsv_by_image: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    usable_images: list = []
    for entry in store.images:
        try:
            img = load_image_rgb(resolve_image_path(store_path, entry.path))
        except ImageLoadError as exc:
            warnings.append(f"skipping calibration image: {exc}")
            continue
        arr = extract_working_roi(img, store.roi, store.working_width, store.resample)
        hsv_by_image[entry.path] = rgb_to_hsv(arr)
        usable_images.append(entry)

    # 1) Color models from pooled sample rectangles.
    bins: list[BinModel] = []
    for decl in store.bins:
        hue_parts: list[np.ndarray] = []
        sat_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        n_sample_images = 0
        for entry in usable_images:
            rects = entry.samples.get(decl.id, [])
            if not rects:
                continue
            n_sample_images += 1
            hue, sat, val = hsv_by_image[entry.path]
            height, width = sat.shape
            for rect in rects:
                x0, y0, x1, y1 = rect_to_pixels(
                    rect.x, rect.y, rect.w, rect.h, width, height
                )
                hue_parts.append(hue[y0:y1, x0:x1].ravel())
                sat_parts.append(sat[y0:y1, x0:x1].ravel())
                val_parts.append(val[y0:y1, x0:x1].ravel())
        if not hue_parts:
            raise CalibrationError(
                f"bin {decl.id}: no sample rectangles in any calibration image"
            )
        color = learn_color_model(
            np.concatenate(hue_parts),
            np.concatenate(sat_parts),
            np.concatenate(val_parts),
            bin_id=decl.id,
        )
        warnings.extend(color.warnings)
        color.stats["n_sample_images"] = n_sample_images
        # Singular-vs-plural condition, not a tuned cutoff: percentile
        # floors learned from one single image carry zero cross-image
        # variance (lighting, encoder noise) and sit knife-edge on that
        # image's pixel values.
        if n_sample_images < 2:
            warnings.append(
                f"bin {decl.id}: all lid samples come from a single image - "
                "learned color floors have no cross-condition slack; add "
                "sample rectangles from more snapshots (different light/"
                "weather) and re-run learn"
            )
        bins.append(
            BinModel(
                id=decl.id,
                name=decl.name,
                hue_center_deg=color.hue_center_deg,
                hue_tol_deg=color.hue_tol_deg,
                sat_min=color.sat_min,
                val_min=color.val_min,
                min_area_frac=1.0,  # placeholder, learned in step 2
                learning_stats=color.stats,
            )
        )

    # 2) Area thresholds on top of the final color models.
    for model in bins:
        pos_areas: list[float] = []
        neg_areas: list[float] = []
        for entry in usable_images:
            if model.id in entry.present:
                target = pos_areas
            elif model.id in entry.absent:
                target = neg_areas
            else:
                continue
            hue, sat, val = hsv_by_image[entry.path]
            total = sat.shape[0] * sat.shape[1]
            area = largest_component_area(bin_mask(hue, sat, val, model))
            target.append(area / total)
        result = learn_area_threshold(pos_areas, neg_areas, bin_id=model.id)
        warnings.extend(result.warnings)
        model.min_area_frac = result.min_area_frac
        model.learning_stats.update(result.stats)

    # 3) Daylight quality gates (all calibration images are daylight by
    # contract - documented calibration rule). Alongside the saturation
    # floor, learn the overexposure ceiling: the clip fraction counts
    # ROI pixels within one 8-bit quantum of full brightness (a pixel
    # at 254 or 255 is clipped or about to; derived from
    # RGB_8BIT_LEVELS, not chosen). The maxima over the calibration set
    # define "worse than anything ever calibrated", the same extremum
    # logic as daylight_sat_min in the opposite direction.
    clip_floor = 1.0 - 1.0 / RGB_8BIT_LEVELS
    median_sats: list[float] = []
    clip_fracs: list[float] = []
    median_vals: list[float] = []
    for e in usable_images:
        _hue, sat, val = hsv_by_image[e.path]
        median_sats.append(float(np.median(sat)))
        clip_fracs.append(float(np.mean(val >= clip_floor)))
        median_vals.append(float(np.median(val)))
    if not median_sats:
        raise CalibrationError("store contains no usable calibration images")
    daylight_sat_min = min(median_sats)
    overexposure_clip_max = max(clip_fracs)
    daylight_val_max = max(median_vals)
    daylight_stats = {
        "n_images": len(median_sats),
        "min_median_sat": daylight_sat_min,
        "median_of_medians": float(np.median(np.asarray(median_sats))),
        "max_clip_frac": overexposure_clip_max,
        "max_median_val": daylight_val_max,
    }

    # 4) Ambiguity diagnosis: overlapping learned hue bands.
    for i, a in enumerate(bins):
        for b in bins[i + 1 :]:
            if _hue_bands_overlap(a, b):
                warnings.append(
                    f"bins {a.id} and {b.id}: learned hue bands overlap "
                    f"({a.hue_center_deg:.0f}°±{a.hue_tol_deg:.0f}° vs "
                    f"{b.hue_center_deg:.0f}°±{b.hue_tol_deg:.0f}°) - one bin "
                    "can produce blobs in both masks; results for these two "
                    "bins are not independent"
                )

    profile = Profile(
        roi=store.roi,
        working_width=store.working_width,
        resample=store.resample,
        daylight_sat_min=daylight_sat_min,
        overexposure_clip_max=overexposure_clip_max,
        daylight_val_max=daylight_val_max,
        daylight_stats=daylight_stats,
        bins=bins,
    )
    return profile, warnings
