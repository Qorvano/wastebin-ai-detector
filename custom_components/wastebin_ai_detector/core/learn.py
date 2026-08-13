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
- ``COHERENT_POSTERIOR_MIN = 0.5``: the Bayes/MAP decision boundary
  between the mixture components, not an adjustable cutoff.
- ``COHERENT_MAJORITY_MIN = 0.5``: the theoretical breakdown point of
  robust estimation; less than a coherent majority means the sample
  cannot claim to show the lid color.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ccl import largest_component_area
from .region import region_mask
from .color import (
    HUE_DEG_PER_SEXTANT,
    RGB_8BIT_LEVELS,
    circular_dist_deg,
    circular_mean_deg,
    fit_vonmises_uniform_mixture,
    rgb_to_hsv,
    vonmises_kappa_from_resultant,
)
from .detect import bin_mask, row_duplicate_fraction, select_component
from .errors import CalibrationError, ImageLoadError, RoiError
from .imageio import extract_working_roi, load_image_rgb, rect_to_pixels, roi_to_pixels
from .profile import BinModel, Profile, Roi
from .store import (
    CalibrationStore,
    image_rect_in_roi,
    learning_view,
    resolve_image_path,
    validate_store,
)

HUE_TOL_PERCENTILE = 95.0
SV_MIN_PERCENTILE = 5.0
PROVISIONAL_AREA_SAFETY = 0.5
DEGENERATE_HUE_BAND_DEG = 180.0

# Mixture-learning constants, both derived rather than tuned:
# - 0.5 posterior is the Bayes/MAP decision boundary between the two
#   mixture components, not an adjustable cutoff.
# - 0.5 coherent share is the theoretical breakdown point of robust
#   estimation: past it, signal and contamination are indistinguishable
#   in principle, and it is the weakest form of the user's claim "this
#   rectangle shows the lid color".
COHERENT_POSTERIOR_MIN = 0.5
COHERENT_MAJORITY_MIN = 0.5


def seeded_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """The connected component of ``mask`` touching ``seed`` (both 2-D
    bool). Exact 8-connectivity via iterative dilation-by-shifts until
    stable - calibration-time only, so the O(diameter) loop is fine.
    Returns None when mask and seed do not overlap."""
    current = mask & seed
    if not bool(current.any()):
        return None
    while True:
        grown = current.copy()
        grown[1:, :] |= current[:-1, :]
        grown[:-1, :] |= current[1:, :]
        grown[:, 1:] |= current[:, :-1]
        grown[:, :-1] |= current[:, 1:]
        grown[1:, 1:] |= current[:-1, :-1]
        grown[1:, :-1] |= current[:-1, 1:]
        grown[:-1, 1:] |= current[1:, :-1]
        grown[:-1, :-1] |= current[1:, 1:]
        grown &= mask
        if bool((grown == current).all()):
            return current
        current = grown


def shape_bounds(
    observations: list[tuple[int, int, int, int]],
    pooled_aspect_span: float,
) -> dict[str, float]:
    """Learned plausibility bounds from (area_px, box_w, box_h, denom)
    observations of ONE bin, plus the pooled lid-aspect span of ALL
    bins in this installation.

    Two criteria, both position-invariant by construction:
    - FILL (area / box area): a lid is a compact solid at any position
      and any light; hedge fringes are not. Bound = extremum minus
      max(successive-difference slack, perimeter-pixel floor) - the
      quality-gate slack convention plus the structural resolution of
      a discretized blob boundary.
    - LOG ASPECT: bounds = own extrema widened by the pooled span of
      lid aspects observed across ALL bins - each bin's lid is a lid
      seen at a DIFFERENT position, so the pooled span is this
      installation's measured scale of position/perspective-induced
      aspect variation (a moved bin must stay plausible; a vertical
      body streak lies far outside any lid aspect). One-pixel box
      resolution is the floor.

    Deliberately NO area bound: lid areas vary by orders of magnitude
    across light regimes (field: factor >100 for blue), and area is
    already governed by the learned threshold plus ambiguity band.
    Bounds live in log space so an observation equal to its own
    extremum can never fall outside through an exp/log round-trip.
    """

    def slack(series: np.ndarray) -> float:
        if series.size < 2:
            return 0.0
        return float(np.median(np.abs(np.diff(series))))

    aspects, fills, rot_fill_bounds = [], [], []
    abs_aspect_max = 0.0
    aspect_floor = fill_floor = 0.0
    for area_px, box_w, box_h, _denom in observations:
        log_aspect = math.log(box_w / box_h)
        aspects.append(log_aspect)
        fill = area_px / (box_w * box_h)
        fills.append(fill)
        # In-plane rotation geometry of a rigid planar shape (derived,
        # not chosen): rotating a shape whose bbox has aspect a moves
        # the bbox aspect within [1/a, a] and shrinks the bbox fill by
        # at most the rectangle factor 2a/(1+a)^2 (worst case at 45°).
        # A bin turned on its spot must stay plausible.
        a = math.exp(abs(log_aspect))
        abs_aspect_max = max(abs_aspect_max, abs(log_aspect))
        rot_fill_bounds.append(fill * 2.0 * a / ((1.0 + a) ** 2))
        perimeter = 2.0 * (box_w + box_h)
        aspect_floor = max(
            aspect_floor, math.log1p(1.0 / box_w) + math.log1p(1.0 / box_h)
        )
        fill_floor = max(fill_floor, perimeter / (box_w * box_h))
    aspect_arr = np.asarray(aspects, dtype=np.float64)
    fill_arr = np.asarray(fills, dtype=np.float64)
    s_aspect = max(slack(aspect_arr), aspect_floor, pooled_aspect_span)
    s_fill = max(slack(fill_arr), fill_floor)
    # Aspect band symmetric around square: rotation can carry any
    # observed aspect through 1 to its transpose, so the band spans
    # ±(largest observed |log aspect|) plus the widening terms.
    half_band = abs_aspect_max + s_aspect
    return {
        "shape_n": int(aspect_arr.size),
        "shape_log_aspect_min": -half_band,
        "shape_log_aspect_max": half_band,
        "shape_fill_min": max(
            min(min(rot_fill_bounds), float(fill_arr.min())) - s_fill, 0.0
        ),
    }


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
    """Learn one bin's color model from pooled sample pixels (1-D arrays).

    Robust against junk pixels (overexposed highlights, edge artifacts):
    a von Mises + uniform mixture separates the coherent color
    population from scatter, and band and floors are learned from the
    coherent pixels only. Field measurements showed that plain
    percentiles break as soon as more than the percentile share of the
    sample is junk.
    """
    warnings: list[str] = []
    n_px = int(sat.size)
    if n_px == 0:
        raise CalibrationError(f"bin {bin_id}: no sample pixels")
    _center0, resultant = circular_mean_deg(hue)  # raises if all hues NaN
    valid_mask = ~np.isnan(hue)
    valid_hue = np.asarray(hue, dtype=np.float64)[valid_mask]
    sat_valid = np.asarray(sat, dtype=np.float64)[valid_mask]
    val_valid = np.asarray(val, dtype=np.float64)[valid_mask]
    n_valid = int(valid_hue.size)
    # chroma = sat · val, since sat = c/maxc and val = maxc.
    chroma_valid = sat_valid * val_valid

    # Deterministic initialization from the hue histogram. The bin
    # width is the hue quantization of an 8-bit pixel at the median
    # chroma of the sample (60°/(255·c)), a structural resolution, not
    # a chosen granularity.
    median_chroma = float(np.median(chroma_valid))
    bin_width = HUE_DEG_PER_SEXTANT / (RGB_8BIT_LEVELS * median_chroma)
    n_bins = max(int(np.ceil(360.0 / bin_width)), 1)
    hist, edges = np.histogram(valid_hue, bins=n_bins, range=(0.0, 360.0))
    peak = int(np.argmax(hist))
    mu0 = float((edges[peak] + edges[peak + 1]) / 2.0)
    # Moment start for the mixture weight: under the model, the share
    # of pixels within 90° (the geometric half-circle boundary) of the
    # center is w + (1-w)/2, hence w0 = 2·(share − 1/2).
    within = circular_dist_deg(valid_hue, mu0) < 90.0
    w0 = 2.0 * (float(within.mean()) - 0.5)
    if bool(within.any()):
        _, r_within = circular_mean_deg(valid_hue[within])
        kappa0 = vonmises_kappa_from_resultant(r_within)
    else:
        kappa0 = 0.0
    # Kappa cap: below the finest representable hue step, concentration
    # is not measurable (same quantization logic as the tol floor).
    max_chroma = float(np.max(chroma_valid))
    finest_quantum_deg = HUE_DEG_PER_SEXTANT / (RGB_8BIT_LEVELS * max_chroma)
    sigma_min_rad = float(np.deg2rad(finest_quantum_deg / 2.0))
    kappa_max = 1.0 / (sigma_min_rad * sigma_min_rad)

    fit = fit_vonmises_uniform_mixture(
        valid_hue,
        init_center_deg=mu0,
        init_weight=w0,
        init_kappa=kappa0,
        kappa_max=kappa_max,
    )
    coherent = fit.posterior >= COHERENT_POSTERIOR_MIN
    n_coherent = int(coherent.sum())
    junk_fraction = 1.0 - (n_coherent / n_valid) if n_valid else 1.0
    if n_coherent <= n_valid * COHERENT_MAJORITY_MIN:
        raise CalibrationError(
            f"bin {bin_id}: only {n_coherent} of {n_valid} coherent sample "
            f"pixels (junk fraction {junk_fraction:.0%}) - the samples have "
            "no consistent majority color. Re-draw them on a colored lid "
            "area; grey/black lids cannot be color-calibrated - attach a "
            "small colored marker to the lid and sample that instead"
        )

    center = fit.center_deg
    dist = circular_dist_deg(valid_hue[coherent], center)
    tol = float(np.percentile(dist, HUE_TOL_PERCENTILE))
    if tol <= 0.0:
        # All coherent pixels share one exact float hue (uniform
        # synthetic patches). Half the finest 8-bit hue quantum keeps
        # the band non-empty without admitting any neighboring value.
        tol = finest_quantum_deg / 2.0
    if 2.0 * tol >= DEGENERATE_HUE_BAND_DEG:
        raise CalibrationError(
            f"bin {bin_id}: learned hue band ±{tol:.1f}° covers at least half "
            f"the color circle (resultant R={resultant:.3f}) - the samples "
            "have no consistent color. Re-draw them on a colored lid area; "
            "grey/black lids cannot be color-calibrated - attach a small "
            "colored marker to the lid and sample that instead"
        )
    # Floors from the coherent pixels only: junk (overexposed, greyish)
    # pixels must not drag them toward zero, or the runtime mask would
    # re-admit exactly the junk the mixture just removed.
    sat_min = float(np.percentile(sat_valid[coherent], SV_MIN_PERCENTILE))
    val_min = float(np.percentile(val_valid[coherent], SV_MIN_PERCENTILE))
    # Order-statistics-derived warning (no chosen cutoff): if
    # q/100·(n−1) < 1, the q-th percentile IS the sample minimum, i.e.
    # the sample is too small for the percentile to differ from min.
    if SV_MIN_PERCENTILE / 100.0 * (n_coherent - 1) < 1.0:
        warnings.append(
            f"bin {bin_id}: only {n_coherent} coherent sample pixels - the "
            f"{SV_MIN_PERCENTILE:g}th percentile equals the sample minimum; "
            "draw larger sample rectangles"
        )
    # More junk than the percentile convention absorbs by design (the
    # documented capacity of HUE_TOL_PERCENTILE): model is learned from
    # the coherent majority, but the sample spot deserves a re-draw.
    if junk_fraction > (100.0 - HUE_TOL_PERCENTILE) / 100.0:
        warnings.append(
            f"bin {bin_id}: {junk_fraction:.0%} of the sample pixels are "
            "junk (overexposed or off-color); the model was learned from "
            "the coherent majority - consider re-drawing this sample in "
            "better light"
        )
    return ColorLearnResult(
        hue_center_deg=center,
        hue_tol_deg=tol,
        sat_min=sat_min,
        val_min=val_min,
        stats={
            "n_sample_px": n_px,
            "n_valid_hue_px": n_valid,
            "n_coherent_px": n_coherent,
            "junk_fraction": junk_fraction,
            "mixture_weight": fit.weight,
            "kappa": fit.kappa,
            "em_iterations": fit.n_iter,
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


def derive_quality_gates(
    gate_samples: list[list[float]],
) -> dict[str, float] | None:
    """Quality-gate limits from unlabeled daylight frames.

    Presence labels are human ground truth, but "how bright/clipped do
    frames get in this yard" is written into every archived daylight
    frame. Each sample is ``[median_sat, median_val, clip_frac]`` of
    one frame in capture order. The limit is the observed extremum
    extended by the median absolute successive difference of the
    series: the measured frame-to-frame noise scale of that metric,
    a data-derived slack instead of a knife edge. Returns None when no
    samples exist.

    Documented limitations: the slack scales with the caller's capture
    cadence (successive differences of sparser series are larger), and
    a single anomalous frame (e.g. a camera glitch) widens the derived
    gates for as long as it stays in the sample window. The integration
    keeps a one-day rolling window and re-derives after a relearn, so
    anomalies age out within a day.
    """
    if not gate_samples:
        return None
    arr = np.asarray(gate_samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise CalibrationError(
            "gate samples must be [sat, val, clip, row_dup] quadruples, "
            f"got shape {arr.shape}"
        )

    def slack(series: np.ndarray) -> float:
        if series.size < 2:
            return 0.0
        return float(np.median(np.abs(np.diff(series))))

    sats, vals, clips, dups = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return {
        "daylight_sat_min": max(float(sats.min()) - slack(sats), 0.0),
        "daylight_val_max": min(float(vals.max()) + slack(vals), 1.0),
        "overexposure_clip_max": min(float(clips.max()) + slack(clips), 1.0),
        "row_dup_max": min(float(dups.max()) + slack(dups), 1.0),
    }


def _hue_bands_overlap(a: BinModel, b: BinModel) -> bool:
    """Circular interval overlap of two learned hue bands (pure geometry)."""
    centers = abs(((a.hue_center_deg - b.hue_center_deg + 180.0) % 360.0) - 180.0)
    return centers <= a.hue_tol_deg + b.hue_tol_deg


def learn_profile(
    store: CalibrationStore, store_path: str | Path
) -> tuple[Profile, list[str]]:
    """Full recomputation: store → profile. Returns (profile, warnings).

    The store is first passed through :func:`learning_view`, which
    decides - purely geometrically and via epoch stamps - which stored
    evidence counts under the CURRENT configuration. A bin whose
    evidence is not learnable right now degrades to "untrained"
    (warning, absent from the profile, its sensor stays unavailable)
    instead of failing the whole relearn; the hard error remains only
    when nothing at all is learnable.
    """
    validate_store(store)
    store, warnings = learning_view(store)
    if not store.bins:
        raise CalibrationError("store declares no active bins")

    # One full-frame load per image; ROI grids are extracted on demand
    # per (image, roi) pair below. Images whose file vanished from the
    # archive are skipped with a loud warning instead of poisoning
    # every future relearn: the store may legitimately outlive
    # individual snapshot files.
    frames: dict[str, np.ndarray] = {}
    usable_images: list = []
    # Aspect ratios only of CURRENT-view frames: older-view images stay
    # in the set for their color samples and may legitimately have a
    # different format after a marked camera swap.
    aspects: set[tuple[int, int]] = set()
    for entry in store.images:
        try:
            img = load_image_rgb(resolve_image_path(store_path, entry.path))
        except ImageLoadError as exc:
            warnings.append(f"skipping calibration image: {exc}")
            continue
        frames[entry.path] = img
        if entry.view_epoch == store.view_epoch:
            width, height = img.size
            divisor = math.gcd(width, height)
            aspects.add((width // divisor, height // divisor))
        usable_images.append(entry)
    # Exact, threshold-free cross-check: relative coordinates are only
    # comparable between frames of identical aspect ratio. Mixed
    # aspects inside one view epoch mean the camera (or its stream
    # format) changed without the view being marked as changed.
    if len(aspects) > 1:
        warnings.append(
            "calibration images have differing aspect ratios "
            f"({sorted(aspects)}) - was the camera changed without "
            "marking the view as changed? Area evidence across formats "
            "is not comparable"
        )

    hsv_cache: dict[
        tuple[str, Roi], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    def hsv_for(path: str, roi: Roi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (path, roi)
        if key not in hsv_cache:
            arr = extract_working_roi(
                frames[path], roi, store.working_width, store.resample
            )
            hsv_cache[key] = rgb_to_hsv(arr)
        return hsv_cache[key]

    # 1) Color models from pooled sample rectangles. Every rect is
    # extracted through its own draw-time ROI grid (SampleRect.roi):
    # the pixels the user marked stay the pixels that are learned, no
    # matter how often the ROI changes afterwards. Color statistics
    # tolerate mixing extraction grids (hue does not depend on the
    # crop); area learning below never does.
    bins: list[BinModel] = []
    untrained: dict[str, str] = {}
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
            for sample in rects:
                grid = image_rect_in_roi(sample.rect, sample.roi)
                if grid is None:  # guarded by validate_store already
                    continue
                hue, sat, val = hsv_for(entry.path, sample.roi)
                height, width = sat.shape
                try:
                    x0, y0, x1, y1 = rect_to_pixels(
                        grid.x, grid.y, grid.w, grid.h, width, height
                    )
                except RoiError as exc:
                    # A rect that rounds to zero pixels on its grid
                    # carries no evidence: skip it instead of aborting
                    # the whole relearn.
                    warnings.append(
                        f"bin {decl.id}: skipping sample rect in "
                        f"{entry.path} - {exc}"
                    )
                    continue
                hue_parts.append(hue[y0:y1, x0:x1].ravel())
                sat_parts.append(sat[y0:y1, x0:x1].ravel())
                val_parts.append(val[y0:y1, x0:x1].ravel())
        try:
            if not hue_parts:
                raise CalibrationError(
                    f"bin {decl.id}: no usable sample rectangles in any "
                    "calibration image"
                )
            color = learn_color_model(
                np.concatenate(hue_parts),
                np.concatenate(sat_parts),
                np.concatenate(val_parts),
                bin_id=decl.id,
            )
        except CalibrationError as exc:
            untrained[decl.id] = str(exc)
            warnings.append(f"bin {decl.id}: untrained - {exc}")
            continue
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

    # Region mask per image on the CURRENT region's grid; None = whole
    # crop (rect fast path). The crop is integer-rounded per frame, so
    # the mask maps through each frame's actual crop box.
    poly_cache: dict[str, np.ndarray | None] = {}

    def poly_for(path: str) -> np.ndarray | None:
        if path not in poly_cache:
            if store.roi_polygons is None:
                poly_cache[path] = None
            else:
                img = frames[path]
                sx0, sy0, sx1, sy1 = roi_to_pixels(
                    store.roi, img.width, img.height
                )
                crop = Roi(
                    x=sx0 / img.width,
                    y=sy0 / img.height,
                    w=(sx1 - sx0) / img.width,
                    h=(sy1 - sy0) / img.height,
                )
                _hue, sat, _val = hsv_for(path, store.roi)
                poly_cache[path] = region_mask(
                    store.roi_polygons, sat.shape[1], sat.shape[0], crop
                )
        return poly_cache[path]

    def masked_bin_mask(entry_path: str, model: BinModel):
        hue, sat, val = hsv_for(entry_path, store.roi)
        mask = bin_mask(hue, sat, val, model)
        poly = poly_for(entry_path)
        if poly is not None:
            mask &= poly
        denom = int(poly.sum()) if poly is not None else mask.size
        return mask, denom

    # 1.5) Shape models from the PRESENT-labeled images, referenced
    # exactly through the bin's sample rectangles: the observed shape
    # is the connected component touching a rect the user drew on the
    # lid - never "the largest blob", which under harsh light can be a
    # background object and would poison the learned shape forever.
    # Present images without current-epoch rects contribute no shape
    # observation (their area evidence below stays untouched).
    shape_obs: dict[str, list[tuple[int, int, int, int]]] = {}
    for model in bins:
        observations: list[tuple[int, int, int, int]] = []
        for entry in usable_images:
            if model.id not in entry.present:
                continue
            rects = entry.samples.get(model.id, [])
            if not rects:
                continue
            mask, denom = masked_bin_mask(entry.path, model)
            height, width = mask.shape
            seed = np.zeros_like(mask)
            for sample in rects:
                grid = image_rect_in_roi(sample.rect, store.roi)
                if grid is None:
                    continue
                try:
                    x0, y0, x1, y1 = rect_to_pixels(
                        grid.x, grid.y, grid.w, grid.h, width, height
                    )
                except RoiError:
                    continue
                seed[y0:y1, x0:x1] = True
            component = seeded_component(mask, seed)
            if component is None:
                continue
            ys, xs = np.nonzero(component)
            box_w = int(xs.max()) - int(xs.min()) + 1
            box_h = int(ys.max()) - int(ys.min()) + 1
            observations.append((int(component.sum()), box_w, box_h, denom))
        shape_obs[model.id] = observations
    # Pooled lid-aspect span across ALL bins: the installation's
    # measured scale of position-induced aspect variation (each bin is
    # a lid observed at a different spot).
    all_aspects = [
        math.log(w / h)
        for obs in shape_obs.values()
        for (_a, w, h, _d) in obs
    ]
    pooled_span = (
        max(all_aspects) - min(all_aspects) if len(all_aspects) >= 2 else 0.0
    )
    for model in bins:
        if shape_obs[model.id]:
            model.learning_stats.update(
                shape_bounds(shape_obs[model.id], pooled_span)
            )

    # 2) Area thresholds on top of the final color AND shape models,
    # computed under the CURRENT region only, with exactly the
    # plausible-only component selection detection uses (pipeline
    # identity: thresholds must be learned on the areas that will be
    # measured at runtime).
    trained: list[BinModel] = []
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
            mask, denom = masked_bin_mask(entry.path, model)
            selected = select_component(mask, model, denom)
            area_frac = (selected[0] / denom) if selected else 0.0
            if target is pos_areas and area_frac <= 0.0:
                # The shape filter (or the region) leaves no plausible
                # blob in a present-labeled image: exclude the
                # observation with a warning instead of hard-failing
                # the bin (same policy as stale-geometry evidence).
                warnings.append(
                    f"bin {model.id}: present-labeled {entry.path} yields "
                    "no plausible blob under the current region/shape "
                    "model - observation excluded (the threshold is then "
                    "learned without this worst case; if this happens for "
                    "typical frames, widen the region or add samples from "
                    "such frames)"
                )
                continue
            target.append(area_frac)
        try:
            result = learn_area_threshold(pos_areas, neg_areas, bin_id=model.id)
        except CalibrationError as exc:
            untrained[model.id] = str(exc)
            warnings.append(f"bin {model.id}: untrained - {exc}")
            continue
        warnings.extend(result.warnings)
        model.min_area_frac = result.min_area_frac
        model.learning_stats.update(result.stats)
        trained.append(model)
    bins = trained
    if not bins:
        raise CalibrationError(
            "no bin could be trained: "
            + "; ".join(f"{b}: {reason}" for b, reason in untrained.items())
        )

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
    row_dups: list[float] = []
    # Gate statistics only from CURRENT-view frames: the current ROI
    # crop of an old-view frame depicts a different scene region, and
    # its light statistics would poison the gates (same reasoning as
    # clearing gate_samples on a view bump).
    for e in usable_images:
        if e.view_epoch != store.view_epoch:
            continue
        _hue, sat, val = hsv_for(e.path, store.roi)
        median_sats.append(float(np.median(sat)))
        clip_fracs.append(float(np.mean(val >= clip_floor)))
        median_vals.append(float(np.median(val)))
        row_dups.append(
            row_duplicate_fraction(
                extract_working_roi(
                    frames[e.path], store.roi, store.working_width, store.resample
                )
            )
        )
    if not median_sats:
        raise CalibrationError("store contains no usable calibration images")
    daylight_sat_min = min(median_sats)
    overexposure_clip_max = max(clip_fracs)
    daylight_val_max = max(median_vals)
    row_dup_max = max(row_dups)
    daylight_stats = {
        "n_images": len(median_sats),
        "min_median_sat": daylight_sat_min,
        "median_of_medians": float(np.median(np.asarray(median_sats))),
        "max_clip_frac": overexposure_clip_max,
        "max_median_val": daylight_val_max,
        "max_row_dup_frac": row_dup_max,
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
        row_dup_max=row_dup_max,
        roi_polygons=(
            None
            if store.roi_polygons is None
            else [list(ring) for ring in store.roi_polygons]
        ),
        daylight_stats=daylight_stats,
        bins=bins,
    )
    return profile, warnings
