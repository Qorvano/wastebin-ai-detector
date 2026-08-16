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

from .ccl import largest_component_area, seeded_component
from .region import interior_depth, region_mask
from .color import (
    HUE_DEG_PER_SEXTANT,
    RGB_8BIT_LEVELS,
    circular_dist_deg,
    circular_mean_deg,
    fit_vonmises_uniform_mixture,
    rgb_to_hsv,
    vonmises_kappa_from_resultant,
    weighted_percentile,
)
from .detect import (
    bin_mask,
    edge_band_filter,
    edge_band_min_frac,
    exclusive_bin_masks,
    resolve_candidates,
    row_duplicate_fraction,
    select_component,
)
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


def learn_edge_band(touch_depth_fracs: list[float]) -> dict[str, Any]:
    """Learned region-edge band from BOUNDARY-TOUCHING lid depths.

    ``touch_depth_fracs`` holds, per present-frame lid observation
    whose component touches the region boundary (in store order), the
    maximum interior depth it reaches, as a FRACTION of the working
    grid width (resolution-independent: with working_width None the
    grids are native crops and may differ between frames and between
    learn and detect). Interior observations are deliberately absent:
    their depth measures the parking position, not lid geometry, and
    a band learned from them would veto legitimate lids parked against
    the contour.

    The band is the largest depth that provably keeps every calibrated
    boundary-touching lid: the minimum of the series minus the
    established successive-difference slack (the shape_bounds /
    derive_quality_gates convention; it also absorbs the ±1 px wobble
    of integer crop rounding). Clamped at 0. Without touching
    observations the band stays inactive - no evidence about touching
    lids, no filtering (the mechanism never invents separation).
    """
    stats: dict[str, Any] = {"region_edge_depth_n": len(touch_depth_fracs)}
    if not touch_depth_fracs:
        return stats
    series = np.asarray(touch_depth_fracs, dtype=np.float64)
    obs_min = float(series.min())
    stats["region_edge_depth_obs_min_frac"] = obs_min
    slack = (
        float(np.median(np.abs(np.diff(series)))) if series.size >= 2 else 0.0
    )
    stats["region_edge_depth_min_frac"] = max(obs_min - slack, 0.0)
    return stats


@dataclass
class ColorLearnResult:
    hue_center_deg: float
    hue_tol_deg: float
    sat_min: float
    val_min: float
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def learn_color_model(
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    *,
    bin_id: str = "?",
    weights: np.ndarray | None = None,
) -> ColorLearnResult:
    """Learn one bin's color model from pooled sample pixels (1-D arrays).

    Robust against junk pixels (overexposed highlights, edge artifacts):
    a von Mises + uniform mixture separates the coherent color
    population from scatter, and band and floors are learned from the
    coherent pixels only. Field measurements showed that plain
    percentiles break as soon as more than the percentile share of the
    sample is junk.

    Optional ``weights`` (per pixel, aligned with the pooled arrays)
    weight every statistic - the caller uses 1/(pixels of that image)
    so each sample IMAGE carries one vote regardless of rectangle size
    (field evidence: one big washed-out rectangle repeatedly outvoted
    several small clean ones pixel-wise). ``None`` means unit weights,
    i.e. every pixel votes.

    The mixture fit runs from two deterministic starts (histogram
    density mode and mass mean direction) on BOTH paths, weighted or
    not: a washed-out patch produces a tight quantization spike that
    traps a single local start (field failure: a lid model collapsed
    onto one hue value). Unweighted results therefore differ from
    releases before the two-start change wherever that trap applied;
    that difference is the fix, not a regression.
    """
    warnings: list[str] = []
    n_px = int(sat.size)
    if n_px == 0:
        raise CalibrationError(f"bin {bin_id}: no sample pixels")
    if weights is not None and np.asarray(weights).size != n_px:
        raise ValueError(
            f"weights/pixels mismatch: {np.asarray(weights).size} vs {n_px}"
        )
    # Raises if all hues are NaN; resultant is the stats-reported
    # concentration of the full (valid) pool.
    _center0, resultant = circular_mean_deg(hue, weights=weights)
    valid_mask = ~np.isnan(hue)
    valid_hue = np.asarray(hue, dtype=np.float64)[valid_mask]
    sat_valid = np.asarray(sat, dtype=np.float64)[valid_mask]
    val_valid = np.asarray(val, dtype=np.float64)[valid_mask]
    w_valid = (
        None
        if weights is None
        else np.asarray(weights, dtype=np.float64).ravel()[valid_mask]
    )
    n_valid = int(valid_hue.size)

    def pct(values: np.ndarray, wts: np.ndarray | None, q: float) -> float:
        """Percentile under the pool's weighting convention."""
        if wts is None:
            return float(np.percentile(values, q))
        return weighted_percentile(values, wts, q)

    # chroma = sat · val, since sat = c/maxc and val = maxc.
    chroma_valid = sat_valid * val_valid

    # Deterministic initialization from the hue histogram. The bin
    # width is the hue quantization of an 8-bit pixel at the median
    # chroma of the sample (60°/(255·c)), a structural resolution, not
    # a chosen granularity.
    median_chroma = pct(chroma_valid, w_valid, 50.0)
    bin_width = HUE_DEG_PER_SEXTANT / (RGB_8BIT_LEVELS * median_chroma)
    n_bins = max(int(np.ceil(360.0 / bin_width)), 1)
    hist, edges = np.histogram(
        valid_hue, bins=n_bins, range=(0.0, 360.0), weights=w_valid
    )
    peak = int(np.argmax(hist))
    mu0 = float((edges[peak] + edges[peak + 1]) / 2.0)
    # Kappa cap: below the finest representable hue step, concentration
    # is not measurable (same quantization logic as the tol floor).
    max_chroma = float(np.max(chroma_valid))
    finest_quantum_deg = HUE_DEG_PER_SEXTANT / (RGB_8BIT_LEVELS * max_chroma)
    sigma_min_rad = float(np.deg2rad(finest_quantum_deg / 2.0))
    kappa_max = 1.0 / (sigma_min_rad * sigma_min_rad)

    def fit_from(mu_init: float) -> Any:
        # Moment start for the mixture weight: under the model, the
        # share of pixel MASS within 90° (the geometric half-circle
        # boundary) of the center is w + (1-w)/2, so
        # w_start = 2·(share − 1/2).
        within = circular_dist_deg(valid_hue, mu_init) < 90.0
        if w_valid is None:
            share = float(within.mean())
        else:
            share = float(w_valid[within].sum() / w_valid.sum())
        w_start = 2.0 * (share - 0.5)
        if bool(within.any()):
            _, r_within = circular_mean_deg(
                valid_hue[within],
                weights=None if w_valid is None else w_valid[within],
            )
            kappa_start = vonmises_kappa_from_resultant(r_within)
        else:
            kappa_start = 0.0
        return fit_vonmises_uniform_mixture(
            valid_hue,
            init_center_deg=mu_init,
            init_weight=w_start,
            init_kappa=kappa_start,
            kappa_max=kappa_max,
            weights=w_valid,
        )

    fit = fit_from(mu0)
    # Second start at the mass-mean direction: the histogram argmax is
    # a DENSITY mode and can sit on a tight junk spike that carries
    # little total mass, trapping the local EM there. The circular mean
    # of all mass is the moment start for the heaviest mode. Standard
    # deterministic multi-start: the better (weighted) likelihood wins.
    # A zero resultant means the mean direction is undefined (perfectly
    # cancelling hues) - then only the histogram start exists.
    if resultant > 0.0:
        fit_mean = fit_from(_center0)
        if fit_mean.log_likelihood > fit.log_likelihood:
            fit = fit_mean
    coherent = fit.posterior >= COHERENT_POSTERIOR_MIN
    n_coherent = int(coherent.sum())
    # Junk share and the majority guard run on observation MASS: with
    # per-image weights a huge junk rectangle must count as one bad
    # image, not as a pixel majority.
    if w_valid is None:
        coherent_share = (n_coherent / n_valid) if n_valid else 0.0
    else:
        coherent_share = float(w_valid[coherent].sum() / w_valid.sum())
    junk_fraction = 1.0 - coherent_share
    # Weighted pools vote per IMAGE, unweighted ones per pixel; the
    # messages below must name the unit they actually measured.
    unit = "pixels" if w_valid is None else "evidence (one vote per image)"
    if coherent_share <= COHERENT_MAJORITY_MIN:
        raise CalibrationError(
            f"bin {bin_id}: only {coherent_share:.0%} of the sample "
            f"{unit} is coherent ({n_coherent} of {n_valid} pixels lie in "
            "the majority color) - the samples have no consistent majority "
            "color. Re-draw them on a colored lid area; grey/black lids "
            "cannot be color-calibrated - attach a small colored marker to "
            "the lid and sample that instead"
        )

    center = fit.center_deg
    dist = circular_dist_deg(valid_hue[coherent], center)
    tol = pct(
        dist,
        None if w_valid is None else w_valid[coherent],
        HUE_TOL_PERCENTILE,
    )
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
    w_coherent = None if w_valid is None else w_valid[coherent]
    sat_min = pct(sat_valid[coherent], w_coherent, SV_MIN_PERCENTILE)
    val_min = pct(val_valid[coherent], w_coherent, SV_MIN_PERCENTILE)
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
            f"bin {bin_id}: {junk_fraction:.0%} of the sample {unit} is "
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
            # Mass share, not n_coherent_px/n_valid_hue_px: with
            # per-image weights these differ by design.
            "junk_fraction": junk_fraction,
            "junk_fraction_is_per_image": w_valid is not None,
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
        weight_parts: list[np.ndarray] = []
        n_sample_images = 0
        for entry in usable_images:
            rects = entry.samples.get(decl.id, [])
            if not rects:
                continue
            n_sample_images += 1
            img_hue_parts: list[np.ndarray] = []
            img_sat_parts: list[np.ndarray] = []
            img_val_parts: list[np.ndarray] = []
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
                img_hue_parts.append(hue[y0:y1, x0:x1].ravel())
                img_sat_parts.append(sat[y0:y1, x0:x1].ravel())
                img_val_parts.append(val[y0:y1, x0:x1].ravel())
            img_px = int(sum(part.size for part in img_hue_parts))
            if img_px == 0:
                continue
            hue_parts.extend(img_hue_parts)
            sat_parts.extend(img_sat_parts)
            val_parts.extend(img_val_parts)
            # One image, one vote: every pixel carries 1/(pixels of its
            # image), so a huge rectangle in one frame cannot outvote
            # several small clean ones from other frames.
            weight_parts.append(np.full(img_px, 1.0 / img_px))
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
                weights=np.concatenate(weight_parts),
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

    # Region mask (None = whole crop, the rect fast path) and interior
    # depth of the region universe. Both depend ONLY on the geometry -
    # the frame's integer-rounded crop and the working grid - never on
    # image content, so they are cached by that geometry: a calibration
    # set from one camera shares a single computation instead of
    # repeating it per image. Measured on a 25-image set at working
    # width 640, the per-image depth map was the single largest cost of
    # a relearn.
    geom_cache: dict[tuple, tuple[np.ndarray | None, np.ndarray]] = {}

    def _geometry(path: str) -> tuple[np.ndarray | None, np.ndarray]:
        img = frames[path]
        sx0, sy0, sx1, sy1 = roi_to_pixels(store.roi, img.width, img.height)
        crop = Roi(
            x=sx0 / img.width,
            y=sy0 / img.height,
            w=(sx1 - sx0) / img.width,
            h=(sy1 - sy0) / img.height,
        )
        _hue, sat, _val = hsv_for(path, store.roi)
        key = (sat.shape, crop)
        if key not in geom_cache:
            poly = (
                None
                if store.roi_polygons is None
                else region_mask(
                    store.roi_polygons, sat.shape[1], sat.shape[0], crop
                )
            )
            universe = np.ones(sat.shape, dtype=bool) if poly is None else poly
            geom_cache[key] = (poly, interior_depth(universe))
        return geom_cache[key]

    def poly_for(path: str) -> np.ndarray | None:
        return _geometry(path)[0]

    def depth_for(path: str) -> np.ndarray:
        return _geometry(path)[1]

    def seed_for(entry, model_id: str, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        seed = np.zeros(shape, dtype=bool)
        for sample in entry.samples.get(model_id, []):
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
        return seed

    # Exclusive per-image masks over the CURRENT competitor set: a
    # pixel belongs to at most one bin, so two bins whose learned
    # bands overlap can no longer both count it. Cached per image and
    # rebuilt whenever the competitor set changes (see the elimination
    # loop below), because a pixel's winner depends on who competes.
    exclusive_cache: dict[str, list[np.ndarray]] = {}

    # How much of each bin's plain color mask goes to a competitor?
    # Pure measurement, reported with the existing overlap warning: two
    # bins with nearly identical learned colors shred each other, and
    # the resulting threshold can collapse to a few pixels.
    exclusive_kept: dict[str, int] = {}
    exclusive_total: dict[str, int] = {}

    def exclusive_masks_for(path: str, models: list[BinModel]):
        if path not in exclusive_cache:
            hue, sat, val = hsv_for(path, store.roi)
            plain = [bin_mask(hue, sat, val, m) for m in models]
            masks = exclusive_bin_masks(
                hue, sat, val, models, enabled=True
            )
            for model, before, after in zip(models, plain, masks):
                exclusive_total[model.id] = exclusive_total.get(
                    model.id, 0
                ) + int(before.sum())
                exclusive_kept[model.id] = exclusive_kept.get(
                    model.id, 0
                ) + int(after.sum())
            exclusive_cache[path] = masks
        return exclusive_cache[path]

    def layer1_mask(entry_path: str, model: BinModel, models: list[BinModel]):
        """The bin's exclusive color mask under the current region."""
        index = models.index(model)
        mask = exclusive_masks_for(entry_path, models)[index]
        poly = poly_for(entry_path)
        if poly is not None:
            mask = mask & poly
        denom = int(poly.sum()) if poly is not None else mask.size
        return mask, denom

    def banded_bin_mask(
        entry_path: str, model: BinModel, models: list[BinModel]
    ):
        """layer1_mask plus the bin's edge-band filter: exactly the
        pixels detection counts before the occupancy veto (pipeline
        identity, shared edge_band_filter). Whole boundary-touching
        components that never reach the learned depth are dropped;
        nothing else changes, in particular not the denominator."""
        mask, denom = layer1_mask(entry_path, model, models)
        band_frac = edge_band_min_frac(model)
        if band_frac is None:
            return mask, denom
        band_px = band_frac * mask.shape[1]
        if band_px <= 1.0:
            # One pixel or less on this grid removes nothing (every
            # region pixel has depth >= 1) - same no-op identity as in
            # detect().
            return mask, denom
        return edge_band_filter(mask, depth_for(entry_path), band_px), denom

    def _lost_to_the_band(
        entry_path: str, model: BinModel, models: list[BinModel]
    ) -> bool:
        """Would this frame have had a plausible blob WITHOUT the edge
        band? Measured on the unbanded layer-1 mask, because the banded
        candidate cannot answer it (it is already filtered)."""
        if edge_band_min_frac(model) is None:
            return False
        raw_mask, raw_denom = layer1_mask(entry_path, model, models)
        raw = select_component(raw_mask, model, raw_denom)
        return raw is not None and raw[0] > 0

    def universe_for(path: str) -> np.ndarray:
        poly = poly_for(path)
        if poly is not None:
            return poly
        _hue, sat, _val = hsv_for(path, store.roi)
        return np.ones(sat.shape, dtype=bool)

    # The competition set must equal the set that ships in the profile,
    # or a bin that steals pixels while learning but never reports
    # would leave the survivors' thresholds measured against a
    # competitor detection does not have. A bin that cannot learn a
    # threshold is dropped and everything is recomputed; the set only
    # shrinks, and dropping a competitor can only make the survivors'
    # areas larger, so a drop can never cause a new failure - the loop
    # terminates, in practice after one round.
    competitors = list(bins)
    # Warnings about bins that dropped out survive the round that
    # dropped them (round-local warnings are discarded when the set
    # changes and everything is recomputed).
    dropout_warnings: list[str] = []
    while True:
        exclusive_cache.clear()
        exclusive_kept.clear()
        exclusive_total.clear()
        # Stats measured against the PREVIOUS competitor set would be
        # stale now (a dropped competitor changes every mask), so no
        # round may inherit them.
        for model in competitors:
            for key in (
                "region_edge_clutter_max_frac",
                "region_edge_separable",
                "veto_qualify_min_area_frac",
                "veto_qualify_separable",
                "veto_qualify_provisional",
            ):
                model.learning_stats.pop(key, None)
        round_warnings: list[str] = []
        round_untrained: dict[str, str] = {}

        # 1.5a) UNBANDED shape observations from the PRESENT-labeled
        # images, referenced exactly through the bin's sample
        # rectangles: the observed shape is the connected component
        # touching a rect the user drew on the lid - never "the largest
        # blob", which under harsh light can be a background object and
        # would poison the learned shape forever. Present images
        # without current-epoch rects contribute no observation (their
        # area evidence below stays untouched). Observations whose
        # component TOUCHES the region boundary additionally record how
        # deep it reaches - the evidence behind the edge band. Interior
        # observations carry NO information about the reach of a
        # boundary-touching lid (their depth measures the parking
        # position, not lid geometry) and must not vote, or the band
        # would silently veto legitimate lids parked against the
        # contour. Depths are normalized by the working grid width so
        # the band survives resolution changes (working_width None runs
        # on native crops). All of this MUST be measured unbanded
        # (anything else is circular). The occupancy veto is
        # deliberately NOT applied here: it can only delete whole
        # components, never reshape one, so it could only discard the
        # user's own seeded ground truth.
        shape_obs: dict[str, list[tuple[int, int, int, int]]] = {}
        edge_depths: dict[str, list[float]] = {}
        for model in competitors:
            observations: list[tuple[int, int, int, int]] = []
            touch_depths: list[float] = []
            for entry in usable_images:
                if model.id not in entry.present:
                    continue
                if not entry.samples.get(model.id):
                    continue
                mask, denom = layer1_mask(entry.path, model, competitors)
                component = seeded_component(
                    mask, seed_for(entry, model.id, mask.shape)
                )
                if component is None:
                    continue
                ys, xs = np.nonzero(component)
                box_w = int(xs.max()) - int(xs.min()) + 1
                box_h = int(ys.max()) - int(ys.min()) + 1
                observations.append(
                    (int(component.sum()), box_w, box_h, denom)
                )
                depth = depth_for(entry.path)
                if bool((depth[component] == 1).any()):
                    touch_depths.append(
                        float(depth[component].max()) / mask.shape[1]
                    )
            shape_obs[model.id] = observations
            edge_depths[model.id] = touch_depths
        for model in competitors:
            model.learning_stats.update(learn_edge_band(edge_depths[model.id]))

        # No banded re-measurement of shapes is needed: the reach filter
        # only drops WHOLE components, and every calibrated lid
        # component reaches the band by construction, so the
        # observations above are exactly what detection will see.
        # Pooled lid-aspect span across ALL bins: the installation's
        # measured scale of position-induced aspect variation (each bin
        # is a lid observed at a different spot).
        all_aspects = [
            math.log(w / h)
            for obs in shape_obs.values()
            for (_a, w, h, _d) in obs
        ]
        pooled_span = (
            max(all_aspects) - min(all_aspects)
            if len(all_aspects) >= 2
            else 0.0
        )
        for model in competitors:
            if shape_obs[model.id]:
                model.learning_stats.update(
                    shape_bounds(shape_obs[model.id], pooled_span)
                )

        # 2a) Areas under the pixel layer only. Their outcome becomes
        # each bin's qualification for the occupancy veto: only a bin
        # whose own calibration separates, is not provisional and whose
        # blob is at least the weakest lid ever confirmed for it may
        # erase another bin's evidence.
        qualified_failures: list[str] = []
        zero_area_positives: dict[str, list[str]] = {}
        for model in competitors:
            pos_areas: list[float] = []
            neg_areas: list[float] = []
            for entry in usable_images:
                if model.id in entry.present:
                    target = pos_areas
                elif model.id in entry.absent:
                    target = neg_areas
                else:
                    continue
                mask, denom = banded_bin_mask(entry.path, model, competitors)
                selected = select_component(mask, model, denom)
                area_frac = (selected[0] / denom) if selected else 0.0
                if target is pos_areas and area_frac <= 0.0:
                    zero_area_positives.setdefault(model.id, []).append(
                        entry.path
                    )
                    continue
                target.append(area_frac)
            try:
                pass_a = learn_area_threshold(
                    pos_areas, neg_areas, bin_id=model.id
                )
            except CalibrationError as exc:
                round_untrained[model.id] = str(exc)
                dropout_warnings.append(f"bin {model.id}: untrained - {exc}")
                # A bin that drops out here never reaches pass 2b, so
                # the per-image diagnosis explaining WHY it has no
                # usable positive must be emitted now or never.
                for path in zero_area_positives.get(model.id, []):
                    dropout_warnings.append(
                        f"bin {model.id}: present-labeled {path} yields no "
                        "plausible blob under the current region/shape "
                        "model - observation excluded"
                    )
                qualified_failures.append(model.id)
                continue
            model.learning_stats["veto_qualify_min_area_frac"] = (
                pass_a.stats.get("min_pos_area_frac")
            )
            model.learning_stats["veto_qualify_separable"] = bool(
                pass_a.stats.get("separable", False)
            )
            model.learning_stats["veto_qualify_provisional"] = bool(
                pass_a.stats.get("provisional", True)
            )
        if qualified_failures:
            competitors = [
                m for m in competitors if m.id not in qualified_failures
            ]
            untrained.update(round_untrained)
            if not competitors:
                break
            continue

        # 2b) Final areas under BOTH layers, measured exactly as
        # detection measures them (shared resolve_candidates), images
        # outer because the veto is a relation between bins in one
        # frame.
        pos_by_bin: dict[str, list[float]] = {m.id: [] for m in competitors}
        neg_by_bin: dict[str, list[float]] = {m.id: [] for m in competitors}
        clutter_by_bin: dict[str, float | None] = {
            m.id: None for m in competitors
        }
        for entry in usable_images:
            hue, sat, val = hsv_for(entry.path, store.roi)
            poly = poly_for(entry.path)
            depth = depth_for(entry.path)
            denom = int(poly.sum()) if poly is not None else hue.size
            candidates = resolve_candidates(
                hue,
                sat,
                val,
                competitors,
                universe=universe_for(entry.path),
                region=poly,
                depth=depth,
                denom=denom,
                exclusion=True,
            )
            for model, cand in zip(competitors, candidates):
                if model.id in entry.present:
                    target = pos_by_bin[model.id]
                elif model.id in entry.absent:
                    target = neg_by_bin[model.id]
                else:
                    continue
                band_frac = edge_band_min_frac(model)
                if target is neg_by_bin[model.id] and band_frac is not None:
                    # Diagnosis on the unbanded layer-1 mask: how deep
                    # does boundary-attached clutter reach in this bin's
                    # color? A component entering from outside must
                    # cross the outermost region pixel layer (depth ==
                    # 1) - a structural criterion, no threshold.
                    raw_mask, _raw_denom = layer1_mask(
                        entry.path, model, competitors
                    )
                    attached = seeded_component(
                        raw_mask, raw_mask & (depth == 1)
                    )
                    if attached is not None:
                        reach = (
                            float(depth[attached].max()) / raw_mask.shape[1]
                        )
                        previous = clutter_by_bin[model.id]
                        clutter_by_bin[model.id] = (
                            reach if previous is None else max(previous, reach)
                        )
                area_frac = cand.area / denom
                if target is pos_by_bin[model.id] and area_frac <= 0.0:
                    # The region, the shape filter, the edge band or
                    # another bin's detected area leaves no plausible
                    # blob in a present-labeled image: exclude the
                    # observation with a warning instead of hard-failing
                    # the bin (same policy as stale-geometry evidence),
                    # naming the cause so the cure is obvious.
                    if cand.excluded_by is not None:
                        round_warnings.append(
                            f"bin {model.id}: present-labeled {entry.path} "
                            f"lies inside the detected area of "
                            f"{cand.excluded_by} - observation excluded; "
                            "two bins cannot occupy the same spot, so "
                            "check the labels or the samples of both"
                        )
                    elif _lost_to_the_band(entry.path, model, competitors):
                        round_warnings.append(
                            f"bin {model.id}: present-labeled {entry.path} "
                            "loses its only plausible blob to the edge "
                            "band (boundary-touching, shallower than "
                            "every calibrated touching lid) - observation "
                            "excluded; draw a sample rect on this pose so "
                            "the band learns it"
                        )
                    else:
                        round_warnings.append(
                            f"bin {model.id}: present-labeled {entry.path} "
                            "yields no plausible blob under the current "
                            "region/shape model - observation excluded "
                            "(the threshold is then learned without this "
                            "worst case; if this happens for typical "
                            "frames, widen the region or add samples from "
                            "such frames)"
                        )
                    continue
                target.append(area_frac)

        trained: list[BinModel] = []
        final_failures: list[str] = []
        for model in competitors:
            band_frac = edge_band_min_frac(model)
            if band_frac is not None:
                clutter_max = clutter_by_bin[model.id]
                model.learning_stats["region_edge_clutter_max_frac"] = (
                    clutter_max
                )
                separable = clutter_max is None or clutter_max < band_frac
                model.learning_stats["region_edge_separable"] = separable
                if not separable:
                    round_warnings.append(
                        f"bin {model.id}: boundary clutter in absent frames "
                        f"reaches interior depth {clutter_max:.4f} of the "
                        f"grid width, at or beyond the learned edge band of "
                        f"{band_frac:.4f} - such clutter still counts toward "
                        "detection; check the region contour or add absent "
                        "labels from that light"
                    )
            try:
                result = learn_area_threshold(
                    pos_by_bin[model.id],
                    neg_by_bin[model.id],
                    bin_id=model.id,
                )
            except CalibrationError as exc:
                round_untrained[model.id] = str(exc)
                dropout_warnings.append(f"bin {model.id}: untrained - {exc}")
                final_failures.append(model.id)
                continue
            round_warnings.extend(result.warnings)
            model.min_area_frac = result.min_area_frac
            model.learning_stats.update(result.stats)
            trained.append(model)
        if final_failures:
            # The round is recomputed without these bins, so their
            # per-image diagnoses would otherwise be lost with the
            # round-local warnings - they are the only explanation of
            # why the bin disappeared.
            dropout_warnings.extend(
                w
                for w in round_warnings
                if any(w.startswith(f"bin {b}: ") for b in final_failures)
            )
            competitors = [m for m in competitors if m.id not in final_failures]
            untrained.update(round_untrained)
            if not competitors:
                break
            continue
        untrained.update(round_untrained)
        warnings.extend(round_warnings)
        competitors = trained
        break

    warnings.extend(dropout_warnings)
    bins = competitors
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

    for model in bins:
        total = exclusive_total.get(model.id, 0)
        model.learning_stats["exclusive_keep_frac"] = (
            1.0 if total == 0 else exclusive_kept.get(model.id, 0) / total
        )

    # 4) Ambiguity diagnosis: overlapping learned hue bands.
    for i, a in enumerate(bins):
        for b in bins[i + 1 :]:
            if _hue_bands_overlap(a, b):
                losses = " ".join(
                    f"{m.id} keeps "
                    f"{m.learning_stats.get('exclusive_keep_frac', 1.0):.0%} "
                    "of its matching pixels."
                    for m in (a, b)
                )
                warnings.append(
                    f"bins {a.id} and {b.id}: learned hue bands overlap "
                    f"({a.hue_center_deg:.0f}°±{a.hue_tol_deg:.0f}° vs "
                    f"{b.hue_center_deg:.0f}°±{b.hue_tol_deg:.0f}°) - every "
                    "contested pixel goes to the closer color, so the two "
                    f"bins compete for evidence. {losses}"
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
        # Thresholds above were measured under mutual exclusion, so
        # detection must apply it too (see resolve_candidates).
        mutual_exclusion=True,
    )
    return profile, warnings
