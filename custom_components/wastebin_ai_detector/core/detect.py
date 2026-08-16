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

import math

from .ccl import component_regions, seeded_component
from .color import RGB_8BIT_LEVELS, circular_dist_deg, rgb_to_hsv
from .imageio import extract_working_roi, load_image_rgb, roi_to_pixels
from .profile import BinModel, Profile, Roi, validate_profile
from .region import interior_depth, region_mask

# Singular-vs-plural rule (same convention as the color-floor warning):
# with fewer than two shape observations there is no between-sample
# variation to learn bounds from, so the plausibility filter stays
# inactive rather than guessing.
SHAPE_MIN_OBSERVATIONS = 2


def shape_plausible(
    model: BinModel,
    area_frac: float,
    box: tuple[int, int, int, int],
    area_px: int,
) -> bool:
    """Is this component geometrically plausible as the bin's lid?

    Bounds are learned per bin from the calibration images (extremum
    plus the established successive-difference slack, computed at learn
    time and stored in learning_stats). The filter is inactive until
    enough shape observations exist - then any component is plausible,
    which is exactly the pre-3b behavior.
    """
    stats = model.learning_stats
    if int(stats.get("shape_n", 0)) < SHAPE_MIN_OBSERVATIONS:
        return True
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0 or area_frac <= 0.0:
        return False
    # Log-space comparison with the learned bounds: same computation as
    # at learn time, so an observation can never fall outside its own
    # extremum through a floating-point round-trip. Deliberately no
    # area criterion (see shape_bounds).
    log_aspect = math.log(box_w / box_h)
    fill = area_px / (box_w * box_h)
    return (
        stats["shape_log_aspect_min"] <= log_aspect <= stats["shape_log_aspect_max"]
        and fill >= stats["shape_fill_min"]
    )


def edge_band_min_frac(model: BinModel) -> float | None:
    """The bin's active edge band as a fraction of the working grid
    width, or None when the band is off.

    The band is the learned minimum reach of this bin's BOUNDARY-
    TOUCHING lid observations (see learn_edge_band). Activation
    follows the singular-vs-plural rule (SHAPE_MIN_OBSERVATIONS, as
    for the shape filter). The fraction is converted to pixels at the
    application site, where the structural no-op identity lives: every
    region pixel has depth >= 1, so a band of one pixel or less on
    that grid removes nothing and is skipped there.
    """
    stats = model.learning_stats
    if int(stats.get("region_edge_depth_n", 0)) < SHAPE_MIN_OBSERVATIONS:
        return None
    band = stats.get("region_edge_depth_min_frac")
    if band is None:
        return None
    band = float(band)
    if band <= 0.0:
        return None
    return band


def edge_band_filter(
    mask: np.ndarray, depth: np.ndarray, band: float
) -> np.ndarray:
    """Drop BOUNDARY-TOUCHING components that never reach ``band``.

    Shared by detection and area learning (pipeline identity). The two
    criteria are both structural, no thresholds:
    - touching = the component contains a pixel of the outermost region
      layer (depth == 1); anything entering from outside the drawn
      contour must cross that layer,
    - reach = the component contains a pixel at depth >= band, the
      learned minimum interior depth of this bin's calibrated lids.
    A component confined to the boundary rim is background cut by the
    region line (field failure: a hedge sliver hugging the contour) and
    is dropped whole. Components that never touch the boundary are
    NEVER filtered, wherever they sit - bins may stand anywhere inside
    the region (position independence is a hard invariant). Surviving
    components keep every pixel; the denominator is untouched.
    """
    touching = seeded_component(mask, mask & (depth == 1))
    if touching is None:
        return mask
    reached = seeded_component(touching, touching & (depth >= band))
    interior = mask & ~touching
    if reached is None:
        return interior
    return interior | reached


def select_component(
    mask, model: BinModel, denom: int
) -> tuple[
    int, tuple[int, int, int, int], tuple[float, float], tuple[int, int]
] | None:
    """The LARGEST PLAUSIBLE component of a bin's color mask, as
    (area, half-open box, centroid, seed pixel).

    Shared by detection and area learning (pipeline identity: learned
    thresholds must be computed on exactly the areas detection sees).
    No plausible component means no evidence: honest zero, so a hedge
    fringe or a sunlit body streak can no longer stand in for a lid.
    The seed pixel lets callers re-derive the exact pixel set without
    a second labelling pass.
    """
    for area, box, centroid, seed in component_regions(mask, with_seed=True):
        if shape_plausible(model, area / denom, box, area):
            return area, box, centroid, seed
    return None


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


def bin_match(
    hue: np.ndarray, sat: np.ndarray, val: np.ndarray, model: BinModel
) -> tuple[np.ndarray, np.ndarray]:
    """Acceptance mask plus circular hue distance for one bin.

    The mask is the unchanged gate. The distance is the gate's own
    statistic in degrees, used to rank competing claims for a pixel.
    Deliberately NOT divided by the bin's tolerance: the tolerance is
    the 95th percentile spread of that bin's samples, i.e. a measure
    of how heterogeneous its calibration light was, not of color
    identity. Ranking by the ratio would hand a pixel to whichever bin
    was sampled most sloppily - measured: a lid at 59 deg, 1 deg from
    its own bin (tolerance 7) and 3 deg from a neighbour (tolerance
    30), went entirely to the neighbour. Degrees are the same physical
    unit for every bin, so the closer color wins. Unmatched hue is
    infinitely far.
    """
    dist = circular_dist_deg(hue, model.hue_center_deg)
    # Undefined hue (grey pixel) can never match a color model.
    dist = np.where(np.isnan(dist), np.inf, dist)
    match = (
        (dist <= model.hue_tol_deg)
        & (sat >= model.sat_min)
        & (val >= model.val_min)
    )
    return match, dist


def bin_mask(
    hue: np.ndarray, sat: np.ndarray, val: np.ndarray, model: BinModel
) -> np.ndarray:
    """Boolean mask of pixels matching one bin's learned color model."""
    return bin_match(hue, sat, val, model)[0]


def exclusive_bin_masks(
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    models: list[BinModel],
    *,
    enabled: bool,
) -> list[np.ndarray]:
    """Per-bin masks in which every pixel belongs to at most one bin.

    A pixel shows ONE object, so when two learned color bands overlap
    (the learner warns about exactly that) both bins claiming the same
    pixel is physically impossible. Each contested pixel goes to the
    bin whose learned color it is CLOSEST to, in degrees (see
    bin_match). An exact tie carries no information about which bin it
    belongs to, so the pixel goes to nobody; that also makes the result
    independent of the order of ``models``.

    The masks are always a subset of the plain per-bin masks: a pixel
    accepted by exactly one bin is never touched, so an installation
    whose bands do not overlap is bit-identical to the pre-exclusion
    behavior. ``enabled=False`` returns exactly those plain masks
    (profiles learned before exclusion measured their thresholds
    without it).
    """
    matches, dists = [], []
    for model in models:
        match, dist = bin_match(hue, sat, val, model)
        matches.append(match)
        dists.append(dist)
    if not enabled or len(models) < 2:
        return matches
    best = np.full(hue.shape, np.inf)
    for match, dist in zip(matches, dists):
        scored = np.where(match, dist, np.inf)
        np.minimum(best, scored, out=best)
    # A pixel is contested-and-undecided when two accepted claims share
    # the minimum exactly; count winners to detect that structurally.
    winners = np.zeros(hue.shape, dtype=np.int_)
    for match, dist in zip(matches, dists):
        winners += (match & (dist == best)).astype(np.int_)
    unique = winners == 1
    return [
        match & (dist == best) & unique
        for match, dist in zip(matches, dists)
    ]


def component_holes(
    component: np.ndarray, universe: np.ndarray, depth: np.ndarray
) -> np.ndarray | None:
    """The region pixels enclosed by ``component`` (its holes).

    Free pixels that cannot be reached from the outermost region layer
    (interior depth 1, the same structural criterion the edge band
    uses) without crossing the component are inside it. The flood is
    8-connected like the component machinery in ccl.py, which is what
    makes containment all-or-nothing: another bin's candidate lies
    wholly inside the holes or wholly outside, never partly, so a veto
    can only ever remove a whole candidate and never reshape a
    surviving one. Returns None when the outside is unreachable (a
    color ringing the entire region), where "inside" is meaningless.
    """
    free = universe & ~component
    outside = seeded_component(free, free & (depth == 1))
    if outside is None:
        return None
    return free & ~outside


def veto_qualified(model: BinModel, provisional_area_frac: float) -> bool:
    """May this bin's detected area veto another bin inside it?

    Only a bin that is itself credible evidence may erase another
    bin's: its calibration must separate (a bin whose own learned
    threshold misclassifies calibration images - the field case of a
    brown model that also matches brick pavement - could otherwise
    delete a real lid), it must not rest on a provisional threshold,
    and its current blob must be at least the weakest lid the user
    ever confirmed for it. All three come from learned statistics; the
    area bar is an observed calibration extremum, not a chosen value.
    """
    stats = model.learning_stats
    if not stats.get("veto_qualify_separable", False):
        return False
    if stats.get("veto_qualify_provisional", True):
        return False
    bar = stats.get("veto_qualify_min_area_frac")
    if bar is None:
        return False
    return provisional_area_frac >= float(bar)


@dataclass
class Candidate:
    """One bin's selected blob after exclusion has been resolved."""

    area: int
    box: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    provisional_area: int
    excluded_by: str | None = None
    conflict_with: str | None = None


def resolve_candidates(
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    models: list[BinModel],
    *,
    universe: np.ndarray,
    region: np.ndarray | None,
    depth: np.ndarray,
    denom: int,
    exclusion: bool,
    veto: bool = True,
) -> list[Candidate]:
    """Select every bin's blob under the exclusion rules.

    THE identity contract: detection and area learning both call this
    and nothing else, so thresholds are always learned on exactly the
    pixels detection counts. Two physical statements are applied:
    pixels belong to at most one bin (exclusive_bin_masks), and a bin
    detected inside another bin's detected area cannot exist
    (component_holes veto).

    Resolution runs to a FIXED POINT rather than in a single pass:
    vetoing a bin moves its blob elsewhere, and the new blob may
    enclose a third bin that the first pass never looked at. Enclosure
    is antisymmetric, so the enclosure chain has no cycles and is at
    most one deep per bin - the loop therefore terminates after at
    most len(models) rounds, and every round strictly shrinks at least
    one mask.

    ``veto=False`` applies the pixel layer only. Learning needs that
    mode for the pass that establishes each bin's qualification bar,
    which the veto layer then consumes - the one genuine cycle between
    the two layers, cut by measuring before vetoing rather than by
    iterating.
    """
    masks = exclusive_bin_masks(hue, sat, val, models, enabled=exclusion)
    apply_veto = exclusion and veto
    prepared: list[np.ndarray] = []
    for model, mask in zip(models, masks):
        if region is not None:
            mask = mask & region
        band_frac = edge_band_min_frac(model)
        if band_frac is not None:
            band_px = band_frac * mask.shape[1]
            if band_px > 1.0:
                mask = edge_band_filter(mask, depth, band_px)
        prepared.append(mask)
    provisional = [
        select_component(mask, model, denom)
        for mask, model in zip(prepared, models)
    ]
    if not apply_veto:
        results = []
        for selected in provisional:
            if selected is None:
                results.append(Candidate(0, None, None, 0))
            else:
                area, box, centroid, _seed = selected
                results.append(Candidate(area, box, centroid, area))
        return results

    n = len(models)
    active = list(prepared)
    selected = list(provisional)
    excluded_by: list[str | None] = [None] * n
    containers: list[list[int]] = [[] for _ in range(n)]
    for _round in range(n):
        # Enclosure is decided independently of who may veto, so that
        # an enclosure by a NOT-credible container is still reported as
        # a conflict instead of silently disappearing. bbox nesting is
        # an exact necessary condition, used only to skip the flood
        # fill; every verdict below is pixel-exact.
        holes: list[np.ndarray | None] = [None] * n
        for j, sel_j in enumerate(selected):
            if sel_j is None:
                continue
            if not any(
                i != j
                and selected[i] is not None
                and _box_inside(selected[i][1], sel_j[1])
                for i in range(n)
            ):
                continue
            component = seeded_component(
                active[j], _seed_mask(active[j], sel_j[3])
            )
            if component is not None:
                holes[j] = component_holes(component, universe, depth)
        containers = [[] for _ in range(n)]
        if any(hole is not None for hole in holes):
            for i, sel_i in enumerate(selected):
                if sel_i is None:
                    continue
                blob = None
                for j, hole in enumerate(holes):
                    if i == j or hole is None:
                        continue
                    if not _box_inside(sel_i[1], selected[j][1]):
                        continue
                    if blob is None:
                        blob = seeded_component(
                            active[i], _seed_mask(active[i], sel_i[3])
                        )
                        if blob is None:
                            break
                    if not bool((blob & ~hole).any()):
                        containers[i].append(j)
        # Only a bin that is credible evidence itself may erase
        # another's, and a container that is itself enclosed by a
        # qualified container may not (that is what resolves a nest of
        # three without any ordering).
        qualified = [
            selected[j] is not None
            and holes[j] is not None
            and veto_qualified(models[j], selected[j][0] / denom)
            for j in range(n)
        ]
        vetoers = [
            qualified[j]
            and not any(qualified[k] for k in containers[j])
            for j in range(n)
        ]
        changed = False
        for i in range(n):
            legal = [j for j in containers[i] if vetoers[j]]
            if not legal:
                continue
            mask = active[i]
            for j in legal:
                mask = mask & ~holes[j]
            active[i] = mask
            # Name the largest legal container, tie-broken by bin id:
            # a deterministic choice that does not depend on the order
            # of ``models``.
            excluded_by[i] = min(
                (models[j] for j in legal),
                key=lambda m: (-selected[models.index(m)][0], m.id),
            ).id
            changed = True
        if not changed:
            break
        selected = [
            select_component(mask, model, denom)
            for mask, model in zip(active, models)
        ]

    results = []
    for i, model in enumerate(models):
        sel = selected[i]
        prov = provisional[i]
        prov_area = 0 if prov is None else prov[0]
        if sel is None:
            results.append(
                Candidate(0, None, None, prov_area, excluded_by=excluded_by[i])
            )
            continue
        area, box, centroid, _seed = sel
        # Enclosed, but by no container credible enough to veto: report
        # the conflict instead of acting on it.
        conflict = (
            None
            if excluded_by[i] is not None or not containers[i]
            else models[containers[i][0]].id
        )
        results.append(
            Candidate(
                area,
                box,
                centroid,
                prov_area,
                excluded_by=excluded_by[i],
                conflict_with=conflict,
            )
        )
    return results


def _seed_mask(mask: np.ndarray, pixel: tuple[int, int]) -> np.ndarray:
    """One-pixel seed for re-deriving a component's exact pixel set."""
    seed = np.zeros_like(mask)
    seed[pixel[0], pixel[1]] = True
    return seed


def _box_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


@dataclass
class BinResult:
    id: str
    name: str
    present: bool
    area_frac: float
    min_area_frac: float
    margin: float  # area_frac / min_area_frac - confidence on a ratio scale
    uncertain: bool = False  # inside the learned ambiguity interval
    # Location of the largest matching blob, in FULL-IMAGE relative
    # coordinates (same frame the sample rects live in): bounding box
    # [x, y, w, h] and centroid [cx, cy]. None when nothing matched.
    bbox: tuple[float, float, float, float] | None = None
    centroid: tuple[float, float] | None = None
    # Mutual exclusion: what the blob measured before another bin's
    # detected area vetoed it, which bin that was, and whether an
    # enclosure was seen but left unresolved.
    provisional_area_frac: float = 0.0
    excluded_by: str | None = None
    exclusion_conflict: bool = False

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
            "bbox": None if self.bbox is None else list(self.bbox),
            "centroid": None if self.centroid is None else list(self.centroid),
            "provisional_area_frac": self.provisional_area_frac,
            "excluded_by": self.excluded_by,
            "exclusion_conflict": self.exclusion_conflict,
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


def _mask_point_to_image(
    point: tuple[float, float], width: int, height: int, roi: Roi
) -> tuple[float, float]:
    """Working-mask pixel coordinates -> full-image relative (0..1).

    Pixel centers sit at index + 0.5; the ROI grid maps affinely back
    into the frame the sample rects and the calibration card live in.
    """
    fx = (point[0] + 0.5) / width
    fy = (point[1] + 0.5) / height
    return (roi.x + fx * roi.w, roi.y + fy * roi.h)


def _mask_box_to_image(
    box: tuple[int, int, int, int], width: int, height: int, roi: Roi
) -> tuple[float, float, float, float]:
    """Half-open working-mask pixel box -> full-image relative [x,y,w,h]."""
    x0, y0, x1, y1 = box
    left = roi.x + (x0 / width) * roi.w
    top = roi.y + (y0 / height) * roi.h
    return (
        left,
        top,
        (x1 - x0) / width * roi.w,
        (y1 - y0) / height * roi.h,
    )


def _cached_region_mask(profile: Profile, width: int, height: int, crop: Roi):
    """Per-profile mask cache; a relearn creates a new Profile object,
    so the cache's lifetime is exactly the profile's - no eviction
    parameter needed, it only grows per distinct frame geometry."""
    if profile.roi_polygons is None:
        return None
    cache = getattr(profile, "_region_mask_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(profile, "_region_mask_cache", cache)
    key = (width, height, crop)
    if key not in cache:
        cache[key] = region_mask(profile.roi_polygons, width, height, crop)
    return cache[key]


def _cached_interior_depth(
    profile: Profile, width: int, height: int, crop: Roi
) -> np.ndarray:
    """Interior-depth map of the region universe, cached per profile
    (same lifetime argument as _cached_region_mask). The universe is
    the polygon mask, or the full crop for a rectangle region - there
    the crop edge is the drawn boundary."""
    cache = getattr(profile, "_interior_depth_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(profile, "_interior_depth_cache", cache)
    key = (width, height, crop)
    if key not in cache:
        poly = _cached_region_mask(profile, width, height, crop)
        universe = (
            np.ones((height, width), dtype=bool) if poly is None else poly
        )
        cache[key] = interior_depth(universe)
    return cache[key]


def detect(img: Image.Image, profile: Profile) -> DetectionResult:
    """Run detection on an already-loaded PIL image."""
    validate_profile(profile)
    arr = extract_working_roi(img, profile.roi, profile.working_width, profile.resample)
    hue, sat, val = rgb_to_hsv(arr)
    height, width = arr.shape[0], arr.shape[1]
    total = height * width
    # The crop is cut on INTEGER source pixels (roi_to_pixels rounds the
    # nominal ROI edges), so blob coordinates must map through the crop
    # that was actually taken, not the nominal ROI - otherwise every
    # box inherits up to half a source pixel of offset per edge.
    src_x0, src_y0, src_x1, src_y1 = roi_to_pixels(
        profile.roi, img.width, img.height
    )
    actual_roi = Roi(
        x=src_x0 / img.width,
        y=src_y0 / img.height,
        w=(src_x1 - src_x0) / img.width,
        h=(src_y1 - src_y0) / img.height,
    )
    # Polygon region: mask applied to the COLOR masks (never to the RGB
    # pixels - the frame gates below must keep seeing the full crop),
    # denominator = pixels of the monitored region. None = whole crop,
    # the exact pre-polygon fast path.
    poly = _cached_region_mask(profile, width, height, actual_roi)
    denom = int(poly.sum()) if poly is not None else total
    universe = (
        np.ones((height, width), dtype=bool) if poly is None else poly
    )
    depth = _cached_interior_depth(profile, width, height, actual_roi)
    candidates = resolve_candidates(
        hue,
        sat,
        val,
        profile.bins,
        universe=universe,
        region=poly,
        depth=depth,
        denom=denom,
        exclusion=profile.mutual_exclusion,
    )
    results: list[BinResult] = []
    for model, cand in zip(profile.bins, candidates):
        bbox = (
            None
            if cand.box is None
            else _mask_box_to_image(cand.box, width, height, actual_roi)
        )
        centroid = (
            None
            if cand.centroid is None
            else _mask_point_to_image(
                cand.centroid, width, height, actual_roi
            )
        )
        frac = cand.area / denom
        conflict = cand.conflict_with is not None
        results.append(
            BinResult(
                id=model.id,
                name=model.name,
                present=frac >= model.min_area_frac,
                area_frac=frac,
                min_area_frac=model.min_area_frac,
                margin=frac / model.min_area_frac,
                # An unresolved enclosure (the container is not credible
                # enough to veto) is reported, never acted on: hold the
                # last safe state instead of flipping on contested
                # evidence.
                uncertain=is_uncertain(frac, model.learning_stats)
                or (conflict and frac >= model.min_area_frac),
                bbox=bbox,
                centroid=centroid,
                provisional_area_frac=cand.provisional_area / denom,
                excluded_by=cand.excluded_by,
                exclusion_conflict=conflict,
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
