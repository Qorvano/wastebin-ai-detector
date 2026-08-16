"""Polygon regions: rasterization and containment.

A region is a bounding box (``Roi``) plus optionally a list of polygon
rings in full-image relative coordinates. ``rings is None`` means "the
whole bounding box" - exactly the pre-polygon semantics, taken as a
zero-cost fast path everywhere.

Rings are interpreted with the even-odd rule on pixel centers
(index + 0.5, the shared convention of the pixel mapping in detect):
- even-odd is orientation-independent (users may draw clockwise or
  counter-clockwise) and well-defined for self-intersecting input, so
  no simplicity test - and therefore no epsilon constants - is needed;
- the calibration card previews the region with SVG fill-rule
  "evenodd", so the user sees exactly the area the core computes;
- multiple rings compose by the same rule (disjoint rings = union),
  which covers physically separate bin spots.

Masks are always rasterized directly on the target grid, never
downsampled from a finer one: resampling a boolean mask would create
soft edges and require a binarization threshold (a magic number).
"""

from __future__ import annotations

import numpy as np

from .errors import ProfileError, RoiError
from .profile import REL_EPS, Roi

# A ring needs at least a triangle to enclose any area.
MIN_RING_VERTICES = 3

Rings = list[list[tuple[float, float]]]


def validate_rings(rings: Rings) -> None:
    """Structural validation (unit coordinates, ring sizes).

    Deliberately no simplicity/self-intersection test: analytic
    detection of near-touching or collinear edges cannot be decided
    without epsilon thresholds, and the even-odd interpretation is
    well-defined for any input. Degeneration to zero pixels is caught
    at rasterization time as RoiError, like every other degenerate
    geometry in the pipeline.
    """
    if not rings:
        raise ProfileError("region polygon list is empty (use None instead)")
    for i, ring in enumerate(rings):
        if len(ring) < MIN_RING_VERTICES:
            raise ProfileError(
                f"ring {i}: {len(ring)} vertices, need at least "
                f"{MIN_RING_VERTICES}"
            )
        for x, y in ring:
            if not (
                -REL_EPS <= x <= 1.0 + REL_EPS
                and -REL_EPS <= y <= 1.0 + REL_EPS
            ):
                raise ProfileError(
                    f"ring {i}: vertex ({x}, {y}) outside the unit frame"
                )


def clamp_rings(rings: Rings) -> Rings:
    """Clamp FP drift into [0, 1] (same policy as clamp_unit_rect)."""
    return [
        [
            (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
            for x, y in ring
        ]
        for ring in rings
    ]


def rings_bbox(rings: Rings) -> Roi:
    """Axis-aligned bounding box of all rings (image-relative)."""
    xs = [x for ring in rings for x, _y in ring]
    ys = [y for ring in rings for _x, y in ring]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
        raise ProfileError("region polygon collapses to zero width/height")
    return Roi(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def rings_equal(a: Rings | None, b: Rings | None) -> bool:
    """Exact structural equality within REL_EPS per coordinate."""
    if (a is None) != (b is None):
        return False
    if a is None and b is None:
        return True
    if len(a) != len(b):
        return False
    for ring_a, ring_b in zip(a, b):
        if len(ring_a) != len(ring_b):
            return False
        for (ax, ay), (bx, by) in zip(ring_a, ring_b):
            if abs(ax - bx) > REL_EPS or abs(ay - by) > REL_EPS:
                return False
    return True


def rect_as_rings(roi: Roi) -> Rings:
    """A rectangle as its 4-vertex ring (the degenerate polygon case)."""
    return [
        [
            (roi.x, roi.y),
            (roi.x + roi.w, roi.y),
            (roi.x + roi.w, roi.y + roi.h),
            (roi.x, roi.y + roi.h),
        ]
    ]


def polygon_mask(
    rings: Rings, width: int, height: int, crop: Roi
) -> np.ndarray:
    """Even-odd rasterization of ``rings`` onto a ``width``x``height``
    grid that covers ``crop`` (image-relative box of the actual pixel
    crop, so the mask cannot inherit the sub-pixel offset between the
    nominal region box and the integer-rounded crop).

    Pixel centers sit at (index + 0.5) like everywhere else in the
    pipeline. A ring edge toggles a row's parity left of its
    intersection; summing toggles over all edges of all rings yields
    the even-odd interior. Raises RoiError when no pixel center falls
    inside (degenerate on this grid), mirroring rect_to_pixels.
    """
    if width <= 0 or height <= 0:
        raise RoiError(f"empty target grid {width}x{height}")
    mask = np.zeros((height, width), dtype=bool)
    # Pixel-center coordinates in image-relative space.
    col_x = crop.x + (np.arange(width) + 0.5) / width * crop.w
    row_y = crop.y + (np.arange(height) + 0.5) / height * crop.h
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % n]
            if y0 == y1:
                continue  # horizontal edges never cross a scanline
            y_lo, y_hi = (y0, y1) if y0 < y1 else (y1, y0)
            # Half-open rule [y_lo, y_hi): shared vertices count once.
            rows = np.nonzero((row_y >= y_lo) & (row_y < y_hi))[0]
            if rows.size == 0:
                continue
            # Edge x at each crossed scanline (linear interpolation).
            t = (row_y[rows] - y0) / (y1 - y0)
            edge_x = x0 + t * (x1 - x0)
            mask[rows] ^= col_x[None, :] < edge_x[:, None]
    if not bool(mask.any()):
        raise RoiError(
            f"region polygon covers no pixel center on the {width}x{height} "
            "grid - it is degenerate at this resolution"
        )
    return mask


def region_mask(
    rings: Rings | None, width: int, height: int, crop: Roi
) -> np.ndarray | None:
    """Mask for a region; None for the rectangle fast path (callers
    treat None as all-True without allocating anything)."""
    if rings is None:
        return None
    return polygon_mask(rings, width, height, crop)


def interior_depth(universe: np.ndarray) -> np.ndarray:
    """Chessboard distance of every region pixel to the nearest
    non-region cell, int32, 0 outside the region.

    - Metric/connectivity: chessboard (8-neighborhood peeling), the
      documented model choice of the component machinery in ccl.py -
      any other metric would be inconsistent with how components are
      defined.
    - The GRID BORDER counts as exterior: the crop edge is part of the
      drawn boundary (for a rectangle region the crop edge IS the
      boundary), so scene content hanging over any edge is banded the
      same way as content crossing a polygon line.
    - Every region pixel has depth >= 1, so a band of 1 or less removes
      nothing (the structural no-op identity the activation predicate
      relies on).

    Pure peeling: per iteration an 8-neighbor erosion (all neighbors
    inside, off-grid counts as outside); a pixel's depth is the number
    of erosions it survives plus one.
    """
    current = np.asarray(universe, dtype=bool).copy()
    depth = np.zeros(current.shape, dtype=np.int32)
    level = 0
    while bool(current.any()):
        level += 1
        depth[current] = level
        eroded = np.zeros_like(current)
        if current.shape[0] > 2 and current.shape[1] > 2:
            eroded[1:-1, 1:-1] = (
                current[1:-1, 1:-1]
                & current[:-2, 1:-1]
                & current[2:, 1:-1]
                & current[1:-1, :-2]
                & current[1:-1, 2:]
                & current[:-2, :-2]
                & current[:-2, 2:]
                & current[2:, :-2]
                & current[2:, 2:]
            )
        current = eroded
    return depth


def region_grid(bbox: Roi, working_width: int) -> tuple[int, int]:
    """A canonical store-level grid for containment tests: width fixed,
    height from the RELATIVE box aspect. Per-image pipeline grids also
    carry the source frame's pixel aspect, which is unknown at store
    level, so this grid can differ from them by that factor; both masks
    of a containment test share THIS grid, keeping the verdict
    internally consistent."""
    height = max(round(working_width * (bbox.h / bbox.w)), 1)
    return working_width, int(height)


def region_contains(
    outer_bbox: Roi,
    outer_rings: Rings | None,
    inner_bbox: Roi,
    inner_rings: Rings | None,
    working_width: int | None,
) -> bool:
    """Is the inner region fully inside the outer one?

    Rect-rect keeps the analytic path (bit-identical to the pre-polygon
    verdicts), as does a rect OUTER around any inner region. When the
    outer has rings, both regions are rasterized on the outer region's
    canonical store-level grid (see region_grid) and tested as a
    subset. Without a working width no such grid exists; polygon-outer
    containment is then conservatively False (the evidence is set
    aside rather than guessed about).
    """
    analytic = (
        inner_bbox.x >= outer_bbox.x - REL_EPS
        and inner_bbox.y >= outer_bbox.y - REL_EPS
        and inner_bbox.x + inner_bbox.w
        <= outer_bbox.x + outer_bbox.w + REL_EPS
        and inner_bbox.y + inner_bbox.h
        <= outer_bbox.y + outer_bbox.h + REL_EPS
    )
    if outer_rings is None and inner_rings is None:
        return analytic
    if not analytic:
        # bbox containment is a necessary condition for any region.
        return False
    if outer_rings is None:
        # The outer region IS its bbox: containment of the inner bbox
        # (affirmed above) already implies containment of anything
        # inside it - grid-free, so this must precede the width guard.
        return True
    if working_width is None:
        return False
    width, height = region_grid(outer_bbox, working_width)
    try:
        outer = region_mask(outer_rings, width, height, outer_bbox)
        if outer is None:
            # Outer is the full bbox: inner (bbox-contained above) is
            # always a subset.
            return True
        inner = region_mask(
            inner_rings
            if inner_rings is not None
            else rect_as_rings(inner_bbox),
            width,
            height,
            outer_bbox,
        )
        assert inner is not None
        return not bool((inner & ~outer).any())
    except RoiError:
        # A region degenerate on the pipeline grid contains nothing and
        # is contained in nothing meaningful.
        return False
