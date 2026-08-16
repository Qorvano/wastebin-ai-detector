"""Connected-component labelling on boolean masks.

Runs-based two-pass algorithm with union-find over row runs - pure
numpy + Python, no scipy. 8-connectivity is a modelling choice, not a
tuning value: JPEG edge artifacts regularly thin a lid blob down to
diagonal contact, and a lid is still one physical surface then.
"""

from __future__ import annotations

import numpy as np


def largest_component_area(mask: np.ndarray) -> int:
    """Return the pixel count of the largest 8-connected True component.

    ``mask`` must be a 2-D array (interpreted as boolean); returns 0 for
    an all-False mask.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {mask.shape}")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    parent: list[int] = []
    size: list[int] = []

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if size[ra] < size[rb]:  # union by size
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return ra

    # (start, end_exclusive, label) runs of the previous row
    prev_runs: list[tuple[int, int, int]] = []
    for row in mask:
        padded = np.zeros(row.size + 2, dtype=np.int8)
        padded[1:-1] = row
        delta = np.diff(padded)
        starts = np.flatnonzero(delta == 1)
        ends = np.flatnonzero(delta == -1)
        cur_runs: list[tuple[int, int, int]] = []
        p = 0
        for b0, b1 in zip(starts.tolist(), ends.tolist()):
            # Runs [a0, a1) and [b0, b1) on adjacent rows are
            # 8-connected iff a0 <= b1 AND b0 <= a1: with half-open
            # bounds the two ≤ carry exactly the ±1 column slack of the
            # diagonal neighbourhood (a1 == b0 means corner contact).
            while p < len(prev_runs) and prev_runs[p][1] < b0:
                p += 1  # a1 < b0: can never touch this or any later run
            label = -1
            q = p
            while q < len(prev_runs) and prev_runs[q][0] <= b1:
                other = prev_runs[q][2]
                label = other if label < 0 else union(label, other)
                q += 1
            if label < 0:
                label = len(parent)
                parent.append(label)
                size.append(0)
            root = find(label)
            size[root] += b1 - b0
            cur_runs.append((b0, b1, root))
        prev_runs = cur_runs

    best = 0
    for i in range(len(parent)):
        if find(i) == i and size[i] > best:
            best = size[i]
    return best


def component_regions(
    mask: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], tuple[float, float]]]:
    """All 8-connected components as (area, half-open box, centroid),
    sorted by area descending. Same runs/union-find scheme as
    :func:`largest_component_area` (untouched for its callers), with
    per-component extent and first moments carried through the merges.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {mask.shape}")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    parent: list[int] = []
    size: list[int] = []
    min_x: list[int] = []
    max_x: list[int] = []
    min_y: list[int] = []
    max_y: list[int] = []
    sum_x: list[float] = []
    sum_y: list[float] = []

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        min_x[ra] = min(min_x[ra], min_x[rb])
        max_x[ra] = max(max_x[ra], max_x[rb])
        min_y[ra] = min(min_y[ra], min_y[rb])
        max_y[ra] = max(max_y[ra], max_y[rb])
        sum_x[ra] += sum_x[rb]
        sum_y[ra] += sum_y[rb]
        return ra

    prev_runs: list[tuple[int, int, int]] = []
    for y, row in enumerate(mask):
        padded = np.zeros(row.size + 2, dtype=np.int8)
        padded[1:-1] = row
        delta = np.diff(padded)
        starts = np.flatnonzero(delta == 1)
        ends = np.flatnonzero(delta == -1)
        cur_runs: list[tuple[int, int, int]] = []
        p = 0
        for b0, b1 in zip(starts.tolist(), ends.tolist()):
            while p < len(prev_runs) and prev_runs[p][1] < b0:
                p += 1
            label = -1
            q = p
            while q < len(prev_runs) and prev_runs[q][0] <= b1:
                other = prev_runs[q][2]
                label = other if label < 0 else union(label, other)
                q += 1
            if label < 0:
                label = len(parent)
                parent.append(label)
                size.append(0)
                min_x.append(b0)
                max_x.append(b1 - 1)
                min_y.append(y)
                max_y.append(y)
                sum_x.append(0.0)
                sum_y.append(0.0)
            root = find(label)
            n = b1 - b0
            size[root] += n
            min_x[root] = min(min_x[root], b0)
            max_x[root] = max(max_x[root], b1 - 1)
            min_y[root] = min(min_y[root], y)
            max_y[root] = max(max_y[root], y)
            # Column sum of the run: arithmetic series b0..b1-1.
            sum_x[root] += (b0 + b1 - 1) * n / 2.0
            sum_y[root] += float(y) * n
            cur_runs.append((b0, b1, root))
        prev_runs = cur_runs

    regions = [
        (
            size[i],
            (min_x[i], min_y[i], max_x[i] + 1, max_y[i] + 1),
            (sum_x[i] / size[i], sum_y[i] / size[i]),
        )
        for i in range(len(parent))
        if find(i) == i and size[i] > 0
    ]
    regions.sort(key=lambda r: r[0], reverse=True)
    return regions


def largest_component_region(
    mask: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], tuple[float, float]] | None:
    """Largest component's (area, half-open box, centroid), or None."""
    regions = component_regions(mask)
    return regions[0] if regions else None


def seeded_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """The connected component of ``mask`` touching ``seed`` (both 2-D
    bool). Exact 8-connectivity via iterative dilation-by-shifts until
    stable (O(diameter) shift loop, exact and dependency-free).
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
