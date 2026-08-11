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
