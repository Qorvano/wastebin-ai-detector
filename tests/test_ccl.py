"""Connected components: shapes that break naive implementations."""

from __future__ import annotations

import numpy as np
import pytest

from wastebin_ai_detector.core import largest_component_area


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[c == "#" for c in row] for row in rows])


def test_empty_and_full():
    assert largest_component_area(np.zeros((5, 7), dtype=bool)) == 0
    assert largest_component_area(np.ones((5, 7), dtype=bool)) == 35


def test_single_pixel():
    assert largest_component_area(_mask(["....", ".#..", "...."])) == 1


def test_l_shape():
    assert largest_component_area(_mask(["#..", "#..", "###"])) == 5


def test_diagonal_chain_is_8_connected():
    assert largest_component_area(np.eye(3, dtype=bool)) == 3
    assert largest_component_area(np.fliplr(np.eye(3)).astype(bool)) == 3


def test_two_blobs_returns_larger():
    mask = _mask(
        [
            "##....",
            "##....",
            "...###",
            "...###",
        ]
    )
    assert largest_component_area(mask) == 6


def test_u_shape_late_merge():
    # Arms only join in the last row - forces transitive union-find.
    mask = _mask(
        [
            "#...#",
            "#...#",
            "#...#",
            "#...#",
            "#####",
        ]
    )
    assert largest_component_area(mask) == 13


def test_diagonal_touch_merges():
    assert largest_component_area(_mask(["#.", ".#"])) == 2


def test_gap_of_two_does_not_merge():
    assert largest_component_area(_mask(["#..#"])) == 1


def test_non_2d_rejected():
    with pytest.raises(ValueError):
        largest_component_area(np.zeros(5, dtype=bool))
