"""Region-edge band: interior depth, the learned band, and the
touch+reach candidate filter (field failure: a hedge sliver hugging
the drawn contour confidently flipped a sensor)."""

from __future__ import annotations

import numpy as np
import pytest

from scenes import YELLOW, make_scene
from wastebin_ai_detector.core import detect_file, learn_profile
from wastebin_ai_detector.core.detect import (
    edge_band_filter,
    edge_band_min_frac,
)
from wastebin_ai_detector.core.learn import learn_edge_band
from wastebin_ai_detector.core.profile import BinModel
from wastebin_ai_detector.core.region import interior_depth
from wastebin_ai_detector.core.store import (
    BinDecl,
    CalibrationStore,
    Rect,
    Roi,
)


class TestInteriorDepth:
    def test_full_rectangle_matches_closed_form(self):
        h, w = 12, 30
        depth = interior_depth(np.ones((h, w), dtype=bool))
        for y in range(h):
            for x in range(w):
                assert depth[y, x] == min(x, y, w - 1 - x, h - 1 - y) + 1

    def test_outside_is_zero_and_edge_is_one(self):
        universe = np.zeros((10, 10), dtype=bool)
        universe[3:7, 2:9] = True
        depth = interior_depth(universe)
        assert depth[universe].min() == 1
        assert (depth[~universe] == 0).all()
        assert depth[3, 2] == 1 and depth[6, 8] == 1
        assert depth[4, 5] == 2 and depth[5, 5] == 2

    def test_one_pixel_strip_is_all_one(self):
        universe = np.zeros((8, 8), dtype=bool)
        universe[4, 1:7] = True
        assert (interior_depth(universe)[universe] == 1).all()

    def test_hole_edge_counts_as_boundary(self):
        universe = np.ones((15, 15), dtype=bool)
        universe[7, 7] = False  # a one-pixel hole
        depth = interior_depth(universe)
        assert depth[7, 6] == 1 and depth[6, 6] == 1
        assert depth[1, 1] == 2

    def test_grid_border_is_exterior(self):
        universe = np.ones((6, 6), dtype=bool)
        depth = interior_depth(universe)
        assert (depth[0, :] == 1).all() and (depth[:, 0] == 1).all()

    def test_empty_and_degenerate_universes_terminate(self):
        assert interior_depth(np.zeros((5, 5), dtype=bool)).max() == 0
        assert interior_depth(np.ones((1, 1), dtype=bool))[0, 0] == 1
        assert (interior_depth(np.ones((1, 9), dtype=bool)) == 1).all()


class TestLearnEdgeBand:
    def test_hand_math(self):
        stats = learn_edge_band([0.05, 0.07, 0.06])
        assert stats["region_edge_depth_n"] == 3
        assert stats["region_edge_depth_obs_min_frac"] == pytest.approx(0.05)
        # successive diffs .02, .01 -> median slack .015 -> band .035
        assert stats["region_edge_depth_min_frac"] == pytest.approx(0.035)

    def test_single_observation_has_zero_slack(self):
        stats = learn_edge_band([0.04])
        assert stats["region_edge_depth_min_frac"] == pytest.approx(0.04)

    def test_empty_series(self):
        assert learn_edge_band([]) == {"region_edge_depth_n": 0}

    def test_activation_predicate(self):
        def model_with(stats):
            return BinModel(
                id="b",
                name="B",
                hue_center_deg=60.0,
                hue_tol_deg=10.0,
                sat_min=0.2,
                val_min=0.2,
                min_area_frac=0.01,
                learning_stats=stats,
            )

        # Singular observation: inactive (no cross-pose variation).
        assert edge_band_min_frac(
            model_with(
                {"region_edge_depth_n": 1, "region_edge_depth_min_frac": 0.09}
            )
        ) is None
        # A zero band filters nothing.
        assert edge_band_min_frac(
            model_with(
                {"region_edge_depth_n": 3, "region_edge_depth_min_frac": 0.0}
            )
        ) is None
        assert edge_band_min_frac(
            model_with(
                {"region_edge_depth_n": 2, "region_edge_depth_min_frac": 0.04}
            )
        ) == 0.04
        # Legacy profile without the fields: inactive.
        assert edge_band_min_frac(model_with({})) is None


class TestEdgeBandFilter:
    def _depth(self, shape):
        return interior_depth(np.ones(shape, dtype=bool))

    def test_shallow_touching_sliver_dropped(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[0:3, 10:16] = True  # hugs the top boundary, depth <= 3
        out = edge_band_filter(mask, self._depth(mask.shape), band=6.0)
        assert not out.any()

    def test_interior_component_never_filtered(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[8:11, 12:18] = True  # small, deep inside
        out = edge_band_filter(mask, self._depth(mask.shape), band=6.0)
        assert (out == mask).all()

    def test_touching_component_that_reaches_band_kept_whole(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[0:9, 10:16] = True  # crosses from the edge to depth 9
        out = edge_band_filter(mask, self._depth(mask.shape), band=6.0)
        assert (out == mask).all()

    def test_mixed_components_filtered_independently(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[0:3, 2:6] = True  # shallow toucher: dropped
        mask[0:9, 10:16] = True  # deep toucher: kept
        mask[8:11, 20:26] = True  # interior: kept
        out = edge_band_filter(mask, self._depth(mask.shape), band=6.0)
        assert not out[0:3, 2:6].any()
        assert out[0:9, 10:16].all()
        assert out[8:11, 20:26].all()

    def test_empty_mask_stays_empty(self):
        mask = np.zeros((10, 10), dtype=bool)
        out = edge_band_filter(mask, self._depth(mask.shape), band=3.0)
        assert not out.any()


# Corridor region around the row of bins, as users draw it. Real
# regions are tight, so lid components CROSS the contour and touch the
# boundary - those are the poses that teach the band how deep a lid
# reaches. The washed pose sits interior and sets a tiny min_pos, the
# field condition under which a boundary sliver becomes a confident
# false positive.
CORRIDOR = [[(0.10, 0.30), (0.90, 0.30), (0.90, 0.70), (0.10, 0.70)]]
CROSS_A = (0.25, 0.24, 0.20, 0.32)
CROSS_B = (0.55, 0.24, 0.20, 0.30)
WASHED = (0.45, 0.45, 0.03, 0.05)
SLIVER = (0.42, 0.301, 0.03, 0.04)  # hugs the contour, lid-like aspect
INTERIOR = (0.30, 0.40, 0.16, 0.20)


def _corridor_store(working_width: int | None = 160) -> CalibrationStore:
    store = CalibrationStore(
        roi=Roi(0.0, 0.0, 1.0, 1.0),
        working_width=working_width,
        resample="bilinear",
        bins=[BinDecl("gelb", "Gelbe Tonne")],
    )
    store.set_region(CORRIDOR)
    return store


def _calibrate(tmp_path, store, lids=(CROSS_A, CROSS_B, WASHED), absent=True):
    for i, lid in enumerate(lids):
        name = f"cal{i}.png"
        make_scene(rects=[(YELLOW, *lid)], seed=i).save(
            tmp_path / name, format="PNG"
        )
        store.add_sample(
            name,
            "gelb",
            Rect(
                lid[0] + lid[2] * 0.25,
                lid[1] + lid[3] * 0.4,
                lid[2] * 0.5,
                lid[3] * 0.25,
            ),
        )
        store.set_labels(name, present=["gelb"])
    if absent:
        make_scene(rects=[], seed=7).save(tmp_path / "empty.png", format="PNG")
        store.set_labels("empty.png", absent=["gelb"])
    return learn_profile(store, tmp_path / "store.json")


def _scene(tmp_path, name, rect, size=(320, 200), seed=9):
    make_scene(size=size, rects=[(YELLOW, *rect)], seed=seed).save(
        tmp_path / name, format="PNG"
    )
    return tmp_path / name


class _band_disabled:
    """Deactivate ONLY the band on a learned profile (the activation
    predicate needs two observations), leaving color, shape and
    thresholds untouched - the control condition for attribution."""

    def __init__(self, profile):
        self.stats = profile.bins[0].learning_stats

    def __enter__(self):
        self.keep = self.stats["region_edge_depth_n"]
        self.stats["region_edge_depth_n"] = 1
        return self

    def __exit__(self, *exc):
        self.stats["region_edge_depth_n"] = self.keep
        return False


class TestEdgeBandEndToEnd:
    def test_band_is_learned_from_touching_poses_only(self, tmp_path):
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        stats = profile.bins[0].learning_stats
        # Two crossing poses vote; the interior washed pose does not.
        assert stats["region_edge_depth_n"] == 2
        assert stats["shape_n"] == 3
        assert edge_band_min_frac(profile.bins[0]) is not None

    def test_sliver_dies_BECAUSE_of_the_band(self, tmp_path):
        """Attribution test: with the band the boundary sliver is an
        honest zero, without it the very same frame is a CONFIDENT
        false positive - the field failure this feature exists for."""
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        path = _scene(tmp_path, "sliver.png", SLIVER)

        res = detect_file(path, profile).bins[0]
        assert res.area_frac == 0.0
        assert res.present is False
        assert res.uncertain is False

        with _band_disabled(profile):
            unbanded = detect_file(path, profile).bins[0]
        assert unbanded.present is True
        assert unbanded.area_frac > profile.bins[0].min_area_frac

    def test_calibrated_poses_stay_detected(self, tmp_path):
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        for name in ("cal0.png", "cal1.png", "cal2.png"):
            assert detect_file(tmp_path / name, profile).bins[0].present, name

    def test_interior_pose_never_vetoed(self, tmp_path):
        """A lid parked deep inside is never touched by the filter."""
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        path = _scene(tmp_path, "interior.png", INTERIOR, seed=12)
        assert detect_file(path, profile).bins[0].present is True

    def test_interior_calibration_does_not_veto_edge_poses(self, tmp_path):
        """Regression (review finding): a band learned from INTERIOR
        lids would encode their parking position and silently kill
        legitimate lids parked against the contour. Interior poses
        therefore do not vote, and the band stays inactive here."""
        profile, _warnings = _calibrate(
            tmp_path,
            _corridor_store(),
            lids=((0.30, 0.42, 0.16, 0.16), (0.55, 0.42, 0.16, 0.16)),
        )
        assert profile.bins[0].learning_stats["region_edge_depth_n"] == 0
        assert edge_band_min_frac(profile.bins[0]) is None
        path = _scene(tmp_path, "edge_pose.png", (0.40, 0.28, 0.16, 0.20), seed=15)
        assert detect_file(path, profile).bins[0].present is True

    def test_band_survives_a_resolution_change(self, tmp_path):
        """The band is a fraction of the grid width, so a store without
        a fixed working_width (CLI default: native crops) keeps its
        verdicts when the camera resolution changes. Rectangle region:
        polygon regions require a working width for containment, and
        there the crop edge IS the drawn boundary."""
        store = CalibrationStore(
            roi=Roi(0.10, 0.30, 0.80, 0.40),
            working_width=None,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne")],
        )
        profile, _warnings = _calibrate(tmp_path, store)
        assert edge_band_min_frac(profile.bins[0]) is not None
        for size in ((320, 200), (640, 400), (160, 100)):
            lid = _scene(tmp_path, f"lid_{size[0]}.png", CROSS_A, size, seed=3)
            assert detect_file(lid, profile).bins[0].present, lid.name
            sliver = _scene(
                tmp_path, f"sliver_{size[0]}.png", SLIVER, size, seed=9
            )
            assert detect_file(sliver, profile).bins[0].present is False, (
                sliver.name
            )

    def test_learn_detect_identity_on_calibration_images(self, tmp_path):
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        stats = profile.bins[0].learning_stats
        fracs = [
            detect_file(tmp_path / name, profile).bins[0].area_frac
            for name in ("cal0.png", "cal1.png", "cal2.png")
        ]
        assert min(fracs) == pytest.approx(stats["min_pos_area_frac"], abs=0.0)

    def test_clutter_diagnosis_warns_when_band_cannot_separate(self, tmp_path):
        """An absent frame whose boundary clutter reaches the band gets
        a loud warning instead of silent overlap."""
        store = _corridor_store()
        for i, lid in enumerate((CROSS_A, CROSS_B)):
            name = f"cal{i}.png"
            make_scene(rects=[(YELLOW, *lid)], seed=i).save(
                tmp_path / name, format="PNG"
            )
            store.add_sample(
                name,
                "gelb",
                Rect(
                    lid[0] + lid[2] * 0.25,
                    lid[1] + lid[3] * 0.4,
                    lid[2] * 0.5,
                    lid[3] * 0.25,
                ),
            )
            store.set_labels(name, present=["gelb"])
        # Absent frame with a DEEP tongue of lid color from the edge.
        make_scene(rects=[(YELLOW, 0.20, 0.28, 0.05, 0.30)], seed=21).save(
            tmp_path / "clutter.png", format="PNG"
        )
        store.set_labels("clutter.png", absent=["gelb"])
        profile, warnings = learn_profile(store, tmp_path / "store.json")
        assert profile.bins[0].learning_stats["region_edge_separable"] is False
        assert any("boundary clutter" in w for w in warnings)

    def test_legacy_profile_without_stats_is_unfiltered(self, tmp_path):
        """Profiles learned before the band keep their exact behavior:
        their thresholds were measured unbanded."""
        profile, _warnings = _calibrate(tmp_path, _corridor_store())
        stats = profile.bins[0].learning_stats
        for key in [k for k in stats if k.startswith("region_edge_")]:
            stats.pop(key)
        assert edge_band_min_frac(profile.bins[0]) is None
        path = _scene(tmp_path, "legacy.png", SLIVER)
        assert detect_file(path, profile).bins[0].area_frac > 0.0
