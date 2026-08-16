"""Mutual exclusion between bins: two bins can never claim the same
pixel, and a bin detected inside another bin's detected area cannot
exist (field case: a white sticker on the brown lid matching yellow)."""

from __future__ import annotations

import numpy as np
import pytest

from scenes import BLUE, YELLOW, make_scene
from wastebin_ai_detector.core import (
    bin_mask,
    bin_match,
    component_holes,
    detect_file,
    exclusive_bin_masks,
    learn_profile,
    resolve_candidates,
    veto_qualified,
)
from wastebin_ai_detector.core.profile import BinModel
from wastebin_ai_detector.core.region import interior_depth
from wastebin_ai_detector.core.store import (
    BinDecl,
    CalibrationStore,
    Rect,
    Roi,
)


def _model(bin_id, center, tol, **stats):
    return BinModel(
        id=bin_id,
        name=bin_id.title(),
        hue_center_deg=center,
        hue_tol_deg=tol,
        sat_min=0.1,
        val_min=0.1,
        min_area_frac=0.001,
        learning_stats=stats,
    )


def _hsv(hues, sat=0.8, val=0.8):
    hue = np.asarray(hues, dtype=np.float64)
    return hue, np.full(hue.shape, sat), np.full(hue.shape, val)


class TestPixelExclusivity:
    def test_disjoint_bands_are_bit_identical(self):
        hue, sat, val = _hsv([[10.0, 60.0, 200.0, 300.0]])
        models = [_model("a", 60.0, 8.0), _model("b", 200.0, 8.0)]
        exclusive = exclusive_bin_masks(hue, sat, val, models, enabled=True)
        for model, mask in zip(models, exclusive):
            assert np.array_equal(mask, bin_mask(hue, sat, val, model))

    def test_contested_pixel_goes_to_the_closer_model(self):
        # 58 deg: 2 deg from the narrow bin (tol 7 -> 0.29 of its band),
        # 47 from the broad one (tol 48 -> 0.98). The narrow bin wins,
        # although BOTH gates accept the pixel.
        hue, sat, val = _hsv([[58.0]])
        narrow = _model("gelb", 60.0, 7.0)
        broad = _model("braun", 11.0, 48.0)
        assert bin_mask(hue, sat, val, narrow)[0, 0]
        assert bin_mask(hue, sat, val, broad)[0, 0]
        gelb, braun = exclusive_bin_masks(
            hue, sat, val, [narrow, broad], enabled=True
        )
        assert gelb[0, 0] and not braun[0, 0]

    def test_pixels_belong_to_at_most_one_bin(self):
        rng = np.random.default_rng(7)
        hue = rng.uniform(0.0, 360.0, (30, 40))
        sat = np.full(hue.shape, 0.9)
        val = np.full(hue.shape, 0.9)
        models = [
            _model("a", 60.0, 30.0),
            _model("b", 80.0, 45.0),
            _model("c", 200.0, 60.0),
        ]
        masks = exclusive_bin_masks(hue, sat, val, models, enabled=True)
        stacked = np.stack(masks).astype(int).sum(axis=0)
        assert stacked.max() <= 1
        # And never claims a pixel its own gate rejects.
        for model, mask in zip(models, masks):
            assert not (mask & ~bin_mask(hue, sat, val, model)).any()

    def test_order_independent(self):
        rng = np.random.default_rng(11)
        hue = rng.uniform(0.0, 360.0, (20, 25))
        sat = np.full(hue.shape, 0.9)
        val = np.full(hue.shape, 0.9)
        a = _model("a", 60.0, 30.0)
        b = _model("b", 80.0, 45.0)
        forward = exclusive_bin_masks(hue, sat, val, [a, b], enabled=True)
        backward = exclusive_bin_masks(hue, sat, val, [b, a], enabled=True)
        assert np.array_equal(forward[0], backward[1])
        assert np.array_equal(forward[1], backward[0])

    def test_identical_models_tie_and_claim_nothing(self):
        hue, sat, val = _hsv([[60.0, 61.0]])
        twins = [_model("a", 60.0, 8.0), _model("b", 60.0, 8.0)]
        masks = exclusive_bin_masks(hue, sat, val, twins, enabled=True)
        assert not masks[0].any() and not masks[1].any()

    def test_disabled_returns_plain_masks(self):
        hue, sat, val = _hsv([[58.0]])
        models = [_model("gelb", 60.0, 7.0), _model("braun", 11.0, 48.0)]
        masks = exclusive_bin_masks(hue, sat, val, models, enabled=False)
        assert masks[0][0, 0] and masks[1][0, 0]

    def test_grey_pixels_belong_to_nobody(self):
        hue = np.array([[np.nan]])
        sat, val = np.array([[0.9]]), np.array([[0.9]])
        models = [_model("a", 60.0, 30.0), _model("b", 200.0, 30.0)]
        masks = exclusive_bin_masks(hue, sat, val, models, enabled=True)
        assert not masks[0].any() and not masks[1].any()

    def test_bin_match_accept_test_is_the_unchanged_gate(self):
        # A hue exactly at the tolerance edge must stay accepted: the
        # acceptance test is the literal distance comparison, and the
        # reported distance is in degrees.
        model = _model("a", 60.0, 7.0)
        hue, sat, val = _hsv([[67.0]])
        match, dist = bin_match(hue, sat, val, model)
        assert match[0, 0]
        assert dist[0, 0] == pytest.approx(7.0)

    def test_closest_color_wins_regardless_of_tolerance(self):
        """Review regression: ranking by distance/tolerance handed a
        bin's own lid core to whichever neighbour had been calibrated
        more sloppily (the wider learned tolerance)."""
        narrow = _model("a", 60.0, 7.0)
        wide = _model("b", 62.0, 30.0)
        hue, sat, val = _hsv([[59.0]])  # 1 deg from a, 3 deg from b
        a_mask, b_mask = exclusive_bin_masks(
            hue, sat, val, [narrow, wide], enabled=True
        )
        assert a_mask[0, 0] and not b_mask[0, 0]


class TestComponentHoles:
    def test_hole_is_found_and_outside_is_not(self):
        universe = np.ones((20, 20), dtype=bool)
        depth = interior_depth(universe)
        component = np.zeros((20, 20), dtype=bool)
        component[5:15, 5:15] = True
        component[9:11, 9:11] = False  # the sticker
        holes = component_holes(component, universe, depth)
        assert holes is not None
        assert holes[9:11, 9:11].all()
        assert holes.sum() == 4
        assert not holes[0, 0]

    def test_no_holes_when_solid(self):
        universe = np.ones((12, 12), dtype=bool)
        component = np.zeros((12, 12), dtype=bool)
        component[3:8, 3:8] = True
        holes = component_holes(component, universe, interior_depth(universe))
        assert holes is not None and not holes.any()

    def test_component_filling_the_region_has_no_outside(self):
        universe = np.ones((8, 8), dtype=bool)
        holes = component_holes(universe, universe, interior_depth(universe))
        assert holes is None

    def test_veto_qualification_requires_credible_evidence(self):
        good = _model(
            "b",
            11.0,
            20.0,
            veto_qualify_separable=True,
            veto_qualify_provisional=False,
            veto_qualify_min_area_frac=0.01,
        )
        assert veto_qualified(good, 0.02)
        assert not veto_qualified(good, 0.005)  # below its weakest lid
        for broken in (
            {"veto_qualify_separable": False, "veto_qualify_provisional": False,
             "veto_qualify_min_area_frac": 0.01},
            {"veto_qualify_separable": True, "veto_qualify_provisional": True,
             "veto_qualify_min_area_frac": 0.01},
            {},
        ):
            assert not veto_qualified(_model("b", 11.0, 20.0, **broken), 0.02)


def _sticker_setup(qualified: bool):
    """A big yellow lid with a small blue sticker inside it."""
    height = width = 40
    hue = np.full((height, width), 300.0)  # background: matches nobody
    hue[10:30, 10:30] = 60.0  # the lid
    hue[18:22, 18:22] = 210.0  # the sticker on the lid
    sat = np.full(hue.shape, 0.9)
    val = np.full(hue.shape, 0.9)
    lid = _model(
        "gelb",
        60.0,
        8.0,
        veto_qualify_separable=qualified,
        veto_qualify_provisional=False,
        veto_qualify_min_area_frac=0.01,
    )
    other = _model(
        "blau",
        210.0,
        8.0,
        veto_qualify_separable=True,
        veto_qualify_provisional=False,
        veto_qualify_min_area_frac=0.01,
    )
    universe = np.ones(hue.shape, dtype=bool)
    return hue, sat, val, [lid, other], universe, interior_depth(universe)


class TestOccupancyVeto:
    def test_sticker_inside_a_lid_is_vetoed(self):
        hue, sat, val, models, universe, depth = _sticker_setup(True)
        candidates = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=depth,
            denom=universe.size, exclusion=True,
        )
        lid, sticker = candidates
        assert sticker.area == 0
        assert sticker.excluded_by == "gelb"
        assert sticker.provisional_area == 16  # what it measured before
        # The container's own area is NEVER inflated by hole filling.
        assert lid.area == 20 * 20 - 16
        assert lid.excluded_by is None

    def test_unqualified_container_reports_a_conflict_instead(self):
        hue, sat, val, models, universe, depth = _sticker_setup(False)
        _lid, sticker = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=depth,
            denom=universe.size, exclusion=True,
        )
        assert sticker.area == 16  # untouched
        assert sticker.excluded_by is None
        assert sticker.conflict_with == "gelb"

    def test_adjacent_bins_are_never_vetoed(self):
        # Two lids side by side; the smaller one's bbox is NOT nested,
        # and even a bbox that nested would not matter: containment is
        # pixel-exact.
        hue = np.full((40, 60), 300.0)
        hue[10:30, 5:25] = 60.0
        hue[15:25, 30:40] = 210.0
        sat = np.full(hue.shape, 0.9)
        val = np.full(hue.shape, 0.9)
        models = [
            _model("gelb", 60.0, 8.0, veto_qualify_separable=True,
                   veto_qualify_provisional=False,
                   veto_qualify_min_area_frac=0.01),
            _model("blau", 210.0, 8.0, veto_qualify_separable=True,
                   veto_qualify_provisional=False,
                   veto_qualify_min_area_frac=0.01),
        ]
        universe = np.ones(hue.shape, dtype=bool)
        gelb, blau = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=interior_depth(universe),
            denom=universe.size, exclusion=True,
        )
        assert gelb.area == 400 and blau.area == 100
        assert gelb.excluded_by is None and blau.excluded_by is None

    def test_veto_layer_off_leaves_the_sticker(self):
        hue, sat, val, models, universe, depth = _sticker_setup(True)
        _lid, sticker = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=depth,
            denom=universe.size, exclusion=True, veto=False,
        )
        assert sticker.area == 16 and sticker.excluded_by is None

    def test_idempotent(self):
        hue, sat, val, models, universe, depth = _sticker_setup(True)
        kwargs = dict(
            universe=universe, region=None, depth=depth,
            denom=universe.size, exclusion=True,
        )
        first = resolve_candidates(hue, sat, val, models, **kwargs)
        second = resolve_candidates(hue, sat, val, models, **kwargs)
        assert [c.area for c in first] == [c.area for c in second]
        assert [c.excluded_by for c in first] == [c.excluded_by for c in second]


class TestProfileValidation:
    def test_broken_veto_stats_are_rejected(self):
        from wastebin_ai_detector.core.errors import ProfileError
        from wastebin_ai_detector.core.profile import Profile, Roi as PRoi
        from wastebin_ai_detector.core.profile import validate_profile

        def profile_with(**stats):
            return Profile(
                roi=PRoi(0.0, 0.0, 1.0, 1.0),
                working_width=160,
                resample="bilinear",
                daylight_sat_min=0.1,
                bins=[_model("a", 60.0, 8.0, **stats)],
            )

        validate_profile(profile_with())  # absent stats stay valid
        validate_profile(
            profile_with(
                veto_qualify_min_area_frac=0.01,
                veto_qualify_separable=True,
                veto_qualify_provisional=False,
            )
        )
        for broken in (
            {"veto_qualify_min_area_frac": "nonsense"},
            {"veto_qualify_min_area_frac": [0.1]},
            {"veto_qualify_min_area_frac": float("nan")},
            {"veto_qualify_min_area_frac": 2.0},
            {"veto_qualify_separable": 1},
            {"veto_qualify_provisional": "yes"},
        ):
            with pytest.raises(ProfileError):
                validate_profile(profile_with(**broken))


class TestResolutionOrderAndDepth:
    """Review regressions: the resolution must not depend on the order
    of profile.bins, and it must not stop after one pass."""

    def _scene_nested(self):
        # A blue square containing a green ring, whose interior holds a
        # red blob: red is enclosed by BOTH green and blue.
        hue = np.full((60, 60), 300.0)
        hue[10:50, 10:50] = 210.0  # blue container
        hue[20:40, 20:40] = 120.0  # green ring
        hue[22:38, 22:38] = 210.0  # ring interior is blue again
        hue[25:35, 25:35] = 0.0  # red blob inside the ring
        sat = np.full(hue.shape, 0.9)
        val = np.full(hue.shape, 0.9)
        return hue, sat, val

    def _bins(self, order):
        made = {
            "rot": _model("rot", 0.0, 8.0, veto_qualify_separable=True,
                          veto_qualify_provisional=False,
                          veto_qualify_min_area_frac=0.001),
            "gruen": _model("gruen", 120.0, 8.0, veto_qualify_separable=True,
                            veto_qualify_provisional=False,
                            veto_qualify_min_area_frac=0.001),
            "blau": _model("blau", 210.0, 8.0, veto_qualify_separable=True,
                           veto_qualify_provisional=False,
                           veto_qualify_min_area_frac=0.001),
        }
        return [made[name] for name in order]

    def _resolve(self, order):
        hue, sat, val = self._scene_nested()
        models = self._bins(order)
        universe = np.ones(hue.shape, dtype=bool)
        cands = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=interior_depth(universe),
            denom=universe.size, exclusion=True,
        )
        return {m.id: c for m, c in zip(models, cands)}

    def test_verdicts_do_not_depend_on_bin_order(self):
        a = self._resolve(["rot", "gruen", "blau"])
        b = self._resolve(["rot", "blau", "gruen"])
        c = self._resolve(["blau", "gruen", "rot"])
        for key in ("rot", "gruen", "blau"):
            assert a[key].area == b[key].area == c[key].area, key
            assert (
                a[key].excluded_by == b[key].excluded_by == c[key].excluded_by
            ), key

    def test_innermost_bin_is_vetoed_in_every_order(self):
        for order in (
            ["rot", "gruen", "blau"],
            ["rot", "blau", "gruen"],
            ["gruen", "rot", "blau"],
            ["blau", "gruen", "rot"],
        ):
            rot = self._resolve(order)["rot"]
            assert rot.area == 0, order
            assert rot.excluded_by is not None, order

    def test_reselected_blob_cannot_enclose_a_third_bin(self):
        """A vetoed bin re-selects elsewhere; that NEW blob may enclose
        a third bin, which the resolution must still catch."""
        hue = np.full((60, 120), 300.0)
        hue[10:50, 5:45] = 60.0  # yellow lid
        hue[20:40, 15:35] = 210.0  # blue sticker on it (largest blue)
        hue[20:38, 70:88] = 210.0  # separate blue donut
        hue[25:33, 75:83] = 0.0  # red blob inside the donut
        sat = np.full(hue.shape, 0.9)
        val = np.full(hue.shape, 0.9)
        models = [
            _model("gelb", 60.0, 8.0, veto_qualify_separable=True,
                   veto_qualify_provisional=False,
                   veto_qualify_min_area_frac=0.001),
            _model("blau", 210.0, 8.0, veto_qualify_separable=True,
                   veto_qualify_provisional=False,
                   veto_qualify_min_area_frac=0.001),
            _model("rot", 0.0, 8.0, veto_qualify_separable=True,
                   veto_qualify_provisional=False,
                   veto_qualify_min_area_frac=0.001),
        ]
        universe = np.ones(hue.shape, dtype=bool)
        gelb, blau, rot = resolve_candidates(
            hue, sat, val, models,
            universe=universe, region=None, depth=interior_depth(universe),
            denom=universe.size, exclusion=True,
        )
        # Blue loses the sticker and falls back to the donut...
        assert blau.excluded_by == "gelb"
        assert blau.area > 0
        # ...and red, now inside blue's FINAL area, must be vetoed too.
        assert rot.area == 0
        assert rot.excluded_by == "blau"


class TestExclusionEndToEnd:
    """The field case: the brown lid carries a white sticker that can
    match another bin's model. The sticker is PERMANENT, so it is part
    of the lid in calibration too - the realistic setup."""

    LID_Y = (0.25, 0.25, 0.30, 0.30)
    LID_B = (0.65, 0.25, 0.20, 0.20)
    # Square in PIXELS (the 320x200 frame is not square in relative
    # coordinates), so the sticker is shape-plausible for the blue
    # model and only the exclusion rule can reject it.
    STICKER = (0.35, 0.35, 0.05, 0.08)

    def _store(self, tmp_path):
        store = CalibrationStore(
            roi=Roi(0.0, 0.0, 1.0, 1.0),
            working_width=160,
            resample="bilinear",
            bins=[BinDecl("gelb", "Gelbe Tonne"), BinDecl("blau", "Blaue")],
        )
        for i, seed in enumerate((0, 3)):
            name = f"cal{i}.png"
            make_scene(
                rects=[
                    (YELLOW, *self.LID_Y),
                    (BLUE, *self.STICKER),
                    (BLUE, *self.LID_B),
                ],
                seed=seed,
            ).save(tmp_path / name, format="PNG")
            store.add_sample(name, "gelb", Rect(0.27, 0.27, 0.06, 0.06))
            store.add_sample(name, "blau", Rect(0.68, 0.28, 0.08, 0.08))
            store.set_labels(name, present=["gelb", "blau"])
        make_scene(rects=[], seed=9).save(tmp_path / "empty.png", format="PNG")
        store.set_labels("empty.png", absent=["gelb", "blau"])
        return store

    def _sticker_only(self, tmp_path, name="sticker.png"):
        """Bin collected: only the yellow lid with its sticker is left."""
        make_scene(
            rects=[(YELLOW, *self.LID_Y), (BLUE, *self.STICKER)], seed=5
        ).save(tmp_path / name, format="PNG")
        return tmp_path / name

    def test_profile_records_exclusion_and_qualification(self, tmp_path):
        profile, _warnings = learn_profile(
            self._store(tmp_path), tmp_path / "store.json"
        )
        assert profile.mutual_exclusion is True
        for model in profile.bins:
            stats = model.learning_stats
            assert "veto_qualify_min_area_frac" in stats
            assert "veto_qualify_separable" in stats
            assert "veto_qualify_provisional" in stats

    def test_learn_detect_identity(self, tmp_path):
        profile, _warnings = learn_profile(
            self._store(tmp_path), tmp_path / "store.json"
        )
        for model in profile.bins:
            fracs = [
                next(
                    b.area_frac
                    for b in detect_file(tmp_path / name, profile).bins
                    if b.id == model.id
                )
                for name in ("cal0.png", "cal1.png")
            ]
            assert min(fracs) == pytest.approx(
                model.learning_stats["min_pos_area_frac"], abs=0.0
            )

    def test_sticker_on_a_lid_is_not_the_other_bin(self, tmp_path):
        profile, _warnings = learn_profile(
            self._store(tmp_path), tmp_path / "store.json"
        )
        result = detect_file(self._sticker_only(tmp_path), profile)
        gelb = next(b for b in result.bins if b.id == "gelb")
        blau = next(b for b in result.bins if b.id == "blau")
        assert gelb.present is True
        assert blau.area_frac == 0.0
        assert blau.present is False
        assert blau.excluded_by == "gelb"
        # What it would have measured without the rule - the false
        # positive this feature exists to prevent.
        assert blau.provisional_area_frac > 0.0

    def test_legacy_profile_keeps_pre_exclusion_behavior(self, tmp_path):
        profile, _warnings = learn_profile(
            self._store(tmp_path), tmp_path / "store.json"
        )
        profile.mutual_exclusion = False
        result = detect_file(self._sticker_only(tmp_path), profile)
        blau = next(b for b in result.bins if b.id == "blau")
        assert blau.area_frac > 0.0
        assert blau.excluded_by is None
