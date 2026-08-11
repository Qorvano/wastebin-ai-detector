"""End-to-end: synthetic scenes through the real calibrate→learn→detect path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from scenes import BLUE, BROWN, YELLOW, inner_rect, make_scene
from wastebin_ai_detector.core import (
    BinDecl,
    CalibrationStore,
    ImageLoadError,
    Rect,
    Roi,
    detect,
    detect_file,
    learn_profile,
    load_image_rgb,
)

ROI = Roi(0.25, 0.25, 0.50, 0.50)

# Lid rectangles (image-relative), all inside the ROI.
RECT_YELLOW = (0.30, 0.30, 0.10, 0.10)
RECT_BLUE = (0.45, 0.35, 0.10, 0.10)
RECT_BROWN = (0.58, 0.50, 0.09, 0.12)

BINS = [
    BinDecl("gelb", "Gelbe Tonne"),
    BinDecl("blau", "Blaue Tonne"),
    BinDecl("braun", "Braune Tonne"),
]


def scene_all(size=(320, 200), seed=0):
    return make_scene(
        size=size,
        rects=[
            (YELLOW, *RECT_YELLOW),
            (BLUE, *RECT_BLUE),
            (BROWN, *RECT_BROWN),
        ],
        seed=seed,
    )


def scene_no_yellow(size=(320, 200), seed=1):
    return make_scene(
        size=size,
        rects=[(BLUE, *RECT_BLUE), (BROWN, *RECT_BROWN)],
        seed=seed,
    )


def scene_shuffled(size=(320, 200), seed=2):
    """Same bins, positions swapped within the ROI - layout must not matter."""
    return make_scene(
        size=size,
        rects=[
            (YELLOW, *RECT_BROWN),
            (BLUE, *RECT_YELLOW),
            (BROWN, *RECT_BLUE),
        ],
        seed=seed,
    )


def _build_store(tmp_path: Path, fmt: str = "PNG", qualities: tuple = ()):
    """Build a calibration store from the two base scenes.

    Documented calibration practice: lid samples from SEVERAL snapshots
    spanning the conditions seen in operation - for JPEG cameras that
    includes encoder quality, so pass e.g. ``qualities=(85, 60)`` to
    calibrate across encodes. Yellow exists only in the "all" scene;
    blue and brown are sampled from every calibration image.
    """
    ext = fmt.lower()
    variants = qualities or (None,)
    all_paths: list[Path] = []
    missing_paths: list[Path] = []
    for q in variants:
        suffix = f"_q{q}" if q is not None else ""
        p_all = tmp_path / f"all{suffix}.{ext}"
        p_missing = tmp_path / f"missing{suffix}.{ext}"
        kwargs = {"quality": q} if q is not None else {}
        scene_all().save(p_all, format=fmt, **kwargs)
        scene_no_yellow().save(p_missing, format=fmt, **kwargs)
        all_paths.append(p_all)
        missing_paths.append(p_missing)

    store = CalibrationStore(
        roi=ROI, working_width=160, resample="bilinear", bins=list(BINS)
    )
    for p_all in all_paths:
        for bin_id, rect in (
            ("gelb", RECT_YELLOW),
            ("blau", RECT_BLUE),
            ("braun", RECT_BROWN),
        ):
            store.add_sample(
                p_all.name,
                bin_id,
                store.image_rect_to_roi_rect(Rect(*inner_rect(rect))),
            )
        store.set_labels(p_all.name, present=["gelb", "blau", "braun"])
    for p_missing in missing_paths:
        for bin_id, rect in (("blau", RECT_BLUE), ("braun", RECT_BROWN)):
            store.add_sample(
                p_missing.name,
                bin_id,
                store.image_rect_to_roi_rect(Rect(*inner_rect(rect))),
            )
        store.set_labels(p_missing.name, present=["blau", "braun"], absent=["gelb"])
    store_path = tmp_path / "store.json"
    return store, store_path


@pytest.fixture
def profile(tmp_path):
    store, store_path = _build_store(tmp_path)
    prof, warnings = learn_profile(store, store_path)
    return prof


def _presence(result):
    return {b.id: b.present for b in result.bins}


def test_learn_produces_sane_models(tmp_path):
    store, store_path = _build_store(tmp_path)
    prof, _warnings = learn_profile(store, store_path)
    by_id = {b.id: b for b in prof.bins}
    assert by_id["gelb"].hue_center_deg == pytest.approx(49.4, abs=3.0)
    assert by_id["blau"].hue_center_deg == pytest.approx(222.9, abs=3.0)
    assert by_id["braun"].hue_center_deg == pytest.approx(25.1, abs=3.0)
    # Yellow has a real negative image → threshold is not provisional.
    assert by_id["gelb"].learning_stats["provisional"] is False
    assert by_id["gelb"].learning_stats["separable"] is True


def test_detect_all_present(profile):
    result = detect(scene_all(), profile)
    assert _presence(result) == {"gelb": True, "blau": True, "braun": True}
    assert all(b.margin > 1.0 for b in result.bins)
    assert result.grayscale_suspect is False


def test_detect_missing_yellow(profile):
    result = detect(scene_no_yellow(), profile)
    assert _presence(result) == {"gelb": False, "blau": True, "braun": True}


def test_positions_do_not_matter(profile):
    result = detect(scene_shuffled(), profile)
    assert _presence(result) == {"gelb": True, "blau": True, "braun": True}


def test_resolution_independence(profile):
    small = detect(scene_all(size=(320, 200)), profile)
    large = detect(scene_all(size=(640, 400)), profile)
    assert _presence(small) == _presence(large)
    for a, b in zip(small.bins, large.bins):
        assert a.area_frac == pytest.approx(b.area_frac, abs=0.01)


def test_exif_orientation_transparent(profile, tmp_path):
    upright = scene_all()
    p_upright = tmp_path / "upright.jpg"
    upright.save(p_upright, format="JPEG", quality=95)

    # Orientation tag 6 means "rotate 90° CW to display"; exif_transpose
    # applies ROTATE_270 (90° CW), so storing the image pre-rotated 90°
    # CCW must round-trip back to the upright scene.
    raw = upright.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    p_rotated = tmp_path / "rotated.jpg"
    raw.save(p_rotated, format="JPEG", quality=95, exif=exif)

    res_upright = detect_file(p_upright, profile)
    res_rotated = detect_file(p_rotated, profile)
    assert _presence(res_upright) == _presence(res_rotated)
    assert _presence(res_rotated) == {"gelb": True, "blau": True, "braun": True}


def test_grayscale_night_frame_flagged(profile):
    grey = ImageOps.grayscale(scene_all()).convert("RGB")
    result = detect(grey, profile)
    assert result.grayscale_suspect is True


def test_jpeg_calibration_survives_recompression(tmp_path):
    store, store_path = _build_store(tmp_path, fmt="JPEG", qualities=(85, 60))
    prof, _warnings = learn_profile(store, store_path)
    p_low = tmp_path / "low.jpg"
    scene_all(seed=3).save(p_low, format="JPEG", quality=60)
    result = detect_file(p_low, prof)
    assert _presence(result) == {"gelb": True, "blau": True, "braun": True}
    p_low_missing = tmp_path / "low_missing.jpg"
    scene_no_yellow(seed=4).save(p_low_missing, format="JPEG", quality=60)
    result = detect_file(p_low_missing, prof)
    assert _presence(result) == {"gelb": False, "blau": True, "braun": True}


def test_image_modes_are_normalized(tmp_path):
    scene = scene_all()
    for mode, name in (("P", "pal.png"), ("RGBA", "rgba.png"), ("L", "grey.png")):
        path = tmp_path / name
        scene.convert(mode).save(path)
        img = load_image_rgb(path)
        assert img.mode == "RGB"


def test_overexposure_gate(profile):
    # Calibration scenes contain essentially no clipped pixels, so the
    # learned ceiling is near zero and a frame with a large blown-out
    # region must trip the gate.
    result = detect(scene_all(), profile)
    assert result.overexposure_suspect is False

    blown = make_scene(
        size=(320, 200),
        rects=[
            ((1.0, 1.0, 1.0), 0.25, 0.25, 0.50, 0.25),
            (BLUE, *RECT_BLUE),
            (BROWN, *RECT_BROWN),
        ],
        seed=6,
    )
    result = detect(blown, profile)
    assert result.overexposure_suspect is True
    assert result.clip_frac > profile.overexposure_clip_max


def test_shrunken_lid_is_uncertain_not_flipped(profile):
    # A lid at 36 percent of its calibrated area lands inside the
    # learned ambiguity interval: below the smallest observed positive,
    # above the largest observed negative (0).
    x, y, w, h = RECT_YELLOW
    small_yellow = make_scene(
        size=(320, 200),
        rects=[
            (YELLOW, x, y, w * 0.6, h * 0.6),
            (BLUE, *RECT_BLUE),
            (BROWN, *RECT_BROWN),
        ],
        seed=7,
    )
    result = detect(small_yellow, profile)
    by_id = {b.id: b for b in result.bins}
    assert by_id["gelb"].uncertain is True
    assert by_id["blau"].uncertain is False


def test_missing_archive_image_is_skipped_with_warning(tmp_path):
    # A store may outlive individual snapshot files; relearn must skip
    # them loudly instead of failing forever.
    store, store_path = _build_store(tmp_path)
    extra = tmp_path / "gone.png"
    scene_all(seed=9).save(extra)
    store.set_labels(extra.name, present=["blau", "braun"], absent=["gelb"])
    extra.unlink()
    prof, warnings = learn_profile(store, store_path)
    assert any("skipping calibration image" in w for w in warnings)
    result = detect(scene_all(), prof)
    assert _presence(result) == {"gelb": True, "blau": True, "braun": True}


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    good = tmp_path / "good.jpg"
    scene_all().save(good, format="JPEG", quality=90)
    path.write_bytes(good.read_bytes()[:120])
    with pytest.raises(ImageLoadError):
        load_image_rgb(path)
