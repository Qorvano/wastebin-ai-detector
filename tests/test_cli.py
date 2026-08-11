"""CLI smoke test: the documented workflow end-to-end, in process."""

from __future__ import annotations

import json

from scenes import inner_rect
from test_pipeline_e2e import (
    BINS,
    RECT_BLUE,
    RECT_BROWN,
    RECT_YELLOW,
    ROI,
    scene_all,
    scene_no_yellow,
)

import calibrate


def test_cli_workflow(tmp_path, capsys):
    p_all = tmp_path / "all.png"
    p_missing = tmp_path / "missing.png"
    scene_all().save(p_all)
    scene_no_yellow().save(p_missing)
    store = tmp_path / "store.json"
    profile = tmp_path / "profile.json"

    assert (
        calibrate.main(
            [
                "init",
                "--store", str(store),
                "--roi", str(ROI.x), str(ROI.y), str(ROI.w), str(ROI.h),
                "--width", "160",
            ]
            + [arg for b in BINS for arg in ("--bin", f"{b.id}={b.name}")]
        )
        == 0
    )

    for bin_id, rect in (
        ("gelb", RECT_YELLOW),
        ("blau", RECT_BLUE),
        ("braun", RECT_BROWN),
    ):
        x, y, w, h = inner_rect(rect)
        assert (
            calibrate.main(
                [
                    "sample",
                    "--store", str(store),
                    "--image", str(p_all),
                    "--bin", bin_id,
                    "--rect", str(x), str(y), str(w), str(h),
                    "--space", "image",
                ]
            )
            == 0
        )

    assert (
        calibrate.main(
            [
                "label", "--store", str(store), "--image", str(p_all),
                "--present", "gelb", "--present", "blau", "--present", "braun",
            ]
        )
        == 0
    )
    assert (
        calibrate.main(
            [
                "label", "--store", str(store), "--image", str(p_missing),
                "--present", "blau", "--present", "braun", "--absent", "gelb",
            ]
        )
        == 0
    )

    assert (
        calibrate.main(
            ["learn", "--store", str(store), "--profile", str(profile)]
        )
        == 0
    )

    capsys.readouterr()
    assert (
        calibrate.main(
            ["detect", "--profile", str(profile), "--image", str(p_missing), "--json"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    presence = {b["id"]: b["present"] for b in result["bins"]}
    assert presence == {"gelb": False, "blau": True, "braun": True}


def test_cli_unknown_bin_returns_error(tmp_path, capsys):
    store = tmp_path / "store.json"
    calibrate.main(
        ["init", "--store", str(store), "--roi", "0.2", "0.2", "0.5", "0.5",
         "--bin", "gelb=Gelbe Tonne"]
    )
    p = tmp_path / "img.png"
    scene_all().save(p)
    rc = calibrate.main(
        ["sample", "--store", str(store), "--image", str(p),
         "--bin", "lila", "--rect", "0.4", "0.4", "0.05", "0.05", "--space", "image"]
    )
    assert rc == 1
    assert "error:" in capsys.readouterr().err
