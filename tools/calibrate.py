#!/usr/bin/env python3
"""Offline calibration CLI for the Wastebin AI Detector.

Subcommands:
  init    create a calibration store (ROI, bins, working width)
  sample  add a lid sample rectangle for one bin in one image
  label   mark bins present/absent in an image
  learn   recompute the full profile from the store
  detect  run detection on an image with a learned profile

Coordinates: the ROI is image-relative (0..1). Sample rectangles are
stored ROI-relative; pass ``--space image`` to enter them relative to
the full image instead (they are converted, pure geometry).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The core lives inside the integration package so HACS ships it;
# for offline use we import it straight from the repository checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "custom_components"))

from wastebin_ai_detector.core import (  # noqa: E402
    BinDecl,
    CalibrationStore,
    Rect,
    Roi,
    WastebinError,
    detect_file,
    learn_profile,
    load_profile,
    load_store,
    save_profile,
    save_store,
)


def _parse_bin(spec: str) -> BinDecl:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"bin spec {spec!r} must be id=Name, e.g. yellow='Yellow bin'"
        )
    bin_id, name = spec.split("=", 1)
    if not bin_id:
        raise argparse.ArgumentTypeError("bin id must not be empty")
    return BinDecl(id=bin_id, name=name or bin_id)


def _store_relative_image(store_path: Path, image: Path) -> str:
    """Store image paths relative to the store file when possible."""
    try:
        return str(image.resolve().relative_to(store_path.resolve().parent))
    except ValueError:
        return str(image.resolve())


def cmd_init(args: argparse.Namespace) -> int:
    store = CalibrationStore(
        roi=Roi(*args.roi),
        working_width=args.width,
        resample=args.resample,
        bins=args.bin,
    )
    save_store(store, args.store)
    print(f"store written: {args.store} ({len(store.bins)} bins)")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    store = load_store(store_path)
    rect = Rect(*args.rect)
    if args.space == "image":
        rect = store.image_rect_to_roi_rect(rect)
    image = _store_relative_image(store_path, Path(args.image))
    store.add_sample(image, args.bin, rect)
    save_store(store, store_path)
    print(f"sample added: {image} / {args.bin} / rect(roi)={rect}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    store = load_store(store_path)
    image = _store_relative_image(store_path, Path(args.image))
    store.set_labels(image, present=args.present, absent=args.absent)
    save_store(store, store_path)
    entry = store.get_image(image)
    print(f"labels for {image}: present={entry.present} absent={entry.absent}")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    store = load_store(store_path)
    profile, warnings = learn_profile(store, store_path)
    save_profile(profile, args.profile)
    print(f"profile written: {args.profile}")
    for bin_model in profile.bins:
        stats = bin_model.learning_stats
        print(
            f"  {bin_model.id}: hue {bin_model.hue_center_deg:.1f}°"
            f"±{bin_model.hue_tol_deg:.1f}°, sat≥{bin_model.sat_min:.2f}, "
            f"val≥{bin_model.val_min:.2f}, area≥{bin_model.min_area_frac:.4f} "
            f"(pos={stats.get('n_pos')}, neg={stats.get('n_neg')}, "
            f"separable={stats.get('separable')}, "
            f"provisional={stats.get('provisional')})"
        )
    daylight = profile.daylight_stats
    print(
        f"  daylight sat floor: {profile.daylight_sat_min:.3f} "
        f"(min of {daylight.get('n_images')} images, "
        f"median {daylight.get('median_of_medians'):.3f})"
    )
    for warning in warnings:
        print(f"  WARNING: {warning}", file=sys.stderr)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    result = detect_file(args.image, profile)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        for b in result.bins:
            status = "PRESENT" if b.present else "absent "
            print(
                f"{status}  {b.name} (area {b.area_frac:.4f} vs "
                f"threshold {b.min_area_frac:.4f}, margin {b.margin:.2f})"
            )
        if result.grayscale_suspect:
            print(
                f"NOTE: image looks greyscale/IR (median sat "
                f"{result.median_sat:.3f}) — result unreliable"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a calibration store")
    p_init.add_argument("--store", required=True)
    p_init.add_argument(
        "--roi", nargs=4, type=float, required=True, metavar=("X", "Y", "W", "H"),
        help="region of interest, image-relative 0..1",
    )
    p_init.add_argument(
        "--width", type=int, default=None,
        help="working width in px (default: native ROI width)",
    )
    p_init.add_argument("--resample", choices=("bilinear", "nearest"), default="bilinear")
    p_init.add_argument(
        "--bin", action="append", required=True, type=_parse_bin,
        metavar="ID=NAME", help="declare a bin (repeatable)",
    )
    p_init.set_defaults(func=cmd_init)

    p_sample = sub.add_parser("sample", help="add a lid sample rectangle")
    p_sample.add_argument("--store", required=True)
    p_sample.add_argument("--image", required=True)
    p_sample.add_argument("--bin", required=True)
    p_sample.add_argument(
        "--rect", nargs=4, type=float, required=True, metavar=("X", "Y", "W", "H"),
    )
    p_sample.add_argument("--space", choices=("roi", "image"), default="roi")
    p_sample.set_defaults(func=cmd_sample)

    p_label = sub.add_parser("label", help="label bins present/absent in an image")
    p_label.add_argument("--store", required=True)
    p_label.add_argument("--image", required=True)
    p_label.add_argument("--present", action="append", default=[])
    p_label.add_argument("--absent", action="append", default=[])
    p_label.set_defaults(func=cmd_label)

    p_learn = sub.add_parser("learn", help="recompute the profile from the store")
    p_learn.add_argument("--store", required=True)
    p_learn.add_argument("--profile", required=True)
    p_learn.set_defaults(func=cmd_learn)

    p_detect = sub.add_parser("detect", help="run detection on an image")
    p_detect.add_argument("--profile", required=True)
    p_detect.add_argument("--image", required=True)
    p_detect.add_argument("--json", action="store_true")
    p_detect.set_defaults(func=cmd_detect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except WastebinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
