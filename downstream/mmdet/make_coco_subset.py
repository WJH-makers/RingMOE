from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_repo_root() -> Path:
    # downstream/mmdet/make_coco_subset.py -> repo_root
    return Path(__file__).resolve().parents[2]


def _pick_ann_file(isaid_root: Path, split: str) -> Path:
    ann_dir = isaid_root / split / "Annotations"
    candidates = sorted(ann_dir.glob(f"iSAID_{split}*.json"))
    if not candidates:
        return ann_dir / f"iSAID_{split}.json"
    return max(candidates, key=lambda p: p.stat().st_size)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _existing_image_names(images_dir: Path) -> set[str]:
    if not images_dir.is_dir():
        return set()
    return {p.name for p in images_dir.iterdir() if p.is_file()}


def _subset_coco(data: dict, existing_names: set[str] | None, max_images: int) -> dict:
    images = data.get("images", [])
    anns = data.get("annotations", [])

    if not isinstance(images, list) or not isinstance(anns, list):
        raise ValueError("Invalid COCO JSON: 'images' and 'annotations' must be lists.")

    keep_images: list[dict] = []
    keep_ids: set[int] = set()

    # Deterministic order: by file_name (fallback to id).
    def _img_sort_key(img: dict) -> tuple[str, int]:
        fn = img.get("file_name")
        if not isinstance(fn, str):
            fn = ""
        img_id = img.get("id")
        if not isinstance(img_id, int):
            img_id = -1
        return (fn, img_id)

    sorted_images = [img for img in images if isinstance(img, dict)]
    sorted_images.sort(key=_img_sort_key)

    for img in sorted_images:
        fn = img.get("file_name")
        img_id = img.get("id")
        if not isinstance(fn, str) or not isinstance(img_id, int):
            continue
        if existing_names is not None and fn not in existing_names:
            continue
        keep_images.append(img)
        keep_ids.add(img_id)
        if max_images > 0 and len(keep_images) >= max_images:
            break

    keep_anns = [
        ann
        for ann in anns
        if isinstance(ann, dict) and isinstance(ann.get("image_id"), int) and ann["image_id"] in keep_ids
    ]

    out = dict(data)
    out["images"] = keep_images
    out["annotations"] = keep_anns
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create a smaller COCO annotation JSON by keeping only images that exist in images_dir, "
            "optionally limiting to the first N images (sorted by file_name)."
        )
    )
    ap.add_argument(
        "--isaid_root",
        type=Path,
        default=_default_repo_root() / "datasets",
        help="iSAID dataset root (default: repo_root/datasets).",
    )
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument(
        "--max_images",
        type=int,
        default=200,
        help="Keep at most N images (0 means all existing images).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <isaid_root>/<split>/Annotations/iSAID_<split>_subset_<N>.json).",
    )
    ap.add_argument(
        "--no_filter_existing",
        action="store_true",
        help=(
            "Do not check/filter by files in images_dir; just take the first N images from the COCO JSON "
            "(still sorted by file_name). Useful when you want to extract only a subset from zip archives."
        ),
    )
    args = ap.parse_args()

    isaid_root = args.isaid_root.resolve()
    split = args.split
    images_dir = isaid_root / split / "images"
    ann_file = _pick_ann_file(isaid_root, split)

    if not ann_file.is_file():
        print(f"[fail] Missing annotation file: {ann_file}", file=sys.stderr)
        return 2
    if args.max_images < 0:
        print("[fail] --max_images must be >= 0", file=sys.stderr)
        return 2

    out_path = args.out
    if out_path is None:
        suffix = "all" if args.max_images == 0 else str(args.max_images)
        out_path = isaid_root / split / "Annotations" / f"iSAID_{split}_subset_{suffix}.json"
    if not out_path.is_absolute():
        out_path = (_default_repo_root() / out_path).resolve()

    existing = None
    if not args.no_filter_existing:
        if not images_dir.is_dir():
            print(f"[fail] Missing images dir: {images_dir}", file=sys.stderr)
            print("[hint] If you haven't extracted images yet, rerun with --no_filter_existing.", file=sys.stderr)
            return 2
        existing = _existing_image_names(images_dir)
        if not existing:
            print(f"[fail] No files found in {images_dir}", file=sys.stderr)
            print("[hint] If you haven't extracted images yet, rerun with --no_filter_existing.", file=sys.stderr)
            return 2

    data = _load_json(ann_file)
    subset = _subset_coco(data, existing_names=existing, max_images=args.max_images)

    _dump_json(out_path, subset)

    n_images = len(subset.get("images", []))
    n_anns = len(subset.get("annotations", []))
    print(f"[ok] wrote subset: split={split} images={n_images} anns={n_anns} -> {out_path}")
    print(f"[info] base_ann={ann_file}")
    if existing is not None:
        print(f"[info] images_dir={images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
