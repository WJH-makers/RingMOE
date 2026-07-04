from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path


_IMG_RE = re.compile(r"^P\d+\.png$")


def _default_isaid_root() -> Path:
    # downstream/mmdet/prepare_isaid.py -> repo_root/datasets
    return Path(__file__).resolve().parents[2] / "datasets"


def _list_pngs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def _find_zip_candidates(split_root: Path) -> list[Path]:
    if not split_root.is_dir():
        return []
    return sorted(split_root.rglob("*.zip"))


def _zip_has_original_images(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if _IMG_RE.match(base):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def _extract_original_images(zip_path: Path, out_dir: Path, wanted_files=None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if not _IMG_RE.match(base):
                continue
            # Optional: only extract images referenced by the COCO JSON for this split.
            if wanted_files is not None and base not in wanted_files:
                continue
            dst = out_dir / base
            if dst.exists():
                continue
            with zf.open(name) as src, dst.open("wb") as f:
                f.write(src.read())
            extracted += 1
    return extracted


def _load_coco_images_and_categories(ann_file: Path) -> tuple[list[dict], list[dict]]:
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("images", []), data.get("categories", [])


def _pick_ann_file(isaid_root: Path, split: str) -> Path:
    ann_dir = isaid_root / split / "Annotations"
    # Prefer timestamped files (e.g. iSAID_train_20190823_114751.json) when present.
    candidates = sorted(ann_dir.glob(f"iSAID_{split}*.json"))
    if not candidates:
        return ann_dir / f"iSAID_{split}.json"
    # Pick the largest file (usually the full release vs. subset).
    return max(candidates, key=lambda p: p.stat().st_size)


def _resolve_ann_file(isaid_root: Path, ann_file: Path) -> Path:
    if ann_file.is_absolute():
        return ann_file
    return isaid_root / ann_file


def _check_split(
    isaid_root: Path,
    split: str,
    extract: bool,
    max_missing_print: int,
    ann_file_override: Path = None,
) -> int:
    ann_file = ann_file_override or _pick_ann_file(isaid_root, split)
    ann_file = _resolve_ann_file(isaid_root, ann_file)
    img_dir = isaid_root / split / "images"

    if not ann_file.is_file():
        print(f"[fail] Missing annotation file: {ann_file}", file=sys.stderr)
        return 2

    images, categories = _load_coco_images_and_categories(ann_file)
    file_names = [img.get("file_name") for img in images if isinstance(img, dict)]
    file_names = [fn for fn in file_names if isinstance(fn, str)]

    wanted = set(file_names)
    present = {p.name for p in _list_pngs(img_dir)}
    to_extract = wanted - present

    if extract and to_extract:
        zip_roots = [isaid_root / split]
        # Common cache directory used by this repo's iSAID prep (downloaded DOTA zips).
        if (isaid_root / ".dota_cache").is_dir():
            zip_roots.append(isaid_root / ".dota_cache")

        zip_candidates: list[Path] = []
        for root in zip_roots:
            zip_candidates.extend(_find_zip_candidates(root))
        zip_candidates = sorted(set(zip_candidates))
        zip_candidates = [z for z in zip_candidates if _zip_has_original_images(z)]
        if zip_candidates:
            total = 0
            for z in zip_candidates:
                n = _extract_original_images(z, img_dir, wanted_files=to_extract)
                print(f"[info] Extracted {n} images from {z} -> {img_dir}")
                total += n
            if total == 0:
                print(f"[warn] No new images extracted for split='{split}'.", file=sys.stderr)
        else:
            print(
                f"[warn] No zip archives with original images found under {isaid_root / split} "
                f"(or {isaid_root / '.dota_cache'}). "
                f"Please download/extract the {split} images so that {img_dir} contains Pxxxx.png files.",
                file=sys.stderr,
            )

    pngs = _list_pngs(img_dir)
    if not pngs:
        print(
            f"[fail] No images found in {img_dir}. Expected files like P0000.png that match the JSON file_name.",
            file=sys.stderr,
        )
        return 2

    missing: list[str] = []
    for fn in file_names:
        if not (img_dir / fn).is_file():
            missing.append(fn)

    print(
        f"[ok] split={split} ann_file={ann_file} images_dir={img_dir} "
        f"pngs={len(pngs)} ann_images={len(file_names)}"
    )
    if categories:
        cat_sorted = sorted(
            [(c.get('id'), c.get('name')) for c in categories if isinstance(c, dict)],
            key=lambda x: (x[0] is None, x[0]),
        )
        print(f"[info] split={split} categories={cat_sorted}")

    if missing:
        print(f"[fail] split={split} missing_images={len(missing)} (showing up to {max_missing_print}):", file=sys.stderr)
        for fn in missing[:max_missing_print]:
            print(f"  - {fn}", file=sys.stderr)
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/check iSAID layout for MMDetection.")
    parser.add_argument(
        "--isaid_root",
        type=Path,
        default=_default_isaid_root(),
        help="iSAID dataset root (default: repo_root/datasets).",
    )
    parser.add_argument(
        "--train_ann",
        type=Path,
        default=None,
        help="Override train annotation JSON (relative to isaid_root or absolute).",
    )
    parser.add_argument(
        "--val_ann",
        type=Path,
        default=None,
        help="Override val annotation JSON (relative to isaid_root or absolute).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="all",
        help="Which split to check/prepare.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Try to extract original images from any zip archives under <split>/ or .dota_cache/.",
    )
    parser.add_argument(
        "--max-missing-print",
        type=int,
        default=20,
        help="How many missing file names to print on error.",
    )

    args = parser.parse_args()
    isaid_root = args.isaid_root.resolve()

    if not isaid_root.is_dir():
        print(f"[fail] isaid_root is not a directory: {isaid_root}", file=sys.stderr)
        return 2

    splits = ("train", "val") if args.split == "all" else (args.split,)
    rc = 0
    for split in splits:
        ann_override = args.train_ann if split == "train" else args.val_ann
        rc = max(
            rc,
            _check_split(
                isaid_root,
                split,
                extract=args.extract,
                max_missing_print=args.max_missing_print,
                ann_file_override=ann_override,
            ),
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
