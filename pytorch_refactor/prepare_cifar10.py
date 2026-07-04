"""Download CIFAR-10 and export as an ImageFolder-style directory + data.json.

This is a convenience script to produce the JSON format expected by
`pytorch_refactor/train.py` (a list of image paths).

Usage:
  python pytorch_refactor/prepare_cifar10.py --out_dir data/cifar10 --split train
  python pytorch_refactor/prepare_cifar10.py --out_dir data/cifar10 --split train --limit 2000

Output:
  <out_dir>/
    images/
      000000.png
      ...
    data.json   # list of relative paths like ["images/000000.png", ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare CIFAR-10 for RingMoE PyTorch refactor")
    p.add_argument("--out_dir", type=str, default="data/cifar10", help="output directory")
    p.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="where to download CIFAR-10 (default: <out_dir>/download_cache)",
    )
    p.add_argument("--split", type=str, choices=["train", "test"], default="train")
    p.add_argument("--limit", type=int, default=None, help="optional cap on number of images to export")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing images/data.json")
    return p.parse_args()


def main() -> int:
    args = _get_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else (out_dir / "download_cache")
    download_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    if images_dir.exists() and args.overwrite:
        for p in images_dir.glob("*.png"):
            p.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    data_json = out_dir / "data.json"
    if data_json.exists() and args.overwrite:
        data_json.unlink()

    try:
        from torchvision.datasets import CIFAR10  # type: ignore
    except Exception as e:
        raise SystemExit(
            "torchvision is required to download CIFAR-10. Install it first (with CUDA-enabled torch).\n"
            f"Root cause: {type(e).__name__}: {e}"
        ) from e

    train = args.split == "train"
    ds = CIFAR10(root=str(download_dir), train=train, download=True)

    limit = int(args.limit) if args.limit is not None else len(ds)
    limit = max(0, min(limit, len(ds)))

    rel_paths: list[str] = []
    for i in range(limit):
        img, _label = ds[i]  # PIL image
        fn = f"{i:06d}.png"
        out_path = images_dir / fn
        img.save(out_path)
        rel_paths.append(str(Path("images") / fn).replace(os.sep, "/"))

    data_json.write_text(json.dumps(rel_paths, ensure_ascii=False, indent=2), encoding="utf-8")

    print("wrote:", str(data_json))
    print("count:", len(rel_paths))
    print("example:", rel_paths[0] if rel_paths else "<empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

