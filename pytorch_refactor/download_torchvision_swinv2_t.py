"""Download a torchvision Swin V2 Tiny pretrained checkpoint and save state_dict.

This is used by the A100 quickstart to provide a "model parameters download" step.
The weights are NOT automatically loaded into RingMoE (architectures differ); this
is just a reproducible download step you can later use for custom initialization.

Usage:
  python pytorch_refactor/download_torchvision_swinv2_t.py --out runs/model_params/swin_v2_t.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download torchvision Swin V2 Tiny weights")
    p.add_argument("--out", type=str, required=True, help="output .pt path")
    return p.parse_args()


def main() -> int:
    args = _get_args()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision  # noqa: F401
        from torchvision.models import swin_v2_t  # type: ignore
    except Exception as e:
        raise SystemExit(
            "torchvision is required to download model weights.\n"
            f"Root cause: {type(e).__name__}: {e}"
        ) from e

    # Try modern weights API first, then fall back to older pretrained=True.
    model = None
    try:
        from torchvision.models import Swin_V2_T_Weights  # type: ignore

        weights = Swin_V2_T_Weights.DEFAULT
        model = swin_v2_t(weights=weights)
        print("weights:", str(weights))
    except Exception as e:
        print("[warn] torchvision weights enum not available, falling back to pretrained=True.", f"{type(e).__name__}: {e}")
        model = swin_v2_t(pretrained=True)

    state = model.state_dict()
    torch.save({"state_dict": state}, out_path)
    print("saved:", str(out_path))
    print("keys:", len(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

