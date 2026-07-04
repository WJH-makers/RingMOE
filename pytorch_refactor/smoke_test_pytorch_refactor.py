"""Smoke test for pytorch_refactor.

Runs a tiny forward/backward step on CPU to validate shapes and loss plumbing.
Does not require DeepSpeed.

Usage:
  python pytorch_refactor/smoke_test_pytorch_refactor.py
"""

from __future__ import annotations

import torch

try:
    from .model import SwinTransformerV2MoE, SimMIM  # type: ignore
except ImportError:  # pragma: no cover
    from model import SwinTransformerV2MoE, SimMIM


def main() -> int:
    torch.manual_seed(0)

    # IMPORTANT: keep MoE disabled here so the test can run without deepspeed installed.
    moe_config = {"moe_stages": [], "num_experts": 1}

    encoder = SwinTransformerV2MoE(
        img_size=192,
        embed_dim=96,
        depths=[1, 1, 1, 1],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        moe_config=moe_config,
    )
    model = SimMIM(encoder=encoder)

    x = torch.randn(2, 3, 192, 192)
    mask = torch.zeros(2, 48, 48, dtype=torch.int64)
    mask[:, :24, :24] = 1

    loss, x_rec, aux = model(x, mask)
    total = loss + aux

    total.backward()

    assert x_rec.shape == x.shape, (x_rec.shape, x.shape)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(aux).all()

    print("OK pytorch_refactor smoke test")
    print("loss=", float(loss.detach()))
    print("aux=", float(aux.detach()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
