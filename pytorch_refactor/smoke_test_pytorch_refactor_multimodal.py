"""Smoke test for pytorch_refactor multi-modal SimMIM.

Validates:
- per-modality channel counts (e.g. 3/8/3/4)
- modality-specific decoders (reconstruction shapes)
- forward/backward on CPU without DeepSpeed

Usage:
  python pytorch_refactor/smoke_test_pytorch_refactor_multimodal.py
"""

from __future__ import annotations

import torch

try:
    from .model import MultiModalSimMIM, MultiModalSwinTransformerV2MoE  # type: ignore
except ImportError:  # pragma: no cover
    from model import MultiModalSimMIM, MultiModalSwinTransformerV2MoE


def main() -> int:
    torch.manual_seed(0)

    modal_in_chans = [3, 8, 3, 4]

    # Keep the model tiny so it runs quickly on CPU.
    encoder = MultiModalSwinTransformerV2MoE(
        img_size=192,
        modal_in_chans=modal_in_chans,
        embed_dim=96,
        depths=[1, 1, 1, 1],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        moe_config=None,
    )
    model = MultiModalSimMIM(encoder=encoder, modal_in_chans=modal_in_chans)

    B = 2
    xs = [
        torch.randn(B, 3, 192, 192),
        torch.randn(B, 8, 192, 192),
        torch.randn(B, 3, 192, 192),
        torch.randn(B, 4, 192, 192),
    ]
    masks = []
    for _ in range(len(xs)):
        m = torch.zeros(B, 48, 48, dtype=torch.int64)
        m[:, :24, :24] = 1
        masks.append(m)

    loss, recons, aux = model(xs[0], masks[0], xs[1], masks[1], xs[2], masks[2], xs[3], masks[3])
    total = loss + aux
    total.backward()

    assert len(recons) == 4
    assert recons[0].shape == xs[0].shape
    assert recons[1].shape == xs[1].shape
    assert recons[2].shape == xs[2].shape
    assert recons[3].shape == xs[3].shape
    assert torch.isfinite(loss).all()
    assert torch.isfinite(aux).all()

    print("OK pytorch_refactor multi-modal smoke test")
    print("loss=", float(loss.detach()))
    print("aux=", float(aux.detach()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
