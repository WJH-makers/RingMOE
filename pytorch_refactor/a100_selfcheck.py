"""Self-check for running the PyTorch/DeepSpeed refactor on NVIDIA A100/H100.

This does a lightweight CUDA+AMP+model forward/backward validation and prints
DeepSpeed/MoE kernel status when available.

Recommended usage on Linux/A100:
  python -m pytorch_refactor.a100_selfcheck
"""

from __future__ import annotations

import platform
import sys

import torch

try:
    from .model import SwinTransformerV2MoE, SimMIM  # type: ignore
except ImportError:  # pragma: no cover
    from model import SwinTransformerV2MoE, SimMIM


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _check_deepspeed() -> None:
    try:
        import deepspeed  # type: ignore

        print("deepspeed:", getattr(deepspeed, "__version__", "unknown"))
    except Exception as e:
        print("deepspeed: not installed (ok for --no_deepspeed).", f"{type(e).__name__}: {e}")
        return

    try:
        from deepspeed.ops.op_builder import MoEBuilder  # type: ignore

        b = MoEBuilder()
        is_compatible = bool(getattr(b, "is_compatible", lambda: True)())
        is_installed = getattr(b, "is_installed", lambda: None)()
        print("deepspeed MoE kernels compatible:", is_compatible)
        if is_installed is not None:
            print("deepspeed MoE kernels installed:", bool(is_installed))
    except Exception as e:
        print("deepspeed MoE kernels: check failed.", f"{type(e).__name__}: {e}")


def _check_cuda() -> None:
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This self-check requires an NVIDIA GPU + CUDA build of PyTorch.")

    props = torch.cuda.get_device_properties(0)
    print("device:", props.name)
    print("compute capability:", (props.major, props.minor))
    print("total memory (GB):", round(props.total_memory / (1024**3), 2))
    print("bf16 supported:", torch.cuda.is_bf16_supported())


def _check_model() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Keep MoE disabled for the self-check (so it doesn't require DS kernels).
    moe_config = {"moe_stages": [], "num_experts": 1}
    encoder = SwinTransformerV2MoE(
        img_size=192,
        embed_dim=96,
        depths=[1, 1, 1, 1],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        use_checkpoint=True,
        moe_config=moe_config,
    )
    model = SimMIM(encoder=encoder).to(device)
    model.train()

    x = torch.randn(1, 3, 192, 192, device=device)
    mask = torch.zeros(1, 48, 48, dtype=torch.int64, device=device)
    mask[:, :24, :24] = 1

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = _make_grad_scaler(enabled=(amp_dtype == torch.float16))

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        loss, x_rec, aux = model(x, mask)
        total = loss + aux

    if scaler.is_enabled():
        scaler.scale(total).backward()
    else:
        total.backward()

    assert x_rec.shape == x.shape, (x_rec.shape, x.shape)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(aux).all()
    print("model fwd/bwd: ok", f"(amp={amp_dtype}, scaler={scaler.is_enabled()})")


def main() -> int:
    try:
        _check_cuda()
        _check_deepspeed()
        _check_model()
    except Exception as e:
        print("SELF-CHECK FAILED:", f"{type(e).__name__}: {e}")
        return 1
    print("SELF-CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

