import argparse
import os
import torch

try:
    import deepspeed  # type: ignore
except Exception:  # pragma: no cover
    deepspeed = None

try:
    from .model import SwinTransformerV2MoE, SimMIM  # type: ignore
except ImportError:  # pragma: no cover
    from model import SwinTransformerV2MoE, SimMIM

try:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # type: ignore
except Exception:  # pragma: no cover
    get_fp32_state_dict_from_zero_checkpoint = None


def get_args():
    p = argparse.ArgumentParser("Export consolidated PyTorch checkpoint from DeepSpeed checkpoint")
    p.add_argument("--ds_ckpt", type=str, required=True, help="DeepSpeed checkpoint directory")
    p.add_argument("--tag", type=str, default=None, help="checkpoint tag (default: latest)")
    p.add_argument("--out", type=str, default="model.pt", help="output .pt path")
    p.add_argument("--moe_experts", type=int, default=8)
    p.add_argument("--disable_moe", action="store_true", help="export a non-MoE model (use if checkpoint was trained with --disable_moe)")
    p.add_argument("--input_size", type=int, default=192)
    if deepspeed is not None:
        p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def build_model(moe_experts: int, *, input_size: int, disable_moe: bool) -> SimMIM:
    moe_config = None if disable_moe else {"moe_stages": [2, 3], "num_experts": moe_experts}
    encoder = SwinTransformerV2MoE(
        img_size=input_size,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        moe_config=moe_config,
    )
    return SimMIM(encoder=encoder)


def main():
    args = get_args()

    if deepspeed is None:
        raise ImportError("deepspeed is not installed; it is required to load ZeRO-sharded checkpoints")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if get_fp32_state_dict_from_zero_checkpoint is not None:
        # Preferred: DeepSpeed utility that works for ZeRO stage 2/3 checkpoints.
        state = get_fp32_state_dict_from_zero_checkpoint(args.ds_ckpt, tag=args.tag)
        torch.save({"state_dict": state}, args.out)
        print(f"exported: {args.out}")
        return

    # Fallback: Initialize DS engine so it can load ZeRO shards (older DeepSpeed builds).
    model = build_model(args.moe_experts, input_size=args.input_size, disable_moe=bool(args.disable_moe))
    engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
    loaded, _ = engine.load_checkpoint(args.ds_ckpt, tag=args.tag, load_module_strict=False)
    if not loaded:
        raise RuntimeError(f"Failed to load DeepSpeed checkpoint: {args.ds_ckpt} tag={args.tag or 'latest'}")

    if engine.global_rank == 0:
        state = engine.module.state_dict()
        torch.save({"state_dict": state}, args.out)
        print(f"exported: {args.out}")


if __name__ == "__main__":
    main()

