import argparse
import os
import time
import random
import torch
import json
import numpy as np
from contextlib import nullcontext
from typing import Optional

# DeepSpeed is required for actual multi-GPU ZeRO/MoE training.
# Keep import optional so the module can still be imported in minimal environments.
try:
    import deepspeed  # type: ignore
except Exception:  # pragma: no cover
    deepspeed = None

from torch.utils.data import DataLoader

try:
    from .dataset import RingMoEDataset  # type: ignore
    from .model import (  # type: ignore
        MultiModalSimMIM,
        MultiModalSwinTransformerV2MoE,
        SimMIM,
        SwinTransformerV2MoE,
    )
except ImportError:  # pragma: no cover
    from dataset import RingMoEDataset
    from model import MultiModalSimMIM, MultiModalSwinTransformerV2MoE, SimMIM, SwinTransformerV2MoE


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_tf32(enable: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable else "highest")


def _get_local_rank(args) -> Optional[int]:
    env_local_rank = os.environ.get("LOCAL_RANK", None)
    if env_local_rank is not None:
        try:
            return int(env_local_rank)
        except ValueError:
            return None
    arg_local_rank = getattr(args, "local_rank", None)
    if isinstance(arg_local_rank, int) and arg_local_rank >= 0:
        return arg_local_rank
    return None


def get_args():
    parser = argparse.ArgumentParser(description='RingMoE Pretraining (PyTorch + DeepSpeed)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to dataset json')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader workers')
    parser.add_argument('--moe_experts', type=int, default=8,
                        help='number of experts for MoE')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='checkpoint output directory')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='path to a DeepSpeed checkpoint directory to resume from')
    parser.add_argument('--resume_tag', type=str, default=None,
                        help='checkpoint tag to load (defaults to latest)')
    parser.add_argument('--no_deepspeed', action='store_true',
                        help='disable DeepSpeed and run a single-GPU smoke test')
    parser.add_argument('--disable_moe', action='store_true',
                        help='disable MoE blocks; useful for quick smoke or if DS MoE ops are unavailable')
    parser.add_argument('--mask_ratio', type=float, default=0.6,
                        help='mask ratio for SimMIM')
    parser.add_argument('--input_size', type=int, default=192,
                        help='input image size')
    parser.add_argument('--modal_num', type=int, default=1,
                        help='number of modalities per sample (JSON must contain list-of-paths when >1)')
    parser.add_argument('--modal_in_chans', type=str, default=None,
                        help='comma-separated input channels per modality, e.g. "3,8,3,4"')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--tf32', action='store_true',
                        help='enable TF32 matmul/cudnn (recommended on A100/H100)')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='enable gradient checkpointing inside Swin blocks')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size when --no_deepspeed is used')
    parser.add_argument('--micro_batch', type=int, default=None,
                        help='override train_micro_batch_size_per_gpu for quick tuning')
    parser.add_argument('--dry_run', action='store_true',
                        help='load one batch, print shapes, and exit (no training)')
    parser.add_argument('--force_fp16', action='store_true',
                        help='force fp16 in DeepSpeed config (disables bf16)')
    parser.add_argument('--amp', action='store_true',
                        help='enable autocast AMP when --no_deepspeed is used')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (no_deepspeed only)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (no_deepspeed only)')
    parser.add_argument('--aux_loss_factor', type=float, default=0.001,
                        help='scale factor for MoE auxiliary loss')
    parser.add_argument('--log_every', type=int, default=10,
                        help='log every N steps')

    # DeepSpeed adds: --deepspeed, --deepspeed_config, --local_rank, ...
    if deepspeed is not None:
        parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def _dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _make_grad_scaler(enabled: bool):
    # torch.cuda.amp.GradScaler is deprecated in PyTorch 2.9+.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _maybe_warn_deepspeed_moe_kernels() -> None:
    if deepspeed is None:
        return
    try:
        from deepspeed.ops.op_builder import MoEBuilder  # type: ignore
    except Exception:
        print(
            "[warn] DeepSpeed MoE kernels status unknown (MoEBuilder not available). "
            "If MoE fails, reinstall deepspeed with DS_BUILD_OPS=1 or pass --disable_moe."
        )
        return

    try:
        builder = MoEBuilder()
        is_compatible = True
        if hasattr(builder, "is_compatible"):
            is_compatible = bool(builder.is_compatible())
        is_installed = None
        if hasattr(builder, "is_installed"):
            is_installed = bool(builder.is_installed())
    except Exception as e:
        print(
            f"[warn] DeepSpeed MoE kernels check failed: {type(e).__name__}: {e}. "
            "If MoE fails, reinstall deepspeed with DS_BUILD_OPS=1 or pass --disable_moe."
        )
        return

    if is_installed is False:
        print(
            "[warn] DeepSpeed MoE kernels are not installed. "
            "MoE may be slow or fail. Consider reinstalling deepspeed with DS_BUILD_OPS=1 "
            "or pass --disable_moe."
        )
    elif is_compatible is False:
        print(
            "[warn] DeepSpeed MoE kernels are not compatible with this environment. "
            "Consider fixing CUDA toolchain/versions or pass --disable_moe."
        )


def main():
    args = get_args()

    use_deepspeed = (deepspeed is not None) and (not args.no_deepspeed)
    if use_deepspeed and not torch.cuda.is_available():
        raise RuntimeError("DeepSpeed training requires CUDA; use --no_deepspeed for a CPU smoke test.")

    local_rank = _get_local_rank(args)
    if use_deepspeed and torch.cuda.is_available() and local_rank is not None:
        torch.cuda.set_device(local_rank)

    _set_seed(args.seed)
    _configure_tf32(args.tf32)

    # Load and adjust DeepSpeed config if needed
    ds_config = None
    default_ds_config = os.path.join(os.path.dirname(__file__), "ds_config.json")
    if use_deepspeed:
        config_path = getattr(args, "deepspeed_config", None) or (default_ds_config if os.path.exists(default_ds_config) else None)
        if config_path is None:
            raise FileNotFoundError("DeepSpeed enabled but no --deepspeed_config provided and default ds_config.json not found")
        with open(config_path, "r", encoding="utf-8") as f:
            ds_config = json.load(f)
        if args.micro_batch is not None:
            ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch
        bf16_supported = torch.cuda.is_bf16_supported()
        if args.force_fp16 or not bf16_supported:
            ds_config.setdefault("fp16", {})
            ds_config.setdefault("bf16", {})
            ds_config["fp16"]["enabled"] = True
            ds_config["bf16"]["enabled"] = False
            print("[info] using fp16 in DeepSpeed config (bf16 unsupported or force_fp16)")
        # Mirror config value to args for loader sizing
        args.train_micro_batch_size_per_gpu = ds_config.get(
            "train_micro_batch_size_per_gpu",
            getattr(args, "train_micro_batch_size_per_gpu", 1),
        )

    # Allow CLI override of DS micro-batch size
    if use_deepspeed and args.micro_batch is not None:
        args.train_micro_batch_size_per_gpu = args.micro_batch

    if not use_deepspeed and args.disable_moe is False:
        # MoE requires DeepSpeed MoE layer; enforce disable_moe when DS is off
        args.disable_moe = True

    if use_deepspeed and not args.disable_moe:
        _maybe_warn_deepspeed_moe_kernels()

    # Dataset
    modal_in_chans = None
    if args.modal_in_chans:
        modal_in_chans = [int(x.strip()) for x in args.modal_in_chans.split(",") if x.strip()]
    if int(args.modal_num) > 1 and modal_in_chans is None:
        if int(args.modal_num) == 4:
            modal_in_chans = [3, 8, 3, 4]
            print("[info] modal_in_chans not provided; defaulting to 3,8,3,4 for modal_num=4")
        else:
            raise ValueError("When --modal_num > 1 you must provide --modal_in_chans, e.g. --modal_in_chans 3,8,3,4")
    if int(args.modal_num) == 1:
        if modal_in_chans is None:
            modal_in_chans = [3]
        if len(modal_in_chans) != 1:
            raise ValueError(f"modal_in_chans must have 1 value when modal_num=1, got {modal_in_chans}")
    else:
        if len(modal_in_chans) != int(args.modal_num):
            raise ValueError(f"modal_in_chans length must equal modal_num ({args.modal_num}), got {modal_in_chans}")

    dataset = RingMoEDataset(
        data_path=args.data_path,
        input_size=args.input_size,
        mask_ratio=args.mask_ratio,
        modal_num=args.modal_num,
        modal_in_chans=modal_in_chans,
    )

    sampler = None
    train_loader = None

    if not use_deepspeed:
        batch_size = args.batch_size
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=bool(args.num_workers and args.num_workers > 0),
        )

    # Model
    moe_config = None
    if not args.disable_moe:
        moe_config = {
            'moe_stages': [2, 3],
            'num_experts': args.moe_experts
        }

    if int(args.modal_num) == 1:
        encoder = SwinTransformerV2MoE(
            img_size=args.input_size,
            in_chans=int(modal_in_chans[0]),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=6,
            use_checkpoint=args.use_checkpoint,
            moe_config=moe_config,
        )
        model = SimMIM(encoder=encoder)
    else:
        encoder = MultiModalSwinTransformerV2MoE(
            img_size=args.input_size,
            modal_in_chans=modal_in_chans,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=6,
            use_checkpoint=args.use_checkpoint,
            moe_config=moe_config,
        )
        model = MultiModalSimMIM(encoder=encoder, modal_in_chans=modal_in_chans)

    if args.dry_run and not use_deepspeed:
        batch = next(iter(train_loader))
        if int(args.modal_num) == 1:
            images, masks = batch
            print(f"dry_run batch shapes: images={tuple(images.shape)}, masks={tuple(masks.shape)}")
        else:
            parts = []
            for i in range(int(args.modal_num)):
                x_i = batch[i * 2]
                m_i = batch[i * 2 + 1]
                parts.append(f"m{i}: x={tuple(x_i.shape)} mask={tuple(m_i.shape)}")
            print("dry_run batch shapes:", " | ".join(parts))
        return

    if not use_deepspeed:
        os.makedirs(args.output_dir, exist_ok=True)
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; running on CPU will be very slow.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        use_amp = bool(args.amp and device.type == "cuda")
        if use_amp:
            if args.force_fp16:
                amp_dtype = torch.float16
            else:
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            scaler = _make_grad_scaler(enabled=(amp_dtype == torch.float16))
            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            print(f"[info] AMP enabled (dtype={amp_dtype}, scaler={scaler.is_enabled()})")
        else:
            scaler = _make_grad_scaler(enabled=False)
            autocast_ctx = nullcontext()

        os.makedirs(args.output_dir, exist_ok=True)
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            for step, batch in enumerate(train_loader):
                if int(args.modal_num) == 1:
                    images, masks = batch
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    model_inputs = (images, masks)
                else:
                    model_inputs = tuple(t.to(device, non_blocking=True) for t in batch)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    loss, _, aux_loss = model(*model_inputs)
                    total_loss = loss + (args.aux_loss_factor * aux_loss)

                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if step % args.log_every == 0:
                    print(
                        f"Epoch: {epoch} Step: {step} "
                        f"Loss: {float(loss.detach().cpu()):.6f} Aux: {float(aux_loss.detach().cpu()):.6f} "
                        f"Total: {float(total_loss.detach().cpu()):.6f}"
                    )
            print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

            # Save a lightweight PyTorch checkpoint for the non-DeepSpeed path.
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
                torch.save({"state_dict": model.state_dict()}, ckpt_path)
                print(f"[ckpt] saved: {ckpt_path}")
        return

    if deepspeed is None:
        raise ImportError("deepspeed is not installed; install it to run distributed training")

    # Initialize distributed if launched by deepspeed/torchrun
    launched_distributed = any(
        os.environ.get(k) is not None for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if launched_distributed and not torch.distributed.is_initialized():
        deepspeed.init_distributed()

    rank, world_size = _dist_info()

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

    batch_size = getattr(args, "train_micro_batch_size_per_gpu", None) or 1
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        if rank == 0:
            if int(args.modal_num) == 1:
                images, masks = batch
                print(f"dry_run batch shapes: images={tuple(images.shape)}, masks={tuple(masks.shape)}")
            else:
                parts = []
                for i in range(int(args.modal_num)):
                    x_i = batch[i * 2]
                    m_i = batch[i * 2 + 1]
                    parts.append(f"m{i}: x={tuple(x_i.shape)} mask={tuple(m_i.shape)}")
                print("dry_run batch shapes:", " | ".join(parts))
        return

    # DeepSpeed Initialization
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Resume (training)
    if args.resume_from:
        loaded, _ = model_engine.load_checkpoint(args.resume_from, load_module_strict=False, tag=args.resume_tag)
        if rank == 0:
            print(f"[resume] loaded={loaded} from={args.resume_from} tag={args.resume_tag or 'latest'}")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model_engine.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            if int(args.modal_num) == 1:
                images, masks = batch
                images = images.to(model_engine.device, non_blocking=True)
                masks = masks.to(model_engine.device, non_blocking=True)
                model_inputs = (images, masks)
            else:
                model_inputs = tuple(t.to(model_engine.device, non_blocking=True) for t in batch)

            loss, _, aux_loss = model_engine(*model_inputs)
            total_loss = loss + (args.aux_loss_factor * aux_loss)

            model_engine.backward(total_loss)
            model_engine.step()

            if step % args.log_every == 0 and rank == 0:
                print(
                    f"Epoch: {epoch} Step: {step} "
                    f"Loss: {float(loss.detach().cpu()):.6f} Aux: {float(aux_loss.detach().cpu()):.6f} "
                    f"Total: {float(total_loss.detach().cpu()):.6f}"
                )

        if rank == 0:
            print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            tag = f"epoch_{epoch}"  # stable tag for resume
            model_engine.save_checkpoint(args.output_dir, tag=tag)
            if rank == 0:
                print(f"[ckpt] saved: {args.output_dir} tag={tag}")


if __name__ == "__main__":
    main()

