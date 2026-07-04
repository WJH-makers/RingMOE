"""End-to-end single-GPU (A100) quickstart: download CIFAR-10 -> selfcheck -> train.

This script is designed to be "fail-fast with actionable errors":
- If prerequisites are missing, it prints exactly what to fix.
- If CIFAR-10 download fails (offline clusters), it can fall back to a tiny dummy dataset.

Run on Linux + A100:
  python pytorch_refactor/quickstart_single_a100_cifar10.py

Artifacts:
  - Logs: <work_dir>/logs/*.log
  - Data: <data_dir>/ (CIFAR-10 exported) or <work_dir>/dummy_data/ (fallback)
  - Checkpoints: <work_dir>/checkpoints/
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class StepFailure(Exception):
    step: str
    command: list[str]
    returncode: int
    log_path: Path
    tail: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _run_step(step: str, command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = [str(x) for x in command]

    _print_header(f"[STEP] {step}")
    print("[cmd]", " ".join(cmd_list))
    print("[log]", str(log_path))

    tail: deque[str] = deque(maxlen=250)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[step] {step}\n")
        f.write(f"[cmd] {' '.join(cmd_list)}\n\n")
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            tail.append(line)
        rc = proc.wait()

    if rc != 0:
        raise StepFailure(step=step, command=cmd_list, returncode=rc, log_path=log_path, tail="".join(tail))

    print(f"[OK] {step}")


def _python() -> str:
    return sys.executable


def _ensure_torch_cuda_or_die() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        _print_header("[FAIL] PyTorch not installed")
        print("Root cause:", f"{type(e).__name__}: {e}")
        print("\nFix:")
        print("- Install a CUDA-enabled torch + torchvision using the official PyTorch selector for your CUDA/driver.")
        print("- Then verify: python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\"")
        raise SystemExit(2) from e

    import torch

    if not torch.cuda.is_available():
        _print_header("[FAIL] CUDA not available in PyTorch")
        print("Detected:")
        print("- torch:", torch.__version__)
        print("- torch.version.cuda:", torch.version.cuda)
        print("- cuda available:", torch.cuda.is_available())
        print("\nCommon causes:")
        print("- Installed a CPU-only torch wheel.")
        print("- NVIDIA driver is missing/too old for the torch CUDA runtime.")
        print("- Running inside a container without GPU passthrough.")
        print("\nFix:")
        print("- Reinstall CUDA-enabled torch + torchvision from the official PyTorch selector.")
        print("- Verify `nvidia-smi` works and shows an A100.")
        raise SystemExit(3)

    name = torch.cuda.get_device_name(0)
    print("[info] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[info] torch.cuda.device_count:", torch.cuda.device_count())
    try:
        print("[info] torch.cuda.current_device:", torch.cuda.current_device())
    except Exception:
        pass
    print("[info] gpu0:", name)
    print("[info] bf16 supported:", torch.cuda.is_bf16_supported())
    if "A100" not in name.upper():
        print("[warn] GPU is not reported as A100:", name)
        print("       This script will continue, but for your target machine it should be an A100.\n")


def _write_dummy_dataset(work_dir: Path) -> Path:
    """Create a tiny local dataset that always works (no network required)."""
    import json

    import numpy as np

    dummy_dir = work_dir / "dummy_data"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    (dummy_dir / "images").mkdir(parents=True, exist_ok=True)

    # Store as .npy to avoid any image codec issues.
    arr = np.random.rand(64, 64, 3).astype("float32")
    np.save(dummy_dir / "img.npy", arr)
    paths = ["img.npy", "img.npy"]
    (dummy_dir / "data.json").write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    return dummy_dir / "data.json"


def _analyze_failure(err: StepFailure) -> None:
    tail = err.tail
    _print_header(f"[FAIL] {err.step} (exit={err.returncode})")
    print("Command:", " ".join(err.command))
    print("Log saved to:", str(err.log_path))
    print("\n--- last output (tail) ---")
    print(tail.rstrip() if tail.strip() else "<no output captured>")
    print("--- end tail ---\n")

    hints: list[str] = []
    tail_lower = tail.lower()
    if "ModuleNotFoundError" in tail and "torchvision" in tail:
        hints.append("Install torchvision that matches your torch build.")
    if "ModuleNotFoundError" in tail and "torch" in tail:
        hints.append("Install CUDA-enabled torch (CPU-only wheels will not work for A100).")
    if "found no nvidia driver" in tail_lower or "nvidia driver" in tail_lower and "not found" in tail_lower:
        hints.append("NVIDIA driver issue: ensure the host driver is installed and `nvidia-smi` works.")
    if "libcuda.so" in tail_lower or "libcudart.so" in tail_lower:
        hints.append("Missing CUDA libraries: ensure the container/host exposes NVIDIA libs (use nvidia-container-toolkit).")
    if "cudnn" in tail_lower and ("not found" in tail_lower or "could not load" in tail_lower):
        hints.append("cuDNN load failure: use an official PyTorch CUDA wheel (it bundles cuDNN) or fix LD_LIBRARY_PATH.")
    if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
        hints.append("OOM: try smaller `--batch_size 1`, lower `--input_size`, and keep `--use_checkpoint` on.")
    if "cublas_status_alloc_failed" in tail_lower:
        hints.append("OOM/fragmentation (cublas alloc failed): reduce batch/input_size or restart the process to defragment.")
    if "illegal memory access" in tail_lower:
        hints.append("CUDA illegal memory access: try upgrading driver/torch, or re-run with a smaller config to isolate.")
    if "SSL" in tail or "URLError" in tail or "Connection" in tail:
        hints.append("Dataset download/network issue: run on a node with internet or use `--fallback_dummy`.")
    if "PermissionError" in tail or "permission denied" in tail.lower():
        hints.append("Permission issue: choose a writable `--work_dir`/`--data_dir` (e.g., under $HOME).")
    if "no space left on device" in tail_lower:
        hints.append("Disk is full: free space or point `--work_dir/--data_dir` to a larger filesystem.")
    if "file not found" in tail_lower or "no such file or directory" in tail_lower:
        hints.append("Path issue: verify `--data_dir` and that `data.json` exists and paths inside it are valid.")

    if hints:
        print("Hints:")
        for h in hints:
            print("-", h)


def _get_args() -> argparse.Namespace:
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    p = argparse.ArgumentParser("RingMoE single-GPU A100 quickstart (CIFAR-10)")
    p.add_argument("--work_dir", type=str, default="runs/a100_single_cifar10", help="working directory for logs/ckpts")
    p.add_argument("--data_dir", type=str, default="data/cifar10", help="where to export CIFAR-10 images + data.json")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit", type=int, default=256, help="number of images to export (smaller = faster)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--input_size", type=int, default=192)
    p.add_argument("--mask_ratio", type=float, default=0.6)
    if bool_action is None:
        p.add_argument(
            "--amp",
            dest="amp",
            action="store_true",
            default=True,
            help="enable AMP autocast for --no_deepspeed training (default: enabled)",
        )
        p.add_argument("--no_amp", dest="amp", action="store_false", help="disable AMP autocast")

        p.add_argument("--tf32", dest="tf32", action="store_true", default=True, help="enable TF32 (default: enabled)")
        p.add_argument("--no_tf32", dest="tf32", action="store_false", help="disable TF32")

        p.add_argument(
            "--use_checkpoint",
            dest="use_checkpoint",
            action="store_true",
            default=True,
            help="enable activation checkpointing inside Swin blocks (default: enabled)",
        )
        p.add_argument("--no_use_checkpoint", dest="use_checkpoint", action="store_false", help="disable checkpointing")

        p.add_argument(
            "--fallback_dummy",
            dest="fallback_dummy",
            action="store_true",
            default=True,
            help="if CIFAR-10 download fails, generate a tiny dummy dataset and still run training (default: enabled)",
        )
        p.add_argument(
            "--no_fallback_dummy",
            dest="fallback_dummy",
            action="store_false",
            help="do not fall back; fail if CIFAR-10 download fails",
        )

        p.add_argument(
            "--download_model_params",
            dest="download_model_params",
            action="store_true",
            default=True,
            help="download a reference pretrained checkpoint via torchvision (best-effort, default: enabled)",
        )
        p.add_argument(
            "--no_download_model_params",
            dest="download_model_params",
            action="store_false",
            help="skip model params download step",
        )
    else:
        p.add_argument("--amp", action=bool_action, default=True, help="enable AMP autocast for --no_deepspeed training")
        p.add_argument("--tf32", action=bool_action, default=True, help="enable TF32 (recommended on A100/H100)")
        p.add_argument(
            "--use_checkpoint",
            action=bool_action,
            default=True,
            help="enable activation checkpointing inside Swin blocks",
        )
        p.add_argument(
            "--fallback_dummy",
            action=bool_action,
            default=True,
            help="if CIFAR-10 download fails, generate a tiny dummy dataset and still run training",
        )
        p.add_argument(
            "--download_model_params",
            action=bool_action,
            default=True,
            help="download a reference pretrained checkpoint via torchvision (best-effort)",
        )
    p.add_argument("--clean", action="store_true", help="remove existing work_dir before running")
    return p.parse_args()


def main() -> int:
    args = _get_args()

    root = _repo_root()
    os.chdir(root)

    # Default to GPU:0. To FORCE it even if CUDA_VISIBLE_DEVICES is already set:
    #   export RINGMOE_GPU_ID=0
    # To respect a scheduler-provided CUDA_VISIBLE_DEVICES:
    #   export RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES=1
    ringmoe_gpu_id = os.environ.get("RINGMOE_GPU_ID", "0")
    if not ringmoe_gpu_id.isdigit():
        print("[warn] RINGMOE_GPU_ID is not an integer:", ringmoe_gpu_id, "-> using 0")
        ringmoe_gpu_id = "0"
    respect_cvd = os.environ.get("RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES", "0").lower() in {"1", "true", "yes"}
    if not (respect_cvd and os.environ.get("CUDA_VISIBLE_DEVICES")):
        os.environ["CUDA_VISIBLE_DEVICES"] = ringmoe_gpu_id

    work_dir = Path(args.work_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    _print_header("[INFO] Environment")
    print("repo:", str(root))
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("work_dir:", str(work_dir))
    print("data_dir:", str(data_dir))
    print("RINGMOE_GPU_ID:", os.environ.get("RINGMOE_GPU_ID"))
    print("RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES:", os.environ.get("RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    _ensure_torch_cuda_or_die()

    # 1) Self-check (CUDA + model fwd/bwd + optional deepspeed kernel info)
    try:
        _run_step(
            "A100 selfcheck",
            [_python(), "-m", "pytorch_refactor.a100_selfcheck"],
            cwd=root,
            log_path=logs_dir / "01_selfcheck.log",
        )
    except StepFailure as e:
        _analyze_failure(e)
        return 10

    # 2) Prepare CIFAR-10 -> data.json
    data_json = data_dir / "data.json"
    if not data_json.exists():
        try:
            _run_step(
                "Download CIFAR-10 + export data.json",
                [
                    _python(),
                    "pytorch_refactor/prepare_cifar10.py",
                    "--out_dir",
                    str(data_dir),
                    "--split",
                    args.split,
                    "--limit",
                    str(args.limit),
                ],
                cwd=root,
                log_path=logs_dir / "02_prepare_cifar10.log",
            )
        except StepFailure as e:
            _analyze_failure(e)
            if not args.fallback_dummy:
                print(
                    "\nRe-run with `--fallback_dummy` (or omit `--no-fallback_dummy` / `--no_fallback_dummy`) "
                    "to continue offline using a tiny synthetic dataset."
                )
                return 20
            _print_header("[WARN] Falling back to dummy dataset (offline-safe)")
            data_json = Path(_write_dummy_dataset(work_dir))
            print("dummy data.json:", str(data_json))

    # 2b) Download a reference pretrained checkpoint (best-effort; not required for training)
    if getattr(args, "download_model_params", True):
        model_params_dir = work_dir / "model_params"
        model_params_path = model_params_dir / "torchvision_swin_v2_t.pt"
        if model_params_path.exists():
            print("[info] model params already exist:", str(model_params_path))
        else:
            try:
                _run_step(
                    "Download model parameters (torchvision SwinV2-T)",
                    [
                        _python(),
                        "pytorch_refactor/download_torchvision_swinv2_t.py",
                        "--out",
                        str(model_params_path),
                    ],
                    cwd=root,
                    log_path=logs_dir / "02b_download_model_params.log",
                )
            except StepFailure as e:
                _analyze_failure(e)
                print("[warn] model params download failed; continuing without it.")

    # 3) dry-run
    try:
        _run_step(
            "Dry-run (data + shapes)",
            [
                _python(),
                "train_a100.py",
                "--data_path",
                str(data_json),
                "--no_deepspeed",
                "--dry_run",
                "--num_workers",
                str(args.num_workers),
                "--input_size",
                str(args.input_size),
                "--mask_ratio",
                str(args.mask_ratio),
            ]
            + (["--amp"] if args.amp else [])
            + (["--tf32"] if args.tf32 else [])
            + (["--use_checkpoint"] if args.use_checkpoint else []),
            cwd=root,
            log_path=logs_dir / "03_dry_run.log",
        )
    except StepFailure as e:
        _analyze_failure(e)
        return 30

    # 4) Train 1 epoch (default)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_step(
            "Train (single GPU, no DeepSpeed)",
            [
                _python(),
                "train_a100.py",
                "--data_path",
                str(data_json),
                "--no_deepspeed",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--log_every",
                "10",
                "--input_size",
                str(args.input_size),
                "--mask_ratio",
                str(args.mask_ratio),
                "--output_dir",
                str(ckpt_dir),
            ]
            + (["--amp"] if args.amp else [])
            + (["--tf32"] if args.tf32 else [])
            + (["--use_checkpoint"] if args.use_checkpoint else []),
            cwd=root,
            log_path=logs_dir / "04_train.log",
        )
    except StepFailure as e:
        _analyze_failure(e)
        return 40

    _print_header("[SUCCESS] Training completed")
    print("checkpoints:", str(ckpt_dir))
    print("logs:", str(logs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
