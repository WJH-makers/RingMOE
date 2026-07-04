"""Convenience entrypoint for the NVIDIA A100/H100 PyTorch/DeepSpeed refactor.

This repo's original entrypoints (pretrain/finetune/eval) target MindSpore + Ascend 910B.
For Linux + NVIDIA A100/H100, use the refactored implementation under `pytorch_refactor/`.
See: RUNNING_LINUX_A100.md
"""

from pytorch_refactor.train import main


if __name__ == "__main__":
    main()

