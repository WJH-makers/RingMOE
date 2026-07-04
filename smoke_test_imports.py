"""Minimal smoke test for RingMoE imports.

This tries to import the repo's top-level entrypoints and core packages.
It does NOT start training.

Usage:
  python smoke_test_imports.py

Exit code:
  0 if all selected imports succeed, else 1.
"""

from __future__ import annotations

import importlib
import sys


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


BASE_TARGETS = [
    "register",
    "register.config",
]

MINDSPORE_TARGETS = [
    "ringmoe_framework",
    "ringmoe_framework.arch",
    "ringmoe_framework.datasets",
    "ringmoe_framework.loss",
    "ringmoe_framework.lr",
    "ringmoe_framework.models",
    "ringmoe_framework.optim",
    "ringmoe_framework.parallel_config",
    "ringmoe_framework.tools.helper",
    "ringmoe_framework.tools.load_ckpt",
    "ringmoe_framework.trainer",
]

PYTORCH_A100_TARGETS = [
    "pytorch_refactor.dataset",
    "pytorch_refactor.model",
    "pytorch_refactor.train",
]


def main() -> int:
    targets = list(BASE_TARGETS)
    if _can_import("mindspore"):
        targets += MINDSPORE_TARGETS
    else:
        targets += PYTORCH_A100_TARGETS

    failed = []
    for name in targets:
        try:
            importlib.import_module(name)
            print(f"OK  {name}")
        except Exception as e:
            print(f"ERR {name}: {type(e).__name__}: {e}")
            failed.append(name)

    if failed:
        print("\nFAILED:")
        for n in failed:
            print("-", n)
        if not _can_import("mindspore"):
            print("\nHint: MindSpore is not installed; for NVIDIA A100 use `pytorch_refactor/` (see RUNNING_LINUX_A100.md).")
        return 1
    print("\nAll imports OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

