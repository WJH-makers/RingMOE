"""iSAID instance segmentation training config (Mask R-CNN) for MMDetection.

Notes:
- iSAID annotations are MS COCO format.
- Academic use only (per dataset license).
- This config is a baseline that should run on a single A100; adjust batch size, workers, and schedule as needed.

Expected dataset layout (matches the official iSAID release):
  $ISAID_ROOT/
    train/
      images/*.png                 (e.g. P0000.png)
      Annotations/iSAID_train.json
    val/
      images/*.png
      Annotations/iSAID_val.json

Run (recommended via openmim):
  ISAID_ROOT=/data/iSAID mim train mmdet downstream/mmdet/isaid_mask_rcnn_r50_fpn_1x.py --work-dir work_dirs/isaid
"""

# MMDetection 3.x supports package-style base configs via `mmdet::`.
_base_ = "mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# IMPORTANT:
# MMEngine's config system has two modes: non-lazy and "lazy_import".
# If this file contains `import ...` statements, MMEngine may treat it as
# "lazy_import" and reject `_base_ = ...` inheritance with:
#   ConfigParsingError: ... either "lazy_import" or non-"lazy_import" ...
# To keep this config compatible across MMDet/MMEngine versions, avoid `import`
# statements and use `__import__` instead.
_os = __import__("os")
_glob = __import__("glob").glob

ISAID_ROOT = _os.environ.get("ISAID_ROOT", "/path/to/iSAID").rstrip("/") + "/"

def _rel_to_root(path: str) -> str:
    """Return path relative to ISAID_ROOT if it's an absolute path under ISAID_ROOT."""
    if not isinstance(path, str) or not path:
        return path
    if not _os.path.isabs(path):
        return path
    root_abs = _os.path.abspath(ISAID_ROOT)
    path_abs = _os.path.abspath(path)
    try:
        if _os.path.commonpath([root_abs, path_abs]) == root_abs:
            return _os.path.relpath(path_abs, root_abs)
    except Exception:
        # Be conservative; keep the original path if anything looks off.
        return path
    return path


def _abs_from_root(path: str) -> str:
    """Return an absolute path for ann_file (keeps absolute paths as-is)."""
    if _os.path.isabs(path):
        return path
    return _os.path.join(ISAID_ROOT, path)


def _pick_ann(rel_pattern: str, fallback: str) -> str:
    # Prefer the largest matching file (e.g. iSAID_train_YYYYmmdd_HHMMSS.json),
    # otherwise fall back to iSAID_train.json / iSAID_val.json.
    matches = _glob(_os.path.join(ISAID_ROOT, rel_pattern))
    if not matches:
        return fallback
    matches = sorted(matches, key=lambda p: _os.path.getsize(p), reverse=True)
    # Convert to path relative to data_root (ISAID_ROOT)
    return _os.path.relpath(matches[0], ISAID_ROOT)


train_ann_file = _os.environ.get(
    "ISAID_TRAIN_ANN",
    _pick_ann("train/Annotations/iSAID_train_*.json", "train/Annotations/iSAID_train.json"),
)
val_ann_file = _os.environ.get(
    "ISAID_VAL_ANN",
    _pick_ann("val/Annotations/iSAID_val_*.json", "val/Annotations/iSAID_val.json"),
)
train_ann_file = _rel_to_root(train_ann_file)
val_ann_file = _rel_to_root(val_ann_file)

# iSAID categories must match the annotation JSON `categories[].name` values (case-sensitive).
# Hardcode to avoid parsing huge COCO JSON files at config load time.
classes = (
    "storage_tank",
    "Large_Vehicle",
    "Small_Vehicle",
    "plane",
    "ship",
    "Swimming_pool",
    "Harbor",
    "tennis_court",
    "Ground_Track_Field",
    "Soccer_ball_field",
    "baseball_diamond",
    "Bridge",
    "basketball_court",
    "Roundabout",
    "Helicopter",
)
metainfo = dict(classes=classes)


def _env_flag(name: str, default: str = "1") -> bool:
    return _os.environ.get(name, default).strip() in {"1", "true", "True", "yes", "YES"}

# Effective batch/accumulation for lr scaling
_train_batch_size = int(_os.environ.get("ISAID_BATCH_SIZE", "1"))
_accum = int(_os.environ.get("ISAID_ACCUM", _os.environ.get("ISAID_GRAD_ACCUM", "1")))
_accum = max(1, _accum)
_effective_bs = max(1, _train_batch_size * _accum)
# Base mask-rcnn lr is 0.02 for 16 images; scale linearly per image.
_lr_per_img = float(_os.environ.get("ISAID_LR_PER_IMG", str(0.02 / 16.0)))
_scaled_lr = _lr_per_img * _effective_bs

# Dataloader tuning knobs
_num_workers = int(_os.environ.get("ISAID_NUM_WORKERS", "2"))
# When using multiple workers, prefetching helps hide PNG decode/resize latency.
# PyTorch default is 2; you can raise it if you have RAM/CPU to spare.
_prefetch_factor = int(_os.environ.get("ISAID_PREFETCH_FACTOR", "2"))
_prefetch_factor = max(1, _prefetch_factor)

# Override pipelines to use a smaller resize for lower memory.
_train_resize = tuple(int(x) for x in _os.environ.get("ISAID_RESIZE", "1024,640").split(","))  # W,H
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=_train_resize, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
_test_resize = tuple(int(x) for x in _os.environ.get("ISAID_TEST_RESIZE", "1024,640").split(","))
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=_test_resize, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=int(_os.environ.get("ISAID_BATCH_SIZE", "1")),
    num_workers=_num_workers,
    persistent_workers=_env_flag("ISAID_PERSISTENT_WORKERS", "0"),
    pin_memory=_env_flag("ISAID_PIN_MEMORY", "0"),
    # Only valid when num_workers > 0
    prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
    dataset=dict(
        data_root=ISAID_ROOT,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img="train/images/"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=_num_workers,
    persistent_workers=_env_flag("ISAID_PERSISTENT_WORKERS", "0"),
    pin_memory=_env_flag("ISAID_PIN_MEMORY", "0"),
    prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
    dataset=dict(
        data_root=ISAID_ROOT,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img="val/images/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=_abs_from_root(val_ann_file))
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Model head: set num_classes
# ---------------------------------------------------------------------------

# Gradient checkpointing (saves memory, slows training). Default off for speed.
# Enable via:
#   export ISAID_WITH_CP=1
_with_cp = _env_flag("ISAID_WITH_CP", "0")

model = dict(
    backbone=dict(with_cp=_with_cp),
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)),
    )
)

# ---------------------------------------------------------------------------
# Speed/memory knobs (A100)
# ---------------------------------------------------------------------------

# cuDNN autotune for fixed-size training (typically a win on A100).
env_cfg = dict(cudnn_benchmark=True)

optim_wrapper = dict(
    type="AmpOptimWrapper" if _env_flag("ISAID_AMP", "1") else "OptimWrapper",
    optimizer=dict(lr=_scaled_lr),
    accumulative_counts=_accum,
    loss_scale="dynamic" if _env_flag("ISAID_AMP", "1") else None,
)

# ---------------------------------------------------------------------------
# Optional init checkpoint
# ---------------------------------------------------------------------------

# You can set:
#   export ISAID_LOAD_FROM=/path/to/mmdet_checkpoint.pth
load_from = _os.environ.get("ISAID_LOAD_FROM", None)
