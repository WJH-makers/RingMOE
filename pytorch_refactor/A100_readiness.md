# RingMoE PyTorch Refactor — A100 适配指南

本文说明如何在 NVIDIA A100 上运行 PyTorch/DeepSpeed 重构版，概述 `train.py` 的关键改动，并给出精度与 MoE 的验证与调整方法。

## `train.py` 的新变化
- DeepSpeed 配置自动加载 `pytorch_refactor/ds_config.json`（或使用 `--deepspeed_config`）。
- BFloat16 兜底：若设备/torch 不支持 bf16 或传入 `--force_fp16`，则切换到 fp16 并关闭 bf16。
- `--micro_batch` 覆盖 `train_micro_batch_size_per_gpu` 以便快速调节显存；该值会应用到 DataLoader。
- DeepSpeed 的 DataLoader 在分布式初始化之后构建，`world_size > 1` 时使用 `DistributedSampler`。
- 当关闭 DeepSpeed 时自动关闭 MoE（MoE 需要 DS 内核）。
- Checkpoint 使用固定标签 `epoch_<N>`，便于恢复。

## A100 快速环境自检
```bash
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'cpu')
print('bf16_supported', torch.cuda.is_bf16_supported())
PY

python - <<'PY'
try:
    import deepspeed
    from deepspeed.ops.op_builder import MoEBuilder
    print('deepspeed', deepspeed.__version__)
    print('MoE kernels available?', MoEBuilder().is_compatible())
except Exception as e:
    print('deepspeed check failed:', e)
PY
```

若 bf16 显示 `False`，计划加上 `--force_fp16`。若 MoE 内核不可用，可加 `--disable_moe`，或重新安装带 CUDA 内核的 DeepSpeed。

## 默认 DeepSpeed 配置（`pytorch_refactor/ds_config.json`）
- `bf16.enabled: true`（A100 首选）
- `fp16.enabled: false`
- ZeRO stage 2，开启通信重叠
- `train_micro_batch_size_per_gpu: 2`（可用 `--micro_batch` 覆盖）

## 运行示例
### 1) 单卡冒烟（无 DeepSpeed，MoE 关闭）
用于验证数据路径和张量形状。
```bash
python pytorch_refactor/train.py \
  --data_path <json> \
  --no_deepspeed \
  --dry_run
```

### 2) 多卡 DeepSpeed + bf16（A100）
```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --data_path <json> \
  --deepspeed_config pytorch_refactor/ds_config.json
```

### 3) 强制 fp16（bf16 不支持时）
```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --data_path <json> \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --force_fp16
```

### 4) 调整 micro-batch 以适配显存
```bash
# 例：降到 1
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --data_path <json> \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --micro_batch 1
```

### 5) MoE 内核缺失或做 baseline 时关闭 MoE
```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --data_path <json> \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --disable_moe
```

## 恢复与检查点
- 每 `--save_every` 个 epoch 保存一次，标签 `epoch_<N>`，目录 `--output_dir`（默认 `checkpoints`）。
- 恢复示例：
```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --data_path <json> \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --resume_from checkpoints \
  --resume_tag epoch_0
```

## 常见 A100 问题与解决
- **bf16 不支持**：torch/CUDA 版本过旧。用 `--force_fp16`，或安装支持 sm80 的 cu118/cu121 版 torch。
- **缺少 MoE 内核**：用 `DS_BUILD_OPS=1` 重新安装 DeepSpeed，或先加 `--disable_moe`。
- **显存 OOM**：降低 `--micro_batch`，提高 ds_config 的 `gradient_accumulation_steps`，或降低输入分辨率。
- **NCCL 问题**：设 `NCCL_DEBUG=INFO`；部分网络环境可试 `NCCL_IB_DISABLE=1`。

## 建议的操作顺序
- 先跑单卡 `--dry_run`，确认数据与形状。
- 跑 DeepSpeed bf16；若 bf16 报错，改用 `--force_fp16`。
- 若 MoE 构建失败，先用 `--disable_moe` 跑通，再计划重装带 MoE 内核的 DeepSpeed。
