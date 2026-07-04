# 在 Linux + NVIDIA A100 上运行 RingMoE（PyTorch/DeepSpeed 重构版）

本仓库原始实现面向 **MindSpore + Ascend 910B**。若你希望在 **Linux + NVIDIA A100** 上训练/验证，请使用本仓库内的 **PyTorch/DeepSpeed 重构实现**：`pytorch_refactor/`。

建议入口：`train_a100.py`（仓库根目录，便于直接 `deepspeed train_a100.py ...`）；等价入口：`pytorch_refactor/train.py`。

---

## 1. 硬件与软件要求

- Linux x86_64
- NVIDIA A100（建议 40GB/80GB；多卡训练用 NVLink/NVSwitch 更佳）
- NVIDIA Driver（需支持你安装的 CUDA 版本）
- Python 3.8+（建议 3.10/3.11）
- PyTorch 2.x（CUDA 11.8 或 12.x 版本）
- DeepSpeed（需要能编译 CUDA 扩展；MoE 训练建议启用 DS MoE kernels）

---

## 2. 安装（建议 venv/conda）

示例（venv）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

安装 PyTorch（请按你的 CUDA 版本选择对应的官方安装命令）。

安装本重构版依赖：

```bash
pip install -r pytorch_refactor/requirements.txt
```

如果你需要 **DeepSpeed MoE 内核**（推荐），通常需要从源码/带编译选项安装 DeepSpeed（不同环境略有差异）。常见做法：

```bash
# 仅示例：确保能编译 CUDA 扩展（需要 gcc/g++、CUDA Toolkit、ninja 等）
DS_BUILD_OPS=1 pip install --no-cache-dir deepspeed
```

---

## 3. 环境自检（A100 + bf16 + MoE）

```bash
python -m pytorch_refactor.a100_selfcheck

## 或者只做基础 CUDA/BF16 检查：
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

DeepSpeed MoE kernels 自检（可选，但 MoE 训练强烈建议）：

```bash
python - <<'PY'
try:
    from deepspeed.ops.op_builder import MoEBuilder
    print("MoE kernels compatible:", MoEBuilder().is_compatible())
except Exception as e:
    print("DeepSpeed/MoE check failed:", e)
PY
```

若 MoE kernels 不可用：
- 先用 `--disable_moe` 跑通训练链路；或
- 重新安装/编译带 CUDA 扩展的 DeepSpeed。

---

## 4. 数据准备（JSON：图片路径列表）

`pytorch_refactor/train.py` 读取一个 JSON 文件，内容是图片路径列表，例如：

```json
["/data/imgs/0001.jpg", "/data/imgs/0002.jpg"]
```

### 4.1 多模态（可选）

当你需要按“每条样本包含多张图（多模态）”训练时，JSON 的每个元素写成一个路径列表，并在训练时传 `--modal_num` + `--modal_in_chans`：

```json
[
  ["/data/opt/0001.jpg", "/data/sar/0001.npy", "/data/ms/0001.jpg", "/data/hsi/0001.npy"],
  ["/data/opt/0002.jpg", "/data/sar/0002.npy", "/data/ms/0002.jpg", "/data/hsi/0002.npy"]
]
```

说明：
- 当前实现会把多模态在 encoder 内部 **沿 batch 维拼接** 走同一套 SwinV2 backbone（等价于每 step 的有效 batch 乘以 `modal_num`），并在 decoder/loss 处按模态拆分对齐。
- 对于 **非 3 通道** 的模态：请使用 `.npy/.npz`（或 `.tif/.tiff` 并安装 `tifffile`）存储多通道数组；并通过 `--modal_in_chans` 声明每个模态的通道数。

示例（4 模态，通道数 3/8/3/4，对应原始 RingMoE 多模态常见设置）：

```bash
python pytorch_refactor/train.py \
  --data_path data_mm.json \
  --modal_num 4 \
  --modal_in_chans 3,8,3,4 \
  --no_deepspeed \
  --dry_run
```

从图片目录生成 `data.json`（示例：递归收集 jpg/png）：

```bash
python - <<'PY'
import json, glob
paths = []
paths += glob.glob("/data/imgs/**/*.jpg", recursive=True)
paths += glob.glob("/data/imgs/**/*.png", recursive=True)
paths = sorted(paths)
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(paths, f)
print("wrote", len(paths), "paths to data.json")
PY
```

### 4.2 一键下载示例数据集（CIFAR-10）并生成 `data.json`

若你只想先**从零下载数据并把训练链路跑通**（不需要标注），可用 CIFAR-10：

```bash
# 生成: data/cifar10/images/*.png + data/cifar10/data.json
python pytorch_refactor/prepare_cifar10.py --out_dir data/cifar10 --split train
```

---

## 5. 先跑冒烟与 dry-run（强烈建议）

### 5.1 CPU 冒烟（不依赖 DeepSpeed）

```bash
python pytorch_refactor/smoke_test_pytorch_refactor.py
```

多模态冒烟（不依赖 DeepSpeed）：

```bash
python pytorch_refactor/smoke_test_pytorch_refactor_multimodal.py
```

### 5.2 单卡 dry-run（只校验数据与张量形状）

```bash
python pytorch_refactor/train.py \
  --data_path data.json \
  --no_deepspeed \
  --dry_run
```

### 5.3 一键端到端（下载 CIFAR-10 -> 自检 -> 单卡训练）

适合在新机器上做“必跑通”验证（默认只导出 256 张图，训练 1 个 epoch，避免耗时过长）：

```bash
python pytorch_refactor/quickstart_single_a100_cifar10.py
```

如果你希望把“venv 创建 + torch/torchvision 安装 + 数据集下载 + 模型参数下载 + 训练”也一键自动化：

```bash
bash one_click_a100_single.sh
```

也可以用 bash 包装脚本（会创建 `.venv` 并安装最小依赖；但 **torch/torchvision 仍需你按官方方式装 CUDA 版**）：

```bash
bash pytorch_refactor/quickstart_single_a100_cifar10.sh
```

若集群节点无外网可加 `--fallback_dummy` 走离线兜底数据（仍会完成训练闭环）：

```bash
python pytorch_refactor/quickstart_single_a100_cifar10.py --fallback_dummy
```

---

## 6. A100 多卡训练（DeepSpeed）

### 6.1 单机 8 卡（bf16，推荐）

```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --data_path data.json \
  --epochs 100 \
  --micro_batch 2 \
  --tf32 \
  --use_checkpoint \
  --output_dir checkpoints \
  --save_every 1
```

说明：
- 默认 `ds_config.json` 开启 `bf16`；A100 上建议优先使用 bf16。
- 显存不足先调小 `--micro_batch`，再用 `ds_config.json` 里的 `gradient_accumulation_steps` 拉回等效 batch。
- `--use_checkpoint` 会启用 Swin block 内的梯度检查点以省显存（训练更慢但更稳）。
- MoE 的辅助损失会按 `--aux_loss_factor`（默认 `0.001`）加到总 loss 上。

### 6.2 bf16 失败时强制 fp16

```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --data_path data.json \
  --force_fp16
```

### 6.3 MoE 内核缺失时关闭 MoE（先跑通）

```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --data_path data.json \
  --disable_moe
```

### 6.4 多模态训练示例（4 模态 3/8/3/4）

```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --data_path data_mm.json \
  --modal_num 4 \
  --modal_in_chans 3,8,3,4 \
  --epochs 100 \
  --micro_batch 1 \
  --tf32 \
  --use_checkpoint \
  --output_dir checkpoints_mm
```

---

## 7. 恢复训练（Resume）

训练保存的 checkpoint 默认在 `--output_dir`（例如 `checkpoints/`），tag 形如 `epoch_0`、`epoch_1`。

```bash
deepspeed --num_gpus 8 pytorch_refactor/train.py \
  --deepspeed_config pytorch_refactor/ds_config.json \
  --data_path data.json \
  --resume_from checkpoints \
  --resume_tag epoch_0
```

---

## 8. 导出与推理（可选）

### 8.1 从 DeepSpeed ZeRO checkpoint 导出单文件 `.pt`

```bash
python pytorch_refactor/export_pt.py \
  --ds_ckpt checkpoints \
  --tag epoch_0 \
  --out model.pt \
  --moe_experts 8
```

### 8.2 重建图像（SimMIM recon）

```bash
python pytorch_refactor/infer.py \
  --image /path/to/a.jpg \
  --ckpt model.pt \
  --out recon.png \
  --moe_experts 8
```

---

## 9. 常见问题（A100）

- **DeepSpeed 安装后仍然没有 MoE kernels**：通常是 CUDA 扩展没编译出来（编译工具链/CUDA Toolkit/ninja 缺失或版本不匹配）。先用 `--disable_moe` 跑通，再按环境重装 DeepSpeed。
- **NCCL 通信问题**：可先设 `NCCL_DEBUG=INFO` 排查；多机环境还需检查网卡/IB 配置。
- **OOM**：优先降低 `--micro_batch`，再开 `--use_checkpoint`，再调整输入分辨率/模型规模/ZeRO stage。
