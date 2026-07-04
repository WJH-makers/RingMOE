<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=180&section=header&text=RingMoE&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Large-Scale%20MoE%20Pre-training%20for%20Remote%20Sensing&descAlignY=55&descAlign=50" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MindSpore-2.1.0-blue?style=flat-square&logo=mindspore" />
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/DeepSpeed-Available-000?style=flat-square" />
  <img src="https://img.shields.io/badge/Ascend_910B-Supported-00A1E9?style=flat-square" />
  <img src="https://img.shields.io/badge/NVIDIA_A100-Refactored-76B900?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.7%2B-3776AB?style=flat-square&logo=python" />
</p>

## 📋 Overview

RingMoE is a **Mixture-of-Experts (MoE)** pre-training framework for **remote sensing imagery**, built for Ascend 910B (MindSpore) with a PyTorch/DeepSpeed port for NVIDIA GPUs. Supports 4 paradigms of parallelism — data, model, expert (MoE), and pipeline — with MAE, SimMIM, and RingMo architectures.

> **Why RingMoE?** Remote sensing images demand high-resolution, multi-scale feature extraction that dense models cannot afford. MoE activates only relevant experts per token, scaling model capacity without proportional compute.

## 🚀 Quick Start

### MindSpore (Ascend 910B)

```bash
# 8-card distributed pre-training
bash scripts/pretrain_distribute.sh ./rank_table_8pcs.json ./config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml
```

### PyTorch (NVIDIA A100)

```bash
cd pytorch_refactor
pip install -r requirements.txt
python train.py --config ../config/simmim_pcl/...
```

## ✨ Key Features

- **Multi-Paradigm Parallelism**: Data parallel, model parallel, expert parallel (MoE), pipeline stages
- **Rich Architectures**: ViT, Swin, SwinV2 backbones with MAE/SimMIM/RingMo pre-training tasks
- **MoE Variants**: Multi-modal routing, single-expert routing, configurable expert counts
- **Cross-Platform**: MindSpore (Ascend 910B) + PyTorch/DeepSpeed (NVIDIA A100/H100) refactor
- **Downstream Tasks**: Classification, segmentation, detection via mmdetection/mmpretrain/mmseg

## 🏗️ Architecture

```
RingMoE/
├── ringmoe_framework/          # Core framework (MindSpore)
│   ├── arch/                   # Pre-training tasks (MAE, SimMIM, RingMo, MoE variants)
│   ├── datasets/               # Data loaders, mask strategies, MindRecord tools
│   ├── models/                 # Backbones (ViT/Swin/SwinV2), MoE layers, core ops
│   ├── trainer/                # TrainOneStep (loss scale, clip grad, EMA)
│   ├── optim/                  # Optimizers (AdamW, AdamWOP)
│   ├── lr/                     # LR schedules (warmup, cosine, multistep)
│   ├── tools/                  # Context init, checkpoint utils, HCCL tools
│   └── monitors/               # Callbacks & monitoring
├── pytorch_refactor/           # PyTorch/DeepSpeed A100 port
│   ├── train.py                # Training entry point
│   ├── model.py                # SwinTransformerV2MoE, SimMIM, MultiModal models
│   └── dataset.py              # RingMoEDataset
├── config/                     # YAML configuration system
├── scripts/                    # Distributed launch scripts
├── pretrain.py                 # MindSpore pre-training entry
├── finetune.py                 # MindSpore fine-tuning entry
└── eval.py                     # MindSpore evaluation entry
```

## 📦 Requirements

| Environment | Framework | Hardware |
|------------|-----------|----------|
| Production | MindSpore 2.1.0+ | Ascend 910B |
| Refactor | PyTorch 2.1.0 + CUDA 11.8 | NVIDIA A100/H100 |

## 🎓 Academic Context

This project was developed as part of research at **Wuhan University**, School of Computer Science, focusing on remote sensing foundation models.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=100&section=footer" width="100%" />
</p>
