<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=180&section=header&text=RingMoE&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=%E9%81%A5%E6%84%9F%E5%9B%BE%E5%83%8F%20MoE%20%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A1%86%E6%9E%B6&descAlignY=55&descAlign=50" width="100%" />
</p>

| 类别 | 技术栈 |
|------|--------|
| **框架** | MindSpore 2.1, PyTorch 2.1 |
| **并行策略** | DeepSpeed, HCCL, Expert/Data/Model/Pipeline |
| **硬件** | Ascend 910B, NVIDIA A100/H100 |
| **语言** | Python 3.7+ |

## 📋 简介

面向遥感影像的 MoE（混合专家）预训练框架，支持 Ascend 910B（MindSpore）和 NVIDIA GPU（PyTorch/DeepSpeed）双平台，实现数据、模型、专家、流水线四种并行策略。

## 🚀 快速开始

```bash
# MindSpore (Ascend 910B) 8卡分布式预训练
bash scripts/pretrain_distribute.sh ./rank_table_8pcs.json ./config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml

# PyTorch (NVIDIA A100)
cd pytorch_refactor
pip install -r requirements.txt
python train.py --config ../config/simmim_pcl/...
```

## ✨ 功能特性

- **多策略并行**：数据并行、模型并行、专家并行、流水线并行
- **丰富架构**：ViT / Swin / SwinV2 骨干 + MAE / SimMIM / RingMo 预训练任务
- **MoE 变体**：多模态路由、单专家路由、可配置专家数
- **跨平台**：MindSpore（Ascend）+ PyTorch/DeepSpeed（NVIDIA）
- **下游任务**：分类、分割、检测（mmdetection / mmpretrain / mmseg）

## 🏗️ 项目结构

```
RingMoE/
├── ringmoe_framework/        # MindSpore 核心框架
│   ├── arch/                 # 预训练任务（MAE, SimMIM, MoE）
│   ├── datasets/             # 数据加载与预处理
│   ├── models/               # 骨干网络、MoE 层、核心算子
│   ├── trainer/              # 训练流程（loss scale, grad clip, EMA）
│   ├── optim/                # 优化器（AdamW）
│   ├── lr/                   # 学习率调度（warmup, cosine）
│   └── tools/                # 工具（checkpoint, HCCL）
├── pytorch_refactor/         # PyTorch/DeepSpeed 移植
│   ├── train.py              # 训练入口
│   ├── model.py              # 模型定义
│   └── dataset.py            # 数据集
├── config/                   # YAML 配置
├── scripts/                  # 分布式启动脚本
├── pretrain.py               # 预训练入口
├── finetune.py               # 微调入口
└── eval.py                   # 评估入口
```

## ❓ 常见问题

| 问题 | 回答 |
|------|------|
| **可以用单卡训练吗？** | 可以，但推荐 8 卡以上以平衡专家分布 |
| **如何添加新模型？** | 在 `models/` 添加骨干网络，注册配置并创建预训练任务 |
| **数据集格式？** | Ascend 用 MindRecord，移植版用标准 PyTorch Dataset |

## 🔗 相关项目

- [Router-MVP](/WJH-makers/router-mvp) — 多智能体路由，互补的分布式系统研究
- [C++ Compiler](/WJH-makers/compiler-C-PLUS-PLUS) — 编译器 IR 与计算图概念

## 🎓 课程背景

武汉大学计算机学院 · 遥感基础模型研究方向。

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=100&section=footer" width="100%" />
</p>
