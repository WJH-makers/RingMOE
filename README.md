# RingMoE

RingMoE 是一个基于 MindSpore 框架的大规模混合专家（MoE）模型预训练和下游任务评估代码库。它专为 Ascend 910B 集群设计，支持高效的分布式训练。

## 主要特性

*   **大规模并行**: 支持数据并行、模型并行、专家并行（Expert Parallel）和流水线并行。
*   **多模型支持**: 内置 MAE, SimMIM, RingMo 等多种架构及其 MoE 变体。
*   **高性能**: 针对 Ascend 910B 硬件进行深度优化，支持图模式和重计算。
*   **开箱即用**: 提供完整的预训练、微调和评估脚本。

## 快速开始

详细的新手入门教程请查阅 [TUTORIAL.md](./TUTORIAL.md)。

## Linux + NVIDIA A100（PyTorch/DeepSpeed 重构版）

本仓库包含一份面向 **NVIDIA A100/H100** 的 **PyTorch/DeepSpeed** 重构实现，位于 `pytorch_refactor/`。

- 运行指南：`RUNNING_LINUX_A100.md`
- 代码入口：`pytorch_refactor/train.py`

### 环境要求

*   **硬件**: Ascend 910B
*   **框架**: MindSpore 2.1.0+
*   **依赖**: CANN 6.3.RC2, Python 3.7+

### 运行预训练

```bash
cd scripts
# 使用 8 卡分布式训练示例
bash pretrain_distribute.sh ./rank_table_8pcs.json ./config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml
```

## 目录结构

*   `config/`: 模型与训练配置文件
*   `ringmoe_framework/`: 框架核心源码（包含 arch, datasets, models, loss, optim 等）
*   `scripts/`: 分布式启动脚本
*   `pretrain.py` / `finetune.py`: 训练入口文件

## 推荐的环境

- `requirements-a100.txt`: stable PyTorch/cu118 stack (torch 2.1.0 + torchvision 0.16.0, numpy 1.26.4, timm/einops, scipy/pillow). Use this for `pytorch_refactor/` and `train_a100.py`.
- `requirements-isaid.txt`: iSAID/MMDet stack matching the one-click scripts (torch 2.1.0 + cu118, mmcv 2.1.0, mmdet 3.3.0, mmengine 0.10.7, numpy 1.26.4, opencv-python-headless 4.10.0.84).

在需要 cu121 而不是 cu118 的系统上，需一致地更换 extra-index URL 和 `+cu118` 后缀。

## 许可证

请查看项目内的 LICENSE 文件。
