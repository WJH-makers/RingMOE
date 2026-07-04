# RingMoE 新手入门教程

欢迎使用 RingMoE！本教程将帮助你快速上手 RingMoE 框架，进行大规模模型的预训练和微调。

## 1. 项目简介

RingMoE 是一个基于 MindSpore 框架的大规模混合专家（MoE）模型预训练和下游任务评估代码库。它支持在 Ascend 910B 集群上进行高效的分布式训练。

## 2. 环境准备

在开始之前，请确保你的环境满足以下要求：

*   **硬件**: Ascend 910B 集群 (预训练 14.7B 参数模型至少需要 64 个节点)
*   **操作系统**: Linux (推荐)
*   **软件依赖**:
    *   **MindSpore 2.1.0**: [安装指南](https://www.mindspore.cn/install)
    *   **Python 3.7**: 推荐使用 Anaconda 管理环境
    *   **CANN 6.3.RC2**: Ascend 驱动和开发套件
    *   **gdal**: 地理空间数据处理库

## 3. 项目结构说明

了解项目结构有助于你更好地定位文件：

```
RingMoE/
├── config/                 # 配置文件目录
│   ├── base/               # 基础配置 (数据集, 模型, 运行器等)
│   │   ├── datasets/       # 数据集默认配置
│   │   ├── models/         # 模型默认配置
│   │   └── ...
│   └── simmim_pcl/         # 具体任务的组合配置 (如预训练配置)
├── ringmoe_framework/      # 核心框架代码
│   ├── arch/               # 模型架构定义
│   ├── datasets/           # 数据集加载与处理
│   ├── models/             # 模型组件
│   ├── parallel_config.py  # 并行策略配置定义
│   └── ...
├── scripts/                # 启动脚本
│   ├── pretrain_distribute.sh  # 分布式预训练脚本
│   ├── finetune_distribute.sh  # 分布式微调脚本
│   └── eval_distribute.sh      # 分布式评估脚本
├── pretrain.py             # 预训练入口文件
├── finetune.py             # 微调入口文件
├── rank_table_*.json       # 分布式训练的通信配置文件示例
└── README.md               # 项目说明文档
```

## 4. 数据准备

RingMoE 使用 **RingMOSS** 数据集。

1.  **获取数据**: 请访问 [RingMoEDatasets](https://github.com/HanboBizl/RingMoEDatasets) 获取数据。
2.  **数据格式**: 为了加速加载，数据需要转换为 **MindRecord** 格式。
    *   参考 `.ringmoe_framework/datasets/pretrain_dataset.py` 了解数据加载逻辑。
    *   预训练数据通常需要裁剪为 192x192 大小，参考 `.ringmoe_framework/datasets/image_cuts/`。
    *   MindSpore 官方文档：[如何转换 MindRecord](https://www.mindspore.cn/docs/zh-CN/r2.4.10/api_python/mindspore.mindrecord.html)

3.  **配置数据集**:
    修改 `config/base/datasets/pretrain_dataset.yaml` 或在你的主配置文件中覆盖以下字段：
    ```yaml
    pretrain_dataset:
      data_type: "mindrecord"
      data_path: "/path/to/your/mindrecord/files"  # MindRecord 文件路径
      image_ids: "/path/to/image_ids.txt"           # 图片 ID 列表文件
      num_workers: 8                                # 数据加载线程数
      crop_min: 0.67                                # 随机裁剪最小比例
      # ... 其他参数
    ```

## 5. 快速开始：预训练 (Pre-training)

预训练通常在多卡或多机环境下进行。我们提供了 `scripts/pretrain_distribute.sh` 脚本来简化启动过程。

### 步骤

1.  **准备 Rank Table 文件**:
    Ascend 平台分布式训练需要一个 JSON 格式的 rank table 文件来描述集群拓扑。
    项目根目录下提供了示例：
    *   `rank_table_2pcs.json` (2卡)
    *   `rank_table_8pcs.json` (8卡)
    *   `rank_table_16pcs.json` (16卡)
    *   *注意：你需要根据实际的服务器 IP 和设备 ID 修改这些文件。* 详情参考 [MindSpore Rank Table 文档](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/rank_table.html#%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1)。

2.  **选择配置文件**:
    配置文件位于 `config/` 目录下。例如：
    `config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml`

3.  **启动训练**:
    在终端中运行以下命令：

    ```bash
    cd scripts
    bash pretrain_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
    ```

    **示例 (假设在根目录下)**:
    ```bash
    bash scripts/pretrain_distribute.sh ./rank_table_8pcs.json ./config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml
    ```

### 脚本说明
该脚本会自动：
*   设置环境变量 (如 `RANK_SIZE`, `RANK_TABLE_FILE`)。
*   为每个设备创建一个工作目录 (`pretrain_parallel0`, `pretrain_parallel1`, ...)。
*   将代码和配置复制到工作目录。
*   后台启动 `pretrain.py`。
*   日志会保存在各工作目录下的 `pretrain_log` 文件中。

你可以通过 `tail -f pretrain_parallel0/pretrain_log` 查看 0 号卡的训练日志。

## 6. 快速开始：微调 (Fine-tuning)

微调流程与预训练类似，使用 `finetune.py` 和对应的脚本。

1.  **准备配置文件**: 确保你有微调任务的 yaml 配置文件。
2.  **启动微调**:

    ```bash
    bash scripts/finetune_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
    ```

## 7. 框架详解 (Framework Deep Dive)

本章节详细介绍 RingMoE 框架的核心组件，帮助你进行更深度的定制。

### 7.1 支持的模型架构 (Architectures)
在配置文件中通过 `arch` 字段指定。代码位于 `ringmoe_framework/arch`。
*   **mae**: Masked Autoencoders。
*   **simmim**: SimMIM (Simple Masked Image Modeling)。
*   **ringmo_framework**: RingMo 基础架构。
*   **ringmo_mm**: RingMo 多模态架构。
*   **simmim_moe**: 结合 MoE 的 SimMIM。
*   **simmim_single_moe**: 单一 MoE 层的 SimMIM 变体。

### 7.2 损失函数 (Loss Functions)
在配置文件中通过 `loss_type` 字段指定。代码位于 `ringmoe_framework/loss`。
*   **ce_smooth**: 带标签平滑的交叉熵损失 (CrossEntropySmooth)。
*   **ce_smooth_mixup**: 支持 Mixup 的平滑交叉熵损失。
*   **ce_ignore**: 忽略特定标签的交叉熵损失。
*   **soft_ce**: 软目标交叉熵损失 (SoftTargetCrossEntropy)。
*   **InfoNceLoss**: 用于对比学习的 InfoNCE 损失。

### 7.3 优化器 (Optimizers)
在配置文件 `optimizer` 部分的 `optim_name` 字段指定。代码位于 `ringmoe_framework/optim`。
*   **SGD**: 随机梯度下降。
*   **AdamW**: Adam 优化器 + 权重衰减。
*   **AdamWOP**: 优化的 AdamWeightDecayOp。
*   **FP32AdamWOP**: 保持 FP32 状态的 AdamWeightDecay。

### 7.4 学习率策略 (Learning Rate Schedules)
在配置文件 `lr_schedule` 部分的 `lr_type` 字段指定。代码位于 `ringmoe_framework/lr`。
*   **cosine_decay**: 余弦退火。
*   **warmup**: 仅预热。
*   **warmup_cosine_decay**: 预热 + 余弦退火 (V1)。
*   **warmup_cosine_decay_simmim**: 适用于 SimMIM 的预热 + 余弦退火 (V2)。
*   **warmup_multistep_decay**: 预热 + 多步衰减。

### 7.5 监控与断点续训 (Monitoring & Checkpointing)
*   **监控 (Monitors)**:
    *   `LossMonitor`: 打印训练 Loss。
    *   `StateMonitor`: 监控训练状态。
    *   `CheckpointMonitor`: 自动保存 Checkpoint。
*   **断点续训 (Checkpointing)**:
    *   使用 `load_ckpt` 工具函数。
    *   支持分布式 Checkpoint 的加载与合并。
    *   在 `parallel_config` 中配置 `strategy_ckpt_load_file` 可支持半自动并行模式下的策略加载。

### 7.6 训练核心 (Trainer)
代码位于 `ringmoe_framework/trainer`。
*   **trainer.py**: 定义了通用的训练流程。
*   **wrapper.py**: 封装了前向计算、反向传播和优化器更新步骤 (`TrainOneStepCell`)。
*   **ema.py**: 指数移动平均 (Exponential Moving Average) 实现，用于稳定训练。
*   **clip_grad.py**: 梯度裁剪工具。

### 7.7 训练流程详解 (Training Pipeline)

> 目标：把“从配置文件到启动训练/微调/评估”的真实代码调用链讲清楚，方便你按模块快速定位与修改。

#### 7.7.1 总览：三类入口脚本
* **预训练入口**：`pretrain.py`
* **微调入口**：`finetune.py`
* **评估入口**：`eval.py`

三者的结构非常相似：
`加载配置 -> build_context -> build_dataset -> build_parallel_config -> build_model -> build_lr/build_optim -> build_wrapper -> load_ckpt -> callback -> Model.train/eval`

#### 7.7.2 配置链路：YAML -> RingMoEConfig -> 运行时参数
1. **主配置文件（你通常直接用它启动）**：`config/simmim_pcl/*.yaml`
2. **基础配置复用/继承机制**：在 YAML 中通过 `base_config` 字段引入 `config/base/**` 下的基础配置。
3. **配置加载与合并**（关键代码）：
   * `register/config.py: RingMoEConfig._file2dict()`：读取 yaml，并递归合并 `base_config`。
   * `register/config.py: RingMoEConfig.merge_from_dict()`：把命令行 `--options a.b.c=xxx` 覆盖到配置里。
   * `register/config.py: ActionDict`：负责把 `--options key=value` 解析为 dict。

> 实战建议：优先“新建一个 yaml，继承 base_config 并覆盖字段”，避免直接改 `config/base/**`。

---

#### 7.7.3 预训练 (Pretrain) 完整流程与对应文件
入口：`pretrain.py: main(args)`

1. **初始化上下文（图模式/分布式/seed/日志/可选 profile）**
   * 调用：`ringmoe_framework/tools/helper.py: build_context(args)`
   * 关键点：内部会调用 `ac.context_init(...)` 初始化分布式 rank 信息，并设置 `context.set_auto_parallel_context(...)`。

2. **构建预训练数据集**
   * 调用：`ringmoe_framework/datasets/build_dataset.py: build_dataset(config, is_pretrain=True)`
   * 实际分发到：`ringmoe_framework/datasets/pretrain_dataset.py: create_pretrain_dataset(config)`

3. **构建并行策略与 MoE 配置（把 dict 变成 MindSpore 并行对象）**
   * 调用：`ringmoe_framework/parallel_config.py: build_parallel_config(config)`
   * 输出：
     * `config.moe_config -> MoEConfig(...)`
     * `config.recompute_config -> TransformerRecomputeConfig(...)`
     * `config.parallel_config -> TransformerOpParallelConfig(...)`

4. **构建预训练网络（按 arch 分支选择）**
   * 调用：`ringmoe_framework/arch/build_arch.py: build_model(config)`
   * 选择分支：`mae / simmim / ringmo_framework / ringmo_mm / simmim_moe / simmim_single_moe`

5. **构建学习率与优化器**
   * 学习率：`ringmoe_framework/lr/build_lr.py: build_lr(args)`
   * 优化器：`ringmoe_framework/optim/build_optim.py: build_optim(args, net, lr_schedule, logger)`

6. **构建训练封装（LossScale + 反向 + 梯度裁剪 + EMA + 可选 pipeline）**
   * 调用：`ringmoe_framework/trainer/trainer.py: build_wrapper(args, net, optimizer)`
   * 说明：
     * `pipeline_stage > 1` 时走 `TrainPipelineWithClipGNAndEMA` + `PipelineCell/MicroBatchInterleaved`
     * 否则走 `TrainOneStepWithClipGNAndEMA`

7. **加载 checkpoint（断点续训/恢复训练状态）**
   * 调用：`ringmoe_framework/tools/load_ckpt.py: load_ckpt(...)`

8. **构建回调并启动训练**
   * 回调：`ringmoe_framework/monitors/callback.py: build_pretrain_callback(args, cfts)`
   * 启动：`mindspore.train.Model(...).train(...)`（在 `pretrain.py` 内调用）

---

#### 7.7.4 微调 (Finetune) 完整流程与对应文件
入口：`finetune.py: main(args)`

与预训练的主要差异：
* 数据集走 `finetune_dataset`，并且同时构建 train/eval 两份。
* 网络来自 `ringmoe_framework/models`（下游任务模型），并构建 `eval_engine`。
* 会额外构建 `loss`，再包装 `WithLossCell` 或 MoE 的 `ClassificationMoeWrapper`。

关键步骤与文件：
1. `build_context`：`ringmoe_framework/tools/helper.py`
2. train/eval dataset：`ringmoe_framework/datasets/finetune_dataset.py`
3. parallel/moe：`ringmoe_framework/parallel_config.py`
4. 下游模型构建：`ringmoe_framework/models/build_model.py: build_model(args)`
5. 评估引擎：`ringmoe_framework/models/*: build_eval_engine(...)`
6. finetune lr：`ringmoe_framework/lr/build_finetune_lr.py: build_finetune_lr(args)`
7. loss：`ringmoe_framework/loss/build_loss.py: build_loss(args)`
8. wrapper：`ringmoe_framework/trainer/trainer.py: build_wrapper(...)`
9. ckpt：`ringmoe_framework/tools/load_ckpt.py: load_ckpt(..., is_finetune=True, valid_dataset=eval_dataset)`
10. callback：`ringmoe_framework/monitors/callback.py: build_finetune_callback(args, cfts, eval_engine)`

---

#### 7.7.5 评估 (Eval) 完整流程与对应文件
入口：`eval.py: main(args)`

1. `build_context`：`ringmoe_framework/tools/helper.py`
2. eval dataset：`ringmoe_framework/datasets/finetune_dataset.py`（通过 `build_dataset(..., is_train=False)` 分发）
3. parallel/moe：`ringmoe_framework/parallel_config.py`
4. 模型 + eval_engine：`ringmoe_framework/models/build_model.py` 与 `ringmoe_framework/models/*: build_eval_engine`
5. 加载 ckpt：`mindspore.train.serialization.load_checkpoint(...)` + `net.load_pretrained(params_dict)`（在 `eval.py` 内）
6. 执行 eval：`eval_engine.eval()` + `eval_engine.get_result()`

---

#### 7.7.6（可选）如何把“训练流程”写进配置（你可以覆盖的关键字段）
> 说明：RingMoE 的“流程本身”由 `pretrain.py/finetune.py/eval.py` 固定驱动；配置能控制的是 **每一步的行为与超参数**。

你通常会在你的 YAML 里覆盖这些字段来“配置化控制训练流程”：
* `context`: 设备、模式等（对应 `build_context`）
* `pretrain_dataset` / `finetune_dataset`: 数据路径、增强、num_workers（对应 `build_dataset`）
* `parallel_config` / `recompute_config`: 并行与重计算（对应 `build_parallel_config`）
* `moe_config`: 专家数、aux loss（对应 `build_parallel_config`）
* `arch`: 选择预训练架构（对应 `arch/build_arch.py:build_model`）
* `optimizer` + `lr_schedule`: 优化器与 LR（对应 `build_optim/build_lr`）
* `train_wrapper`: LossScale/clip/ema/pipeline 微批（对应 `trainer/trainer.py:build_wrapper`）
* `train_config`: epoch、sink_mode、per_epoch_size、resume_ckpt（对应 train/eval 行为与 ckpt 加载）
* `callback`: ckpt 保存策略、summary/obs 上传等（对应 `monitors/callback.py`）

## 8. 配置详解

RingMoE 使用 YAML 文件进行配置管理。主要的配置项包括：

### 8.1 基础配置
*   **base_config**: 继承的基础配置列表，方便复用。
*   **context**: MindSpore 上下文配置 (模式、设备等)。

### 8.2 并行策略 (Parallel Config)
RingMoE 支持多种并行模式，在 `parallel_config` 中定义：
*   **data_parallel**: 数据并行度。
*   **model_parallel**: 模型并行度 (Tensor Parallel)。
*   **expert_parallel**: 专家并行度 (MoE 特有)。
*   **pipeline_stage**: 流水线并行阶段数。
*   **optimizer_shard**: 是否开启优化器切分。

### 8.3 MoE 配置 (MoE Config)
MoE 模型的具体参数，在 `moe_config` 中定义：
*   **expert_num**: 专家的总数量。
*   **capacity_factor**: 专家容量因子 (>=1.0)，用于控制专家能处理的最大 token 数。
*   **aux_loss_factor**: 负载均衡辅助损失的权重。
*   **num_experts_chosen**: 每个 token 选择的专家数量 (Top-K)。

修改配置时，建议新建一个 yaml 文件，继承基础配置并覆盖你需要修改的参数，而不是直接修改基础配置文件。

## 9. 下游任务评估

RingMoE 的下游任务评估（如分类、分割、检测）通常结合其他开源框架进行：

*   **场景分类**: 使用 [mmpretrain](https://github.com/open-mmlab/mmpretrain)
*   **语义分割**: 使用 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
*   **目标检测**: 使用 [mmdetection](https://github.com/open-mmlab/mmdetection)

你需要将预训练好的 RingMoE 模型权重加载到这些框架中对应的模型结构里进行微调或评估。

## 10. 常见问题 (Troubleshooting)

*   **Rank Table 错误**: 确保 `rank_table.json` 中的 IP 地址与机器实际 IP 一致，且 `device_id` 对应正确。
*   **OOM (Out of Memory)**:
    *   尝试减小 `batch_size`。
    *   增加 `model_parallel` 或 `expert_parallel` 的并行度。
    *   开启 `recompute` (重计算)。
*   **数据加载慢**: 确保数据已转换为 MindRecord 格式，并适当增加 `num_workers`。

---
如有更多问题，请查阅 `README.md` 或参考 MindSpore 官方文档。
