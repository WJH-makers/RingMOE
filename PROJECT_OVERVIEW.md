# RingMoE 项目重点内容总览（含问题清单）

> 目标：把本仓库的“核心代码路径 + 配置体系 + 训练/数据/模型/并行流程 + 运行方式 + 风险点”集中写在一个文档里，便于快速理解与二次开发。

---

## 1. 项目用途与定位

本仓库用于 **RingMoE 预训练** 以及 **下游任务评测/微调**（遥感方向、MindSpore + Ascend 场景为主）。

README 中提到的关键依赖（以仓库实际代码为准）：

- MindSpore 2.1.0
- Python 3.7
- CANN 6.3.RC2
- gdal（部分离线裁剪脚本需要）
- Ascend 910B 集群（多机多卡、rank table 启动方式）

---

## 2. 仓库结构（你应该从哪里开始读）

### 2.1 顶层入口脚本（训练/评估从这里进）

- `pretrain.py`：预训练入口（构建 context → dataset → 并行配置 → 模型 → lr/optim → wrapper → ckpt → callback → `Model.train()`）
- `finetune.py`：微调入口（训练/评估 dataset、loss、eval_engine、wrapper、ckpt、callback）
- `eval.py`：评估入口（加载 ckpt → eval_engine → 输出指标）

其它顶层脚本/文件（辅助理解或用于特定环境）：

- `parallel_config.py`：并行配置的一个独立实现（与 `ringmoe_framework/parallel_config.py` 并存；当前 `pretrain.py` 实际使用的是后者）
- `test_config.py`：用于测试/打印配置解析（当前引用了不存在的 `RingMoConfig`，见问题清单）
- `README.md`：仓库用途、依赖、数据与下游评测入口说明

### 2.2 核心框架目录（主要逻辑都在这里）

目录：`ringmoe_framework/`

- `arch/`：预训练任务级模型组装（MAE / SimMIM / RingMo / RingMoMM / SimMIM+MoE 等）
- `datasets/`：预训练/微调数据、mask 策略、MindRecord 工具、裁剪脚本、增强
- `models/`：backbone（ViT/Swin/SwinV2）与基础层（Attention/Block/MoE 等）
- `trainer/`：TrainOneStep 封装（loss scale / clip grad / EMA / pipeline）
- `optim/`：优化器与组参（AdamW/AdamWOP/FP32AdamWOP 等）
- `lr/`：学习率调度（warmup/cosine/multistep）
- `tools/`：context 初始化、ckpt 加载/重映射、HCCL 工具
- `monitors/`：callback/监控

### 2.3 配置与脚本（跑起来必须看）

- `config/`
  - `config/base/**`：基础配置片段（context/datasets/models/schedules/runner/modelarts/全局 base）
  - `config/simmim_pcl/**`：示例组合配置（包含 MoE + 多模态字段）
  - `config/test_register.yaml`：mask 注册相关的示例
- `scripts/*.sh`：分布式启动脚本（按 device 复制代码到子目录并后台启动多个进程）
- `rank_table_*.json`：Ascend rank table 示例（2/8/16 节点）

### 2.4 ModelArts/OBS 相关

- `aicc_tools-0.2.1-py3-none-any.whl`：ModelArts/OBS/日志/监控等工具包（`ma-pre-start.sh` 会安装）
- `ma-pre-start.sh`：ModelArts 预启动脚本（环境变量、安装 wheel、示例命令）
- `obssync.sh`、`summary.sh`：obsutil 同步脚本（含敏感信息，见“问题清单”）

### 2.5 其它辅助/产物

- `autotune/autotune_0.json`：MindSpore Dataset AutoTune 输出（数据 pipeline 树与结论）
- `register/`：配置解析与注册工厂（当前仓库里多数构建未深度依赖注册机制，但它是基础设施）
- `.idea/`、`__pycache__/`、`*.pyc`：IDE/缓存产物（建议不入库，见“问题清单”）

---

## 3. 典型运行方式（怎么启动）

### 3.1 预训练（分布式）

README 提示方式（Ascend rank table）：

```bash
sh ./scripts/pretrain_distribute.sh RANK_TABLE_FILE CONFIG_PATH
```

`scripts/pretrain_distribute.sh` 的行为：

- 设置 `RANK_TABLE_FILE / RANK_SIZE / DEVICE_ID / RANK_ID`
- 为每个设备创建 `pretrain_parallel{i}` 子目录并复制必要代码（顶层脚本 + config + register + ringmoe_framework）
- 在每个子目录后台执行：

```bash
python pretrain.py --config=$CONFIG_FILE &> pretrain_log &
```

### 3.2 微调（分布式）

```bash
sh ./scripts/finetune_distribute.sh RANK_TABLE_FILE CONFIG_PATH
```

### 3.3 评估（分布式）

```bash
sh ./scripts/eval_distribute.sh RANK_TABLE_FILE CONFIG_PATH
```

> 注意：`scripts/eval_distribute.sh` 当前会复制 `../src`（仓库中不存在），属于已知问题。

---

### 3.4 下游任务评测（仓库只提供说明）

README 提到的下游 benchmark 实验框架（本仓库不包含这些框架代码）：

- 场景分类：mmpretrain
- 语义分割：mmsegmentation
- 目标检测：mmdetection（水平框）/ mmrotate（旋转框）
- 目标跟踪：mmdetection + ByteTrack
- 变化检测：BIT_CD
- 单目深度估计：Binsformer（Monocular-Depth-Estimation-Toolbox）

---

## 4. 配置系统（YAML 如何加载/继承/覆盖）

配置类：`register/config.py` → `RingMoEConfig`

### 4.1 base_config 继承

- YAML 若包含 `base_config` 字段（string 或 list），会先递归加载这些“基配置”，再把当前 YAML 覆盖到合并结果上。
- 示例：`config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml` 通过 `base_config` 引入：
  - context / dataset / model / schedule / runner / modelarts / 全局 base 等，再在本文件中覆写 MoE、多模态与训练细节。

### 4.2 `--options` 覆盖（仅 `pretrain.py` 已实现）

`pretrain.py` 支持：

```bash
python pretrain.py --config xxx.yaml --options train_config.batch_size=2 model.embed_dim=1024
```

覆盖解析由 `register/config.py` 的 `ActionDict` 实现（注意：该文件存在布尔解析 bug，见第 12 节）。

---

## 5. 预训练主流程（`pretrain.py`）

`pretrain.py` 的关键编排顺序：

1. `ringmoe_framework.tools.helper.build_context(args)`
   - 通过 `aicc_tools.context_init(...)` 初始化分布式、获取 `local_rank/device_num`，设置 logger
   - 设置 MindSpore GRAPH_MODE、max_device_memory、comm_fusion、优化器并行、pipeline stages 等
   - 配置策略 ckpt 保存/加载路径（`strategy_ckpt_save_file/strategy_ckpt_load_file`）
2. `ringmoe_framework.datasets.build_dataset(args)` → `create_pretrain_dataset`
3. `ringmoe_framework.parallel_config.build_parallel_config(args)`
   - 把 `moe_config/recompute_config/parallel_config` 从 dict 转成并行对象（含自定义 `MoEConfig`）
4. `ringmoe_framework.arch.build_model(args)`（按 `arch` 选择预训练任务网络）
5. `ringmoe_framework.lr.build_lr(args)`（LR schedule）
6. `ringmoe_framework.optim.build_optim(args, net, lr, ...)`（优化器/组参）
7. `ringmoe_framework.trainer.build_wrapper(args, net, optimizer, ...)`（TrainOneStep：loss scale、clip grad、EMA、pipeline）
8. `ringmoe_framework.tools.load_ckpt.load_ckpt(...)`（断点恢复/加载预训练权重）
9. `ringmoe_framework.monitors.callback.build_pretrain_callback(...)`（loss/ckpt/summary/obs callback）
10. `mindspore.train.Model(...).train(...)`

---

## 6. 微调/评估主流程（`finetune.py` / `eval.py`）

### 6.1 微调（`finetune.py`）

- dataset：
  - `build_dataset(args, is_pretrain=False)` → `ringmoe_framework/datasets/finetune_dataset.py:create_finetune_dataset`
- 模型：
  - `ringmoe_framework/models/build_model.py`（按 `model.backbone`：vit/swin/swin_v2）
- loss：
  - `ringmoe_framework/loss/loss.py:get_loss(args)`（由 `args.loss_type` 决定）
- eval：
  - `ringmoe_framework/models/eval/eval_engine.py` 封装评估网络与分布式 accuracy（见第 10 节）

### 6.2 评估（`eval.py`）

流程与微调相近：context → dataset → parallel_config → model → load ckpt → eval_engine → 输出指标。

> 注意：`finetune.py` / `eval.py` 当前存在若干“跑不通”的明显问题（见第 12 节）。

---

## 7. 模型结构（预训练 arch 与微调 backbone 的分工）

### 7.1 `ringmoe_framework/arch/`：预训练任务级组装

`ringmoe_framework/arch/build_arch.py` 根据 `config.arch` 选择：

- `mae` → `ringmoe_framework/arch/mae.py`
  - Encoder：`VisionTransformerForMae`
  - Loss：`MSELoss`
  - 关键配置：`model.mask_ratio/decoder_layers/decoder_num_heads/decoder_dim/norm_pixel_loss`
- `simmim` → `ringmoe_framework/arch/simmim.py`
  - 支持 backbone：`vit` / `swin` / `swin_v2`
  - Loss：`L1Loss`
  - 关键配置：`model.mask_ratio/mask_patch_size/patch_size/...`
- `ringmo_framework`（注意：这是 arch 字符串，不是目录名）→ `ringmoe_framework/arch/ringmo.py`
  - 支持 backbone：`vit` / `swin`
  - 关键配置：`model.patch_type/inside_ratio/mask_patch_size/use_lbp`
- `ringmo_mm` → `ringmoe_framework/arch/ringmo_mm.py`
  - 多模态版本：重建 +（可选）对比学习（InfoNCE）
  - 关键配置：`model.modal_num/temperature/clr_loss_weight/out_dim/lamda/use_contranst`
- `simmim_moe` → `ringmoe_framework/arch/simmim_moe.py`
  - SimMIM + MoE（支持多模态输入拼接与 aux_loss）
  - 关键配置：`moe_config.*` + `model.modal_num`
- `simmim_single_moe` → `ringmoe_framework/arch/simmim_single_moe.py`
  - 另一种 MoE 变体（单路由/特定专家路由逻辑）

### 7.2 `ringmoe_framework/models/`：微调 backbone 与基础层

- `ringmoe_framework/models/build_model.py`
  - 根据 `model.backbone` 构建微调模型（FinetuneVit / FinetuneSwin / SwinV2 变体）
- `ringmoe_framework/models/backbone/`
  - `vit.py`：ViT + encoder（含 MoE 开关）
  - `swin_transformer.py`：Swin
  - `swin_transformerv2.py`：SwinV2
- `ringmoe_framework/models/layers/`
  - Attention / Block / Patch / MLP / LayerNorm 等
  - MoE：`moe.py`（基础 MoE）、`moe_modal.py`（多模态路由）、`moe_single.py`（特定专家路由）、`moe_new.py`（更复杂版本）
- `ringmoe_framework/models/core/`
  - 相对位置编码、Depth2Space、TokenExchange、SincosPosEmbed、ScatteringCorrection 等

---

## 8. 数据管道（预训练 vs 微调）

### 8.1 预训练数据：`ringmoe_framework/datasets/pretrain_dataset.py`

入口：`create_pretrain_dataset(args)`

关键点：

- 支持两类数据源（`pretrain_dataset.data_type`）：
  - `mindrecord`：`de.MindDataset`（可分片 `num_shards/shard_id`）
  - `custom`：`de.GeneratorDataset`（读取 json id 列表 + 图片）
- 支持多模态（`pretrain_dataset.modal_type == "multi_modal"`）：
  - `build_dataset()` 会返回 dataset 列表（按模态分别读取）
  - 对不同模态做不同预处理（示例：SAR_L1/MS 走 `Tiff_converter + Power_generation`；SAR_L2 走 `Tiff_converter_toRGB`）
  - 对各模态分别应用 transforms（`build_transforms_list`），再统一 mask（`MaskPolicyForSim`）
  - 通过 ZipDataset 合并为 batch，并 repeat 到 epoch

配套离线工具（可用于数据准备）：

- `ringmoe_framework/datasets/tools/mindrecord.py`：示例 MindRecord 生成脚本（schema + 写入 + shard）
- `ringmoe_framework/datasets/tools/cut_image.py`：把大图切成 192×192 patch 的示例脚本
- `ringmoe_framework/datasets/image_cuts/*`：SAR/MS 等 TIFF 影像滑窗裁剪示例（偏离线预处理）

### 8.2 mask 策略：`ringmoe_framework/datasets/mask/mask_policy.py`

- `MaskPolicyForSim`：SimMIM 的 patch mask（按网格随机 mask）
- `MaskPolicyForMae`：MAE 风格 mask（mask/ids_restore/unmask_index）
- `MaskPolicyForRingMoMM`：多模态/对比相关的 mask + ids_restore
- `MaskPolicyForPIMask`：PI mask（inside_ratio；可选 LBP）

### 8.3 微调数据：`ringmoe_framework/datasets/finetune_dataset.py`

入口：`create_finetune_dataset(config, is_train=True/False)`

- 使用 `de.ImageFolderDataset(train_path/eval_path)`（ImageNet 风格目录）
- 训练增强：RandomResizedCrop、HFlip、ColorJitter、Normalize、RandomErasing
- 可选 Mixup/Cutmix：`ringmoe_framework/datasets/transforms/*`
- 评估增强：Resize → CenterCrop → Normalize → HWC2CHW

---

## 9. 并行与分布式（核心概念）

### 9.1 三层并行参数（最常用）

- `parallel`：MindSpore auto parallel context 参数（parallel_mode/full_batch/gradients_mean/...）
- `parallel_config`：Transformer 并行切分（dp/mp/ep/pipeline_stage/micro_batch_num/optimizer_shard/...）
- `moe_config`：MoE 专家配置（expert_num/specific_expert_num/public_expert_num/cross_expert_num/...）

### 9.2 context 初始化细节

`ringmoe_framework/tools/helper.py:build_context()`：

- `aicc_tools.context_init(...)`：分布式初始化、获取 rank/device_num、日志
- 训练性能相关设置：max_device_memory、comm_fusion、parallel_optimizer_config、pipeline_stages 等
- 策略 ckpt：按 rank 设置 `strategy_ckpt_save_file`，并支持从 OBS 拉取 `strategy_ckpt_load_file`

### 9.3 rank table / HCCL

- `rank_table_*.json`：多机多卡 rank table 示例
- `ringmoe_framework/tools/hccl_tools.py`：生成单机 HCCL 配置（偏裸机/容器）

---

## 10. 监控、Checkpoint、恢复训练

### 10.1 ckpt 加载/重映射

`ringmoe_framework/tools/load_ckpt.py`：

- 支持单卡/数据并行/半自动并行加载 ckpt
- 半自动并行会对参数做过滤（如 `adam_*`、relative position bias、patch_embed、decoder 等）
- 微调加载包含重映射/插值逻辑（ViT/Swin 位置编码适配）

### 10.2 回调与 OBS

`ringmoe_framework/monitors/callback.py`：

- `build_pretrain_callback()`：LossMonitor、Checkpoint、Summary、OBS monitor
- `build_finetune_callback()`：Checkpoint、StateMonitor（含 eval interval）、OBS monitor

---

## 11. config 目录索引（有哪些可用配置）

### 11.1 `config/base/models/*`（模型结构片段）

- ViT：`vit_base_p16.yaml`、`vit_large_p16.yaml`、`vit_huge_p14.yaml`
- Swin：`swin_tiny_p4_w6.yaml`、`swin_base_p4_w6.yaml` 等
- SwinV2：`swinv2_base_p4_w12.yaml`、`swinv2_giant_p4_w12.yaml` 等
- MAE：`mae_vit_base_p16.yaml`（注意：`mae_vit_base_p16_moe_32_1.yaml` 文件为空）
- SimMIM：`simmim_vit_base_p16.yaml`、`simmim_vit_3b.yaml`、`simmim_swinv2_giant_p4_w12.yaml` 等
- RingMo / RingMoMM：`ringmo_vit_base_p16.yaml`、`ringmo_swin_base_p4_w6.yaml`、`ringmo_mm_vit_base_p16.yaml`、`ringmo_mm_vit_3b.yaml`

### 11.2 `config/base/context/*`（分布式/并行 context）

- `default_mode.yaml`：默认 context
- `semi_moe_*_mode_4nodes.yaml`：半自动并行 + MoE 的参考配置（不同 expert_parallel / dp 组合）

### 11.3 dataset / runner / schedule

- 预训练 dataset：`config/base/datasets/pretrain_dataset.yaml`
- 微调 dataset：`config/base/datasets/finetune_dataset.yaml`
- runner：`config/base/runner/runner.yaml`
- schedule：`config/base/schedules/default_schedule.yaml`
- modelarts：`config/base/modelarts/aicc.yaml`

### 11.4 示例组合配置

- `config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml`
  - 用于 SimMIM + MoE + 多模态的预训练示例（依赖 ModelArts 路径结构）

---

## 12. 已发现问题与风险点（务必先看）

### 12.1 安全风险（高优先级）

- **硬编码 OBS 访问凭据（AK/SK）**
  - Python：`pretrain.py`、`finetune.py`、`eval.py` 内调用 `ac.obs_register(...)`
  - Shell：`obssync.sh`、`summary.sh` 内执行 `obsutil config -i=... -k=...`
  - 影响：密钥泄露风险极高；建议改为环境变量/密钥管理，避免将含密钥的仓库公开或分享

### 12.2 可运行性问题（会直接报错）

- **模块命名不一致：`ringmo_framework` vs `ringmoe_framework`**
  - 现状：大量文件存在 `from ringmo_framework...` 的导入，但仓库实际目录为 `ringmoe_framework/`
  - 影响：若环境里未额外安装名为 `ringmo_framework` 的包，会直接触发 `ModuleNotFoundError`
  - 代表文件：`ringmoe_framework/lr/build_lr.py`、`ringmoe_framework/loss/build_loss.py`、`ringmoe_framework/models/**/__init__.py`、`ringmoe_framework/datasets/transforms/__init__.py`

- **配置类名写错：`RingMoConfig` 不存在**
  - 现状：`finetune.py`、`eval.py`、`test_config.py` 里使用 `RingMoConfig(...)`
  - 影响：仓库提供的是 `register/config.py` 的 `RingMoEConfig`，上述脚本会在启动阶段就报错

- **`eval.py` 使用了未定义的参数 `args_.options`**
  - 现状：`eval.py` 读取 `args_.options`，但 argparse 未定义 `--options`
  - 影响：运行评估时会触发 `AttributeError`

- **`finetune.py` / `eval.py` 默认 `--config` 指向不存在的 YAML**
  - `finetune.py` 默认：`config/simmim_pcl/finetune_simmim_swinv2_base_p4_w12_aircas_192_200ep.yaml`（仓库中不存在）
  - `eval.py` 默认：`../config/simmim/aircas/vit/pretrain-simmim-vit-moe-p16-01.yaml`（仓库中不存在）

- **`scripts/eval_distribute.sh` 引用不存在目录**
  - 现状：脚本会 `cp -r ../src ...`，但仓库没有 `src/` 目录
  - 影响：按脚本启动评估会在拷贝阶段失败

### 12.3 配置解析/注册机制问题（隐蔽 bug）

- **布尔值解析错误（`TRUE/FALSE`）**
  - 位置：`register/config.py` → `ActionDict._parse_int_float_bool()`
  - 现状：`return val.upper == 'TRUE'` 使用了方法对象比较，导致 `TRUE/FALSE` 的解析结果不符合预期

- **注册工厂 `alias` 参数基本无效**
  - 位置：`register/register.py` → `RingMoEClassFactory.register()` / `register_cls()`
  - 现状：写入 registry 时使用了 `register_class.__name__` 作为 key，而不是 `alias`（且与重复检查的 key 不一致）
  - 影响：如果依赖 `alias` 来取类，可能取不到或出现覆盖行为异常

### 12.4 行为/逻辑“容易误解”的点（不一定是 bug）

- **预训练 epoch 逻辑被硬编码为 1**
  - 位置：`pretrain.py`（`new_epochs = 1`）、`ringmoe_framework/lr/build_lr.py`（`total_epochs = 1`）
  - 影响：`train_config.epoch` 在预训练链路里可能并不生效；实际训练步数更像由 `dataset_size` 与 `callback_step/sink_size` 决定

- **路径强依赖 ModelArts/Ascend 环境**
  - 现状：多个配置/代码默认使用 `/home/ma-user/modelarts/...`、`/cache/...` 等路径与 Ascend 分布式启动方式（rank table）
  - 影响：在非 ModelArts/Ascend 环境直接运行需要改路径/改启动方式

### 12.5 仓库清洁度（可选）

- 仓库包含 `__pycache__/`、`*.pyc`、`.idea/` 等生成物，通常建议加入 `.gitignore` 并从版本库移除（减少噪音与潜在冲突）

---

## 13. 快速索引（从“要改什么”反查文件）

- 启动/跑不起来：`pretrain.py`、`finetune.py`、`eval.py`、`scripts/*.sh`
- 配置/继承/覆盖：`register/config.py`、`config/**`
- 分布式/并行策略：`ringmoe_framework/tools/helper.py`、`ringmoe_framework/parallel_config.py`
- 预训练模型：`ringmoe_framework/arch/*`
- backbone/网络层：`ringmoe_framework/models/*`
- 数据与增强：`ringmoe_framework/datasets/*`
- 优化器/LR：`ringmoe_framework/optim/*`、`ringmoe_framework/lr/*`
- ckpt/恢复/评估：`ringmoe_framework/tools/load_ckpt.py`、`ringmoe_framework/models/eval/*`、`ringmoe_framework/monitors/*`

---

## 14. 代码文件索引（按目录）

> 用于按“目录树”理解工程：每个文件一句话说明职责（只列关键文件）。

### 14.1 顶层

- `pretrain.py`：预训练主入口（context/dataset/parallel/model/lr/optim/wrapper/ckpt/callback）
- `finetune.py`：微调主入口（含 eval_engine、分类损失等）
- `eval.py`：评估入口（加载 task ckpt，跑 eval_engine）
- `parallel_config.py`：并行配置（与 `ringmoe_framework/parallel_config.py` 并存）
- `test_config.py`：配置解析测试脚本
- `README.md`：依赖、数据与下游评测说明
- `ma-pre-start.sh`：ModelArts 预启动脚本（装 wheel/设环境等）
- `obssync.sh`、`summary.sh`：obsutil 同步脚本

### 14.2 `scripts/`

- `scripts/pretrain_distribute.sh`：按 device 复制代码并启动预训练进程
- `scripts/finetune_distribute.sh`：按 device 复制代码并启动微调进程
- `scripts/eval_distribute.sh`：按 device 复制代码并启动评估进程（当前引用了不存在的 `src/`）

### 14.3 `register/`

- `register/config.py`：`RingMoEConfig`（YAML + base_config 合并、`--options` 覆盖解析）
- `register/register.py`：`RingMoEClassFactory`（类注册/实例化工厂）
- `register/__init__.py`：导出 `RingMoEConfig` 与注册相关类

### 14.4 `config/`

- `config/base/__base__.yaml`：全局基础配置（seed/parallel/autotune/aicc 等）
- `config/base/context/*.yaml`：context/并行模式参考配置
- `config/base/datasets/*.yaml`：预训练/微调 dataset 默认项
- `config/base/models/*.yaml`：不同 arch/backbone 的模型结构片段
- `config/base/runner/runner.yaml`：训练 runner/callback/wrapper 默认项
- `config/base/schedules/default_schedule.yaml`：optimizer + lr_schedule 默认项
- `config/base/modelarts/aicc.yaml`：aicc/cfts 默认项
- `config/simmim_pcl/*.yaml`：示例组合配置（SimMIM+MoE+多模态）

### 14.5 `ringmoe_framework/arch/`

- `build_arch.py`：按 `config.arch` 选择具体预训练 arch
- `mae.py`：MAE 预训练任务（MSELoss、decoder、sincos pos 等）
- `simmim.py`：SimMIM 预训练任务（L1Loss）
- `ringmo.py`：RingMo 预训练任务（PI mask 等）
- `ringmo_mm.py`：RingMo 多模态预训练（可选 InfoNCE 对比）
- `simmim_moe.py`：SimMIM + MoE 预训练（aux loss、多模态拼接）
- `simmim_single_moe.py`：另一种 MoE 变体（特定专家/路由逻辑）

### 14.6 `ringmoe_framework/datasets/`

- `build_dataset.py`：预训练/微调 dataset 分发入口
- `pretrain_dataset.py`：预训练数据（mindrecord/custom，多模态、tiff 转换、mask、ZipDataset）
- `finetune_dataset.py`：微调数据（ImageFolderDataset、增强、mixup/cutmix）
- `utils.py`：dataset config 校验/补全（把 train_config/model 中字段下沉到 dataset 配置）
- `mask/mask_policy.py`：SimMIM/MAE/RingMoMM/PI mask 策略
- `tools/mindrecord.py`：MindRecord 生成示例
- `tools/cut_image.py`：图像裁剪示例（生成 192×192 patch）
- `image_cuts/*.py`：遥感 TIFF 裁剪与位深转换示例（偏离线）
- `transforms/*.py`：增强策略（auto augment/mixup/random erasing）

### 14.7 `ringmoe_framework/models/`

- `build_model.py`：按 `model.backbone` 构建微调 backbone（vit/swin/swin_v2）
- `backbone/vit.py`：ViT 与 FinetuneVit（含 MoE 开关）
- `backbone/swin_transformer.py`：Swin backbone
- `backbone/swin_transformerv2.py`：SwinV2 backbone
- `layers/*`：Attention/Block/MLP/Patch/MoE 等实现
- `core/*`：相对位置偏置、Depth2Space、TokenExchange、SincosPosEmbed、ScatteringCorrection 等
- `eval/eval_engine.py`：评估引擎封装（DistAccuracy）
- `eval/metric.py`：分布式正确数统计与 accuracy 计算

### 14.8 `ringmoe_framework/lr/`

- `build_lr.py`：根据 `lr_schedule` 构建 warmup/cosine/multistep 等
- `lr_schedule.py`：具体 schedule 实现（WarmUp/Cosine 等）

### 14.9 `ringmoe_framework/optim/`

- `build_optim.py`：按预训练/微调构建优化器与参数分组（支持 layer-wise lr decay）
- `optimizer.py`：自定义 AdamWOP/FP32StateAdamWeightDecay 等实现

### 14.10 `ringmoe_framework/trainer/`

- `trainer.py`：TrainOneStep 封装（loss scale、clip grad、EMA、pipeline/micro-batch）
- `clip_grad.py`、`clip_grad_v2.py`：全局范数裁剪
- `ema.py`：EMA 权重
- `wrapper.py`：MoE 分类微调的 WithLoss 包装（把 moe_loss 加到总 loss）

### 14.11 `ringmoe_framework/tools/`

- `helper.py`：`build_context()`（分布式初始化、性能参数、策略 ckpt、logger/cfts）
- `load_ckpt.py`：断点恢复/加载预训练或微调 ckpt（含重映射/插值）
- `hccl_tools.py`：生成 HCCL 配置工具

### 14.12 `ringmoe_framework/monitors/`

- `callback.py`：预训练/微调 callback 构建（loss/ckpt/summary/obs）
- `monitor.py`：StateMonitor（统计 loss、fps、按 interval 触发 eval）
