# RingMoE 在 A100 上的“完全重构”深度研究（现状、差距、路线图）

本文面向本仓库维护者/二次开发者：在 **Linux + NVIDIA A100** 场景下，如何把原始 **MindSpore + Ascend 910B** 的 RingMoE 训练体系，系统性地迁移/重构到 **PyTorch + DeepSpeed**。

> 可直接运行的 PyTorch/DeepSpeed 版本入口在 `pytorch_refactor/`，对应运行手册见 `RUNNING_LINUX_A100.md`。

---

## 1. 当前 PyTorch/DeepSpeed 重构版“已经做到什么”

目录：`pytorch_refactor/`

- **训练入口**：`pytorch_refactor/train.py`
  - 单卡/无 DeepSpeed：`--no_deepspeed`（用于快速验证）
  - 多卡/DeepSpeed：默认读取 `pytorch_refactor/ds_config.json`（bf16 优先，必要时可 `--force_fp16`）
  - 支持 `--tf32`、`--use_checkpoint`、`--seed`、`--log_every`、`--aux_loss_factor`
  - 支持“多模态 JSON”（见 `RUNNING_LINUX_A100.md` 4.1）：按模态不同通道数做 patch-embed，并在 encoder 内沿 batch 维拼接训练；decoder/loss 按模态拆分对齐
- **模型**：`pytorch_refactor/model.py`
  - Swin Transformer V2（窗口注意力、PatchMerging、可选 checkpoint）
  - MoE：使用 DeepSpeed `deepspeed.moe.layer.MoE`（存在则启用；否则需 `--disable_moe`）
  - SimMIM：mask token + PixelShuffle decoder + masked L1 loss
- **数据**：`pytorch_refactor/dataset.py`
  - JSON 读路径列表（单模态）或 list-of-paths（多模态，返回 `(x0,m0,x1,m1,...)` 便于不同通道数）
  - 多通道模态支持 `.npy/.npz`（可选 `.tif/.tiff` + `tifffile`）
  - 随机 mask 生成（patch 级，和模型 patch 对齐）
- **导出/推理**：
  - `pytorch_refactor/export_pt.py`：把 DeepSpeed ZeRO checkpoint 导出为单文件 `model.pt`
  - `pytorch_refactor/infer.py`：加载 `model.pt` 做 SimMIM 重建

> 结论：当前版本更像是“可在 A100 上跑起来的最小闭环（SimMIM + SwinV2 + MoE + DeepSpeed）”，并非原始 RingMoE 的全功能等价实现。

---

## 2. 与 MindSpore 原版的关键差距（为什么说还不是“完全重构”）

下面按模块列出原版（`ringmoe_framework/`）能力与 PyTorch 重构版的差距；这也是后续研究/实现的优先级清单。

### 2.1 配置系统与组合（YAML 继承/覆盖）

原版大量依赖 `config/**` 的 `base_config` 继承链、覆盖逻辑、以及与并行/数据/runner 的强绑定。

PyTorch 版目前以 CLI + `ds_config.json` 为主：
- 不支持原版 `base_config` 合并逻辑
- 不支持原版 `parallel_config / moe_config / recompute_config` 的完整语义映射

### 2.2 数据体系（MindRecord、多模态比例采样、遥感格式）

原版数据入口非常复杂（示例：`ringmoe_framework/datasets/pretrain_dataset.py`）：
- MindRecord / 自定义读取
- 多模态（opt/sar/ms/hsi 等）并行加载、按比例组成 batch、不同模态不同 transform
- TIFF/多通道数据转换、以及多种 mask 策略（SimMIM/MAE/RingMoMM/PI Mask）

PyTorch 版目前仅提供：
- JSON 路径列表 + PIL 读 RGB
- 多模态支持 list-of-paths，并可通过 `--modal_in_chans` 处理不同通道数 + 多 decoder 分支
- 仍不包含“多模态比例采样/不同模态不同 transform”的完整逻辑

### 2.3 任务/架构覆盖面（MAE、RingMo、RingMoMM…）

原版 `ringmoe_framework/arch/` 覆盖：
- SimMIM（含单模态与多模态 + MoE 变体）
- MAE、RingMo、RingMoMM 等任务
- ViT/Swin/SwinV2 等不同 backbone

PyTorch 版目前只覆盖：
- SimMIM + SwinV2（+ 可选 MoE）

### 2.4 多模态 SimMIM 的“真实等价”仍缺失

原版多模态 SimMIM（示例：`ringmoe_framework/arch/simmim_moe.py`）存在：
- **不同模态不同通道数**（例如 3/8/3/4）
- **不同 decoder 分支**（每个模态独立 decoder 输出不同通道数）
- 训练 loss 为多模态 loss 之和 + MoE loss

PyTorch 版目前只做了：
- 不同通道数的多模态 patch-embed + 多 decoder 分支 + 多模态 loss 求和（与原版 `3/8/3/4` 结构对齐）
- 仍缺：按模态定制 transform/归一化、以及更复杂的采样/数据组织方式

### 2.5 分布式并行语义不等价

原版依赖 MindSpore 的半自动/自动并行、专家并行、流水线并行等能力（对应 `parallel_config`）。

PyTorch 版目前依赖 DeepSpeed：
- ZeRO（目前 ds_config 为 stage 2）
- MoE（依赖 DS MoE kernels）

但“语义等价”仍缺：
- 专家并行拓扑（DeepSpeed MoE `ep_size` 等参数映射）
- 流水线并行（Pipeline Parallel）与原版 pipeline_stage 的对齐
- 原版策略 ckpt/并行策略保存与恢复

---

## 3. 面向 A100 的“完全重构”推荐路线图（建议按阶段推进）

### 阶段 A：把 PyTorch 训练系统做成可扩展的“框架骨架”

- 引入统一配置层（建议：OmegaConf/Hydra 或 pydantic + PyYAML）
- 将 `train.py` 的超参/模型/数据拆成可组合组件（dataset/model/optimizer/scheduler/runner）
- 约定 checkpoint 格式与日志接口（tensorboard/wandb 可选）

### 阶段 B：数据体系对齐（这是工作量最大且最关键的一步）

- 支持多通道遥感数据（TIFF/NumPy），并支持“按模态不同通道数”的 decode
- 支持多模态 batch 采样策略（按比例混合、不同 transform）
- 将原版 mask 策略迁移为 PyTorch 实现（SimMIM/MAE/RingMoMM/PI Mask）
- 明确数据格式（建议从 MindRecord 迁移到 WebDataset/tar shards 或者 Parquet/Arrow，便于多机吞吐）

### 阶段 C：模型与任务对齐

- SwinV2 结构/超参对齐原版（embed_dim/depths/heads/window_size 等）
- 多模态 SimMIM：按模态配置 decoder 输出通道、loss 计算
- 逐步补齐 MAE / RingMo / RingMoMM（先单模态，再扩多模态）

### 阶段 D：分布式扩展与性能优化（A100 关键）

- DeepSpeed ZeRO stage 3、activation checkpoint、CPU/NVMe offload（按需要）
- MoE：对齐 `capacity_factor`、路由策略、以及 EP（expert parallel）拓扑
- attention/算子优化：优先用 PyTorch 2.x 的 fused/SDPA 路径；必要时再上定制 kernel
- 多机：NCCL/网络拓扑、hostfile/Slurm launch，稳定性与可观测性（健康检查、超时、重启）

---

## 4. 建议你现在先选哪一条继续“深入”

为了避免一次性把范围做爆，建议你在下面 3 个方向里选 1 个，我可以继续把代码推进到可用的下一阶段：

1) **多模态真实对齐**：支持每模态不同通道数 + 多 decoder 分支 + loss 对齐  
2) **配置系统对齐**：把 `config/**` 的关键字段映射到 PyTorch（先 SimMIM + SwinV2 + MoE）  
3) **分布式/性能专项**：多机 DeepSpeed（ZeRO3 + EP）+ A100 性能基线与 profiling

你回复一个方向（1/2/3）并补充你的实际训练条件（单机/多机、GPU 数、数据模态与通道数），我就继续把对应代码补齐。
