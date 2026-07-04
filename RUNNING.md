# 如何运行 RingMoE（极简版）

> 本仓库主要面向 **Linux + Ascend 910B** 的 MindSpore 分布式训练。
> 若你要在 **Linux + NVIDIA A100/H100** 上运行，请直接使用 `pytorch_refactor/`（见 `RUNNING_LINUX_A100.md`，入口 `train_a100.py` 或 `pytorch_refactor/train.py`）。
> 本文只给出“从配置到启动”的最短路径（预训练 / 微调 / 评估）。更完整说明见 `TUTORIAL.md`。

## 0. 你需要准备什么

- **环境**：MindSpore（建议 2.1.0+）、Python、CANN/驱动（Ascend）。
- **数据**：按配置要求准备为 **MindRecord**（见 `TUTORIAL.md` 第 4 节）。
- **配置**：选择/新建一个 yaml（通常在 `config/**`），并根据你自己的机器/数据路径覆盖参数。
- **分布式**：多卡/多机时需要 `rank_table_*.json`（按实际 IP/Device 修改）。

## 1. 三个入口脚本

- 预训练：`pretrain.py`
- 微调：`finetune.py`
- 评估：`eval.py`

三者内部的核心调用链几乎一致：

`加载YAML配置 -> build_context -> build_dataset -> build_parallel_config -> build_model -> (train: lr/optim/wrapper) -> load_ckpt -> callback -> Model.train/eval`

## 2. 预训练（分布式脚本方式，推荐）

1) 选择一个预训练配置，例如：
- `config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml`

2) 使用分布式启动脚本（Linux）：

```bash
bash scripts/pretrain_distribute.sh <RANK_TABLE_JSON> <CONFIG_YAML>
```

日志通常在 `pretrain_parallel*/pretrain_log`。

## 3. 微调（分布式脚本方式）

```bash
bash scripts/finetune_distribute.sh <RANK_TABLE_JSON> <CONFIG_YAML>
```

微调相对预训练的主要差异：会同时构建 train/eval dataset，构建 loss 与 eval_engine。

## 4. 评估（分布式脚本或直接运行）

方式 A：分布式脚本（Linux）：

```bash
bash scripts/eval_distribute.sh <RANK_TABLE_JSON> <CONFIG_YAML>
```

方式 B：直接运行（适合单机调试；是否能在你的环境跑通取决于 MindSpore/Ascend 配置）：

```bash
python eval.py --config <CONFIG_YAML> --eval_path <CKPT_PATH>
```

## 5. 常用参数覆盖（不改 YAML）

三个入口都支持：

- `--config path/to/config.yaml`
- `--device_id N`
- `--seed N`
- `--use_parallel true/false`
- `--options a.b.c=value d.e=value`（把命令行键值覆盖合入配置）

微调额外常用：

- `--finetune_path <CKPT_PATH>`（写入 `train_config.resume_ckpt`）

评估额外常用：

- `--eval_path <CKPT_PATH>`（写入 `train_config.resume_ckpt`）
- `--batch_size N`

---

## 6. 产物在哪里

- 日志：由 `scripts/*_distribute.sh` 为每张卡创建目录并写入 `*_log`。
- Checkpoint：由回调（`ringmoe_framework/monitors/callback.py`）按配置保存。

如果你只想快速理解代码入口：直接从 `pretrain.py / finetune.py / eval.py` 的 `main()` 顺着调用链看即可。

