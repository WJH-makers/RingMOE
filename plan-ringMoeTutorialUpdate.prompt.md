# RingMoE Tutorial Expansion Plan

## 1. Objective
Update `TUTORIAL.md` to include comprehensive details about the RingMoE framework's internal components, derived from a deep dive into the `ringmoe_framework` directory.

## 2. Analysis of Framework Components

### 2.1 Architectures (`ringmoe_framework/arch`)
The framework supports multiple architectures via the `build_model` function:
- **MAE** (`mae`)
- **SimMIM** (`simmim`)
- **RingMo** (`ringmo_framework`)
- **RingMo MM** (`ringmo_mm`)
- **SimMIM MoE** (`simmim_moe`)
- **SimMIM Single MoE** (`simmim_single_moe`)

**Action**: Create a section detailing these architectures and how to select them in the config (`arch` field).

### 2.2 Loss Functions (`ringmoe_framework/loss`)
Supported loss functions identified in `loss.py`:
- **InfoNceLoss**: For contrastive learning.
- **CrossEntropySmooth**: Label smoothing.
- **CrossEntropySmoothMixup**: Mixup support.
- **CrossEntropyIgnore**: Ignore specific labels.
- **SoftTargetCrossEntropy**: For soft targets.

**Action**: Document the `loss_type` configuration and available options.

### 2.3 Optimization (`ringmoe_framework/optim`)
Supported optimizers:
- **SGD**
- **AdamW**
- **AdamWOP** (AdamWeightDecayOp)
- **FP32AdamWOP** (FP32StateAdamWeightDecay)

**Action**: Explain the `optimizer` config section, specifically `optim_name`.

### 2.4 Learning Rate Schedules (`ringmoe_framework/lr`)
Supported schedules:
- **cosine_decay**
- **warmup**
- **warmup_cosine_decay**
- **warmup_cosine_decay_simmim**
- **warmup_multistep_decay**

**Action**: Explain the `lr_schedule` config section and `lr_type`.

### 2.5 Monitoring & Checkpointing (`ringmoe_framework/monitors`, `ringmoe_framework/tools`)
- **Monitors**: `LossMonitor` (basic loss printing), `StateMonitor`, `CheckpointMointor`, `ProfileMonitor`.
- **Checkpointing**: `load_ckpt` function handles distributed checkpoint loading, including remapping for semi-auto parallel modes.

**Action**: Add a section on "Training Monitoring and Checkpoint Management".

## 3. Proposed Document Structure
1.  **Project Introduction** (Existing)
2.  **Environment** (Existing)
3.  **Project Structure** (Existing)
4.  **Data Preparation** (Existing)
5.  **Quick Start** (Existing)
6.  **Framework Deep Dive** (New)
    *   Supported Architectures
    *   Loss Functions
    *   Optimizers & Learning Rates
    *   Monitoring & Checkpointing
7.  **Configuration Guide** (Refined)
8.  **Downstream Tasks** (Existing)

