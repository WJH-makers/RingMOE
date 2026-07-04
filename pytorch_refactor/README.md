# RingMoE PyTorch Refactor

This directory contains the refactored RingMoE framework using PyTorch and DeepSpeed, optimized for NVIDIA A100/H100 clusters.

## Prerequisites

- NVIDIA GPU(s) (A100 or H100 recommended)
- CUDA 11.8 or 12.x
- Python 3.8+
- PyTorch 2.0+
- DeepSpeed

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset JSON file containing a list of image paths.
   Example `data.json`:
   ```json
   ["/path/to/image1.jpg", "/path/to/image2.jpg"]
   ```

   Multi-modal example (`modal_num=4`):
   ```json
   [
     ["/data/opt/0001.jpg", "/data/sar/0001.npy", "/data/ms/0001.jpg", "/data/hsi/0001.npy"],
     ["/data/opt/0002.jpg", "/data/sar/0002.npy", "/data/ms/0002.jpg", "/data/hsi/0002.npy"]
   ]
   ```

2. Configure DeepSpeed in `ds_config.json` if needed.
   - `bf16` is enabled by default for A100/H100.

3. Run pretraining:

```bash
bash run_pretrain.sh /path/to/data.json 1 --epochs 10 --micro_batch 1 --output_dir runs/exp1/checkpoints

# By default (num_gpus==1), the script uses GPU0.
# To change it:
#   RINGMOE_GPU_ID=0 bash run_pretrain.sh /path/to/data.json 1 ...
```

## Architecture

- **Model**: Swin Transformer V2 with Mixture-of-Experts (MoE).
- **Pretraining**: SimMIM (Masked Image Modeling).
- **Parallelism**: DeepSpeed ZeRO + MoE parallelism.

## Files

- `dataset.py`: Data loading and SimMIM masking.
- `model.py`: Model definitions (SwinV2, MoE, SimMIM).
- `train.py`: Main training script.
- `ds_config.json`: DeepSpeed configuration.

