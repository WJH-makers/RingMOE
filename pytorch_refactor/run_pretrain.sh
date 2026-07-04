#!/usr/bin/env bash
set -Eeuo pipefail

# If you want to see every executed command, set:
#   export RINGMOE_DEBUG=1
RINGMOE_DEBUG="${RINGMOE_DEBUG:-0}"
if [[ "${RINGMOE_DEBUG}" == "1" ]]; then
  set -x
fi

# Launch script for NVIDIA A100/H100 clusters (PyTorch + DeepSpeed refactor).
#
# Usage:
#   bash pytorch_refactor/run_pretrain.sh /path/to/data.json [num_gpus] [-- extra train args]
#
# Examples:
#   # Single A100 (GPU0), MoE enabled (DeepSpeed):
#   bash pytorch_refactor/run_pretrain.sh /data/data.json 1 --epochs 10 --micro_batch 1 --output_dir runs/exp1/checkpoints
#
#   # Multi-GPU:
#   bash pytorch_refactor/run_pretrain.sh /data/data.json 8 --epochs 100 --micro_batch 2
#
# GPU selection:
# - When num_gpus==1, this script forces GPU0 by default:
#     export RINGMOE_GPU_ID=0
# - To respect a scheduler-provided CUDA_VISIBLE_DEVICES, set:
#     export RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

_abs_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    echo "${p}"
  else
    echo "${REPO_DIR}/${p}"
  fi
}

_die_trap() {
  local rc=$?
  local lineno="${1:-unknown}"
  local cmd="${2:-unknown}"
  set +e
  trap - ERR

  echo "" >&2
  echo "[fatal] run_pretrain.sh failed (exit=${rc})" >&2
  echo "[fatal] line=${lineno}" >&2
  echo "[fatal] cmd=${cmd}" >&2
  echo "[fatal] repo=${REPO_DIR}" >&2
  echo "[fatal] venv=${VENV_DIR:-<unset>}" >&2
  echo "[fatal] data_path=${DATA_PATH:-<unset>}" >&2
  echo "[fatal] num_gpus=${NUM_GPUS:-<unset>}" >&2
  echo "[fatal] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "[fatal] work_dir=${WORK_DIR_ABS:-<unset>}" >&2
  echo "[fatal] logs_dir=${LOGS_DIR:-<unset>}" >&2

  echo "" >&2
  echo "[fatal] system:" >&2
  (uname -a 2>/dev/null || true) >&2
  (python3 --version 2>/dev/null || true) >&2
  (command -v python >/dev/null 2>&1 && python -m pip -V || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true) >&2

  if [[ -n "${LOGS_DIR:-}" ]] && [[ -d "${LOGS_DIR}" ]]; then
    echo "" >&2
    echo "[fatal] logs directory: ${LOGS_DIR}" >&2
    (ls -la "${LOGS_DIR}" 2>/dev/null || true) >&2
    local latest_log=""
    latest_log="$(ls -1t "${LOGS_DIR}"/*.log 2>/dev/null | head -n1 || true)"
    if [[ -n "${latest_log}" ]]; then
      echo "" >&2
      echo "[fatal] tail -n 200 ${latest_log}:" >&2
      (tail -n 200 "${latest_log}" 2>/dev/null || true) >&2
    fi
  fi

  exit "${rc}"
}

trap '_die_trap ${LINENO} "${BASH_COMMAND}"' ERR

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash pytorch_refactor/run_pretrain.sh /path/to/data.json [num_gpus] [-- extra train args]"
  exit 2
fi

DATA_PATH="${1}"
shift

NUM_GPUS="1"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="${1}"
  shift
fi

# Default single-GPU selection (GPU0) only when num_gpus==1.
RINGMOE_GPU_ID="${RINGMOE_GPU_ID:-0}"
RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES="${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES:-0}"
if [[ "${NUM_GPUS}" == "1" ]]; then
  if [[ "${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES}" == "1" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[info] Respecting existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  else
    if [[ ! "${RINGMOE_GPU_ID}" =~ ^[0-9]+$ ]]; then
      echo "[warn] RINGMOE_GPU_ID is not an integer: ${RINGMOE_GPU_ID}; falling back to 0"
      RINGMOE_GPU_ID="0"
    fi
    export CUDA_VISIBLE_DEVICES="${RINGMOE_GPU_ID}"
  fi
else
  echo "[info] num_gpus=${NUM_GPUS} (multi-GPU): leaving CUDA_VISIBLE_DEVICES unchanged (${CUDA_VISIBLE_DEVICES:-<unset>})"
  echo "       Tip: set CUDA_VISIBLE_DEVICES explicitly if you want to pin/limit visible GPUs."
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[fail] data.json not found: ${DATA_PATH}" >&2
  echo "Expected a JSON list of image paths. See: RUNNING_LINUX_A100.md" >&2
  exit 2
fi

# Optional venv auto-activation.
# - If you used `one_click_a100_single.sh`, the venv is typically `.venv_a100` under the chosen base dir.
# - You can point to it explicitly:
#     export RINGMOE_VENV_DIR=/path/to/.venv_a100
VENV_DIR="${RINGMOE_VENV_DIR:-}"
if [[ -z "${VENV_DIR}" ]] && [[ -d "${REPO_DIR}/.venv_a100" ]]; then
  VENV_DIR="${REPO_DIR}/.venv_a100"
fi
if [[ -n "${VENV_DIR}" ]] && [[ -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[info] Activating venv: ${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

# Logs/work dir (wrapper logs, not DeepSpeed checkpoints).
WORK_DIR="${RINGMOE_WORK_DIR:-runs/a100_pretrain}"
WORK_DIR_ABS="$(_abs_path "${WORK_DIR}")"
LOGS_DIR="${WORK_DIR_ABS}/logs"
mkdir -p "${LOGS_DIR}" 2>/dev/null || true

MAIN_LOG="${LOGS_DIR}/00_run_pretrain_wrapper.log"
echo "[info] repo: ${REPO_DIR}"
echo "[info] data_path: ${DATA_PATH}"
echo "[info] num_gpus: ${NUM_GPUS}"
echo "[info] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[info] work_dir: ${WORK_DIR_ABS}"
echo "[info] main log: ${MAIN_LOG}"

# Mirror wrapper stdout/stderr to a file while keeping console output.
exec > >(tee -a "${MAIN_LOG}") 2>&1

# More actionable stack traces for Python/PyTorch errors.
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[fail] deepspeed not found in PATH." >&2
  echo "Fix options:" >&2
  echo "- Activate your venv first (the one with torch+deepspeed installed), then re-run." >&2
  echo "- Or install it: python -m pip install -U deepspeed" >&2
  echo "  (If you need MoE kernels: DS_BUILD_OPS=1 python -m pip install --no-cache-dir deepspeed)" >&2
  exit 2
fi

echo "[info] Running A100 self-check (CUDA + fwd/bwd)..."
python -m pytorch_refactor.a100_selfcheck 2>&1 | tee "${LOGS_DIR}/01_a100_selfcheck.log"

OUTPUT_DIR="${RINGMOE_OUTPUT_DIR:-checkpoints}"
EPOCHS="${RINGMOE_EPOCHS:-100}"
MOE_EXPERTS="${RINGMOE_MOE_EXPERTS:-8}"
SAVE_EVERY="${RINGMOE_SAVE_EVERY:-1}"

echo "[info] DeepSpeed train log: ${LOGS_DIR}/10_deepspeed_train.log"
deepspeed --num_gpus "${NUM_GPUS}" "${SCRIPT_DIR}/train.py" \
  --deepspeed_config "${SCRIPT_DIR}/ds_config.json" \
  --data_path "${DATA_PATH}" \
  --epochs "${EPOCHS}" \
  --moe_experts "${MOE_EXPERTS}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_every "${SAVE_EVERY}" \
  "$@" 2>&1 | tee "${LOGS_DIR}/10_deepspeed_train.log"

echo ""
echo "[SUCCESS] Training completed"
echo "checkpoints: ${OUTPUT_DIR}"
echo "logs: ${LOGS_DIR}"
