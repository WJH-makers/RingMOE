#!/usr/bin/env bash
set -Eeuo pipefail

# If you want to see every executed command, set:
#   export RINGMOE_DEBUG=1
RINGMOE_DEBUG="${RINGMOE_DEBUG:-0}"
if [[ "${RINGMOE_DEBUG}" == "1" ]]; then
  set -x
fi

# One-command Linux quickstart wrapper.
# It creates a local venv (if missing), installs minimal deps, then runs:
#   python pytorch_refactor/quickstart_single_a100_cifar10.py
#
# IMPORTANT:
# - You must install a CUDA-enabled PyTorch + torchvision FIRST (per PyTorch official selector).
# - This wrapper will verify that `torch.cuda.is_available()` is True and fail with an actionable message otherwise.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Force single-GPU by default: A100 GPU0.
# - Override with: export RINGMOE_GPU_ID=0
# - If you want to respect a pre-set CUDA_VISIBLE_DEVICES (e.g. a scheduler allocation), set:
#     export RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES=1
RINGMOE_GPU_ID="${RINGMOE_GPU_ID:-0}"
RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES="${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES:-0}"
if [[ "${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES}" == "1" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[info] Respecting existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  if [[ ! "${RINGMOE_GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "[warn] RINGMOE_GPU_ID is not an integer: ${RINGMOE_GPU_ID}; falling back to 0"
    RINGMOE_GPU_ID="0"
  fi
  export CUDA_VISIBLE_DEVICES="${RINGMOE_GPU_ID}"
fi

# More actionable stack traces for Python/PyTorch errors.
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

_abs_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    echo "${p}"
  else
    echo "${ROOT_DIR}/${p}"
  fi
}

WORK_DIR_HINT="runs/a100_single_cifar10"
DATA_DIR_HINT="data/cifar10"
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "${arg}" in
    --work_dir)
      j=$((i + 1))
      if [[ $j -le $# ]]; then
        WORK_DIR_HINT="${!j}"
      fi
      ;;
    --work_dir=*)
      WORK_DIR_HINT="${arg#*=}"
      ;;
    --data_dir)
      j=$((i + 1))
      if [[ $j -le $# ]]; then
        DATA_DIR_HINT="${!j}"
      fi
      ;;
    --data_dir=*)
      DATA_DIR_HINT="${arg#*=}"
      ;;
  esac
done

WORK_DIR_ABS="$(_abs_path "${WORK_DIR_HINT}")"
LOGS_DIR="${WORK_DIR_ABS}/logs"

WRAPPER_LOGS_DIR="${WORK_DIR_ABS}.oneclick_logs"
mkdir -p "${WRAPPER_LOGS_DIR}" 2>/dev/null || true

MAIN_LOG="${WRAPPER_LOGS_DIR}/00_quickstart_sh_wrapper.log"
echo "[info] main log: ${MAIN_LOG}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

_die_trap() {
  local rc=$?
  local lineno="${1:-unknown}"
  local cmd="${2:-unknown}"
  set +e
  trap - ERR

  echo "" >&2
  echo "[fatal] quickstart_single_a100_cifar10.sh failed (exit=${rc})" >&2
  echo "[fatal] line=${lineno}" >&2
  echo "[fatal] cmd=${cmd}" >&2
  echo "[fatal] repo=${ROOT_DIR}" >&2
  echo "[fatal] work_dir=${WORK_DIR_HINT} (${WORK_DIR_ABS})" >&2
  echo "[fatal] data_dir=${DATA_DIR_HINT}" >&2
  echo "[fatal] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2

  echo "" >&2
  echo "[fatal] system:" >&2
  (uname -a 2>/dev/null || true) >&2
  (python3 --version 2>/dev/null || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true) >&2

  if [[ -d "${LOGS_DIR}" ]]; then
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
  if [[ -d "${WRAPPER_LOGS_DIR}" ]]; then
    echo "" >&2
    echo "[fatal] wrapper logs directory: ${WRAPPER_LOGS_DIR}" >&2
    (ls -la "${WRAPPER_LOGS_DIR}" 2>/dev/null || true) >&2
    local latest_wrap=""
    latest_wrap="$(ls -1t "${WRAPPER_LOGS_DIR}"/*.log 2>/dev/null | head -n1 || true)"
    if [[ -n "${latest_wrap}" ]]; then
      echo "" >&2
      echo "[fatal] tail -n 200 ${latest_wrap}:" >&2
      (tail -n 200 "${latest_wrap}" 2>/dev/null || true) >&2
    fi
  fi

  exit "${rc}"
}

trap '_die_trap ${LINENO} "${BASH_COMMAND}"' ERR

VENV_DIR="${ROOT_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip >/dev/null

python - <<'PY'
try:
    import torch
except Exception as e:
    raise SystemExit(
        "PyTorch is not installed in this venv.\n"
        "Install CUDA-enabled torch + torchvision using the official PyTorch selector, then re-run.\n"
        f"Root cause: {type(e).__name__}: {e}"
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch CUDA is NOT available.\n"
        f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}\n"
        "Fix: reinstall CUDA-enabled torch+torchvision per the official PyTorch selector, and ensure `nvidia-smi` works.\n"
    )

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
PY

# Minimal deps for this repo's PyTorch refactor quickstart.
python -m pip install -U numpy pillow >/dev/null

python pytorch_refactor/quickstart_single_a100_cifar10.py --amp --tf32 --use_checkpoint "$@"
