#!/usr/bin/env bash
set -Eeuo pipefail

# One-click single-A100 (GPU0) pretraining on YOUR single-modal dataset (not CIFAR).
#
# What it does:
# - (optional) create venv + install CUDA PyTorch/torchvision
# - install PyTorch-refactor deps (incl. DeepSpeed, best-effort)
# - (optional) clone the RingMoEDatasets repo (metadata/README only; actual data download depends on that repo)
# - generate data.json from a single-modality image directory
# - (optional) download a reference torchvision checkpoint (SwinV2-T) as a reproducible "model params download" step
# - run a dry-run then training on 1 GPU
#
# Usage:
#   export RINGMOE_DATA_ROOT=/data/your_images
#   export RINGMOE_BASE_DIR=/data/ringmoe_a100   # recommended (persistent + enough space)
#   bash one_click_a100_single_realdata.sh
#
# Notes:
# - This uses the repo's PyTorch/DeepSpeed refactor under `pytorch_refactor/`.
# - The original RingMoE paper's 14.7B scale is NOT feasible on a single A100; this script is for correctness + workflow.

RINGMOE_DEBUG="${RINGMOE_DEBUG:-0}"
if [[ "${RINGMOE_DEBUG}" == "1" ]]; then
  set -x
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Force single-GPU by default: A100 GPU0.
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

_df_avail_kb() {
  local p="$1"
  local q="$p"
  while [[ ! -e "$q" ]]; do
    q="$(dirname "$q")"
    if [[ "$q" == "/" ]]; then
      break
    fi
  done
  df -Pk "$q" 2>/dev/null | awk 'NR==2 {print $4}'
}

_print_df_report() {
  echo "[info] disk report (df -h):"
  df -h "${ROOT_DIR}" 2>/dev/null || true
  df -h "$(dirname "${VENV_DIR}")" 2>/dev/null || true
  df -h "${WORK_DIR_ABS}" 2>/dev/null || true
  df -h "${DATA_DIR_ABS}" 2>/dev/null || true
  df -h "${RINGMOE_TMP_DIR:-${TMPDIR:-/tmp}}" 2>/dev/null || true
  df -h "${HOME:-/}" 2>/dev/null || true
}

_die_trap() {
  local rc=$?
  local lineno="${1:-unknown}"
  local cmd="${2:-unknown}"
  set +e
  trap - ERR

  echo "" >&2
  echo "[fatal] one_click_a100_single_realdata.sh failed (exit=${rc})" >&2
  echo "[fatal] line=${lineno}" >&2
  echo "[fatal] cmd=${cmd}" >&2
  echo "[fatal] repo=${ROOT_DIR}" >&2
  echo "[fatal] venv=${VENV_DIR:-<unset>}" >&2
  echo "[fatal] work_dir=${WORK_DIR_ABS:-${WORK_DIR:-<unset>}}" >&2
  echo "[fatal] data_dir=${DATA_DIR_ABS:-${DATA_DIR:-<unset>}}" >&2
  echo "[fatal] data_root=${RINGMOE_DATA_ROOT:-<unset>}" >&2
  echo "[fatal] data_json=${DATA_JSON:-<unset>}" >&2
  echo "[fatal] tmp_dir=${TMPDIR:-<unset>}" >&2
  echo "[fatal] torch_home=${TORCH_HOME:-<unset>}" >&2
  echo "[fatal] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2

  echo "" >&2
  echo "[fatal] system:" >&2
  (uname -a 2>/dev/null || true) >&2
  (python3 --version 2>/dev/null || true) >&2
  (command -v python3 >/dev/null 2>&1 && echo "[fatal] python3=$(command -v python3)" || true) >&2
  (command -v python >/dev/null 2>&1 && echo "[fatal] python=$(command -v python)" || true) >&2
  (command -v python >/dev/null 2>&1 && python -m pip -V || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && echo "" && echo "[fatal] nvidia-smi -L:" && nvidia-smi -L || true) >&2

  echo "" >&2
  _print_df_report >&2 || true

  local dirs=()
  if [[ -n "${WORK_DIR_ABS:-}" ]] && [[ -d "${WORK_DIR_ABS}/logs" ]]; then
    dirs+=("${WORK_DIR_ABS}/logs")
  fi
  if [[ -n "${ONECLICK_LOGS_DIR:-}" ]] && [[ -d "${ONECLICK_LOGS_DIR}" ]]; then
    dirs+=("${ONECLICK_LOGS_DIR}")
  fi

  local d
  for d in "${dirs[@]}"; do
    echo "" >&2
    echo "[fatal] logs directory: ${d}" >&2
    (ls -la "${d}" 2>/dev/null || true) >&2
    local latest=""
    latest="$(ls -1t "${d}"/*.log 2>/dev/null | head -n1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "" >&2
      echo "[fatal] tail -n 200 ${latest}:" >&2
      (tail -n 200 "${latest}" 2>/dev/null || true) >&2
    fi
  done

  exit "${rc}"
}

trap '_die_trap ${LINENO} "${BASH_COMMAND}"' ERR

# ---------------------------
# Paths / knobs
# ---------------------------

RINGMOE_BASE_DIR="${RINGMOE_BASE_DIR:-}"
VENV_DIR="${RINGMOE_VENV_DIR:-${ROOT_DIR}/.venv_a100}"
WORK_DIR="${RINGMOE_WORK_DIR:-runs/a100_single_realdata}"
DATA_DIR="${RINGMOE_DATA_DIR:-data/realdata_single}"

if [[ -n "${RINGMOE_BASE_DIR}" ]] && [[ -z "${RINGMOE_VENV_DIR:-}" ]] && [[ -z "${RINGMOE_WORK_DIR:-}" ]] && [[ -z "${RINGMOE_DATA_DIR:-}" ]]; then
  base_abs="$(_abs_path "${RINGMOE_BASE_DIR}")"
  VENV_DIR="${base_abs}/.venv_a100"
  WORK_DIR="${base_abs}/runs/a100_single_realdata"
  DATA_DIR="${base_abs}/data/realdata_single"
fi

WORK_DIR_ABS="$(_abs_path "${WORK_DIR}")"
DATA_DIR_ABS="$(_abs_path "${DATA_DIR}")"

if [[ -z "${TORCH_HOME:-}" ]]; then
  export TORCH_HOME="${WORK_DIR_ABS}/torch_cache"
fi
if [[ -n "${RINGMOE_TMP_DIR:-}" ]]; then
  export TMPDIR="${RINGMOE_TMP_DIR}"
elif [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${WORK_DIR_ABS}/tmp"
fi
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
  export XDG_CACHE_HOME="${WORK_DIR_ABS}/xdg_cache"
fi
if [[ -z "${PIP_CACHE_DIR:-}" ]]; then
  export PIP_CACHE_DIR="${WORK_DIR_ABS}/pip_cache"
fi

mkdir -p "${WORK_DIR_ABS}" "${DATA_DIR_ABS}" "${TORCH_HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" 2>/dev/null || true
mkdir -p "${WORK_DIR_ABS}/logs" 2>/dev/null || true

ONECLICK_LOGS_DIR="${WORK_DIR_ABS}.oneclick_logs"
mkdir -p "${ONECLICK_LOGS_DIR}" 2>/dev/null || true

MAIN_LOG="${ONECLICK_LOGS_DIR}/00_one_click_main.log"
echo "[info] oneclick logs: ${ONECLICK_LOGS_DIR}"
echo "[info] main log: ${MAIN_LOG}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[info] repo: ${ROOT_DIR}"
echo "[info] venv: ${VENV_DIR}"
echo "[info] work_dir: ${WORK_DIR} (${WORK_DIR_ABS})"
echo "[info] data_dir: ${DATA_DIR} (${DATA_DIR_ABS})"
echo "[info] tmp_dir: ${TMPDIR}"
echo "[info] torch_home: ${TORCH_HOME}"
echo "[info] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] nvidia-smi -L:"
  nvidia-smi -L || true
  if [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    echo "[info] selected GPU (${CUDA_VISIBLE_DEVICES}) info:"
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,uuid,memory.total --format=csv,noheader || true
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[fail] python3 not found. Install Python 3.8+ (3.10+ recommended)." >&2
  exit 2
fi

# ---------------------------
# venv
# ---------------------------

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[info] Creating venv: ${VENV_DIR}"
  if [[ "${RINGMOE_VENV_SYSTEM_SITE_PACKAGES:-}" == "1" ]]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
  else
    python3 -m venv "${VENV_DIR}"
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --no-cache-dir -U pip wheel setuptools 2>&1 | tee "${ONECLICK_LOGS_DIR}/01_pip_bootstrap.log"

python - <<'PY'
try:
    import importlib.util  # noqa: F401
except Exception as e:
    import importlib
    import sys
    raise SystemExit(
        "[fail] Python importlib is broken (cannot import importlib.util).\n"
        f"Root cause: {type(e).__name__}: {e}\n"
        f"importlib loaded from: {getattr(importlib, '__file__', '<built-in>')}\n"
        f"sys.path: {sys.path}\n"
        "This usually means something is shadowing the stdlib module (e.g., a stray importlib.py).\n"
        "Fix: remove/rename the conflicting module, then re-run."
    )
PY

# ---------------------------
# Torch install (best-effort)
# ---------------------------

detect_cuda_tag() {
  if [[ -n "${RINGMOE_TORCH_CUDA:-}" ]]; then
    echo "${RINGMOE_TORCH_CUDA}"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cu121"
    return 0
  fi
  local cuda_ver
  cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
  if [[ -z "${cuda_ver}" ]]; then
    echo "cu121"
    return 0
  fi
  local major minor
  major="${cuda_ver%%.*}"
  minor="${cuda_ver#*.}"
  if [[ "${major}" -gt 12 ]] || ([[ "${major}" -eq 12 ]] && [[ "${minor}" -ge 1 ]]); then
    echo "cu121"
  else
    echo "cu118"
  fi
}

TORCH_CUDA_TAG="$(detect_cuda_tag)"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "[info] torch cuda tag: ${TORCH_CUDA_TAG}"
echo "[info] torch index url: ${TORCH_INDEX_URL}"

RINGMOE_SKIP_TORCH_INSTALL="${RINGMOE_SKIP_TORCH_INSTALL:-0}"
RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL="${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL:-25}"

need_install="$(python - <<'PY'
ready = True
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except Exception:
    ready = False
else:
    import torch
    ready = bool(torch.cuda.is_available())
print("0" if ready else "1")
PY
)"

if [[ "${RINGMOE_SKIP_TORCH_INSTALL}" != "1" ]] && [[ "${need_install}" == "1" ]]; then
  need_kb="$((RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL * 1024 * 1024))"
  avail_kb="$(_df_avail_kb "$(dirname "${VENV_DIR}")" || true)"
  if [[ -n "${avail_kb}" ]] && [[ "${avail_kb}" -lt "${need_kb}" ]]; then
    echo "[fail] Not enough free disk space to install CUDA torch wheels into: ${VENV_DIR}" >&2
    echo "       Need >= ${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL}GB free on that filesystem." >&2
    _print_df_report
    echo "Fix: set a big persistent base dir, e.g.:" >&2
    echo "  export RINGMOE_BASE_DIR=/data/ringmoe_a100" >&2
    exit 3
  fi

  pip_log="${ONECLICK_LOGS_DIR}/02_pip_install_torch.log"
  echo "[info] pip install log: ${pip_log}"
  python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple torch torchvision 2>&1 | tee "${pip_log}"
fi

python - <<'PY'
import os
import torch
print("[info] torch:", torch.__version__)
print("[info] torch.version.cuda:", torch.version.cuda)
print("[info] cuda available:", torch.cuda.is_available())
print("[info] env CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[info] device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[info] current_device:", torch.cuda.current_device())
    print("[info] gpu0:", torch.cuda.get_device_name(0))
    print("[info] bf16 supported:", torch.cuda.is_bf16_supported())
else:
    raise SystemExit(
        "[fail] torch.cuda.is_available() is False.\n"
        "This usually means you installed CPU-only torch, or the NVIDIA driver is unavailable.\n"
        "Fix: reinstall CUDA-enabled torch/torchvision and ensure `nvidia-smi` works."
    )
PY

# ---------------------------
# Install repo deps (PyTorch refactor)
# ---------------------------

deps_log="${ONECLICK_LOGS_DIR}/03_pip_install_repo_deps.log"
echo "[info] installing pytorch_refactor deps (log: ${deps_log})"
python -m pip install --no-cache-dir -U timm einops numpy Pillow scipy 2>&1 | tee "${deps_log}"

# Optional: DeepSpeed (required for MoE training path).
RINGMOE_USE_DEEPSPEED="${RINGMOE_USE_DEEPSPEED:-1}"
if [[ "${RINGMOE_USE_DEEPSPEED}" == "1" ]]; then
  ds_log="${ONECLICK_LOGS_DIR}/03a_pip_install_deepspeed.log"
  echo "[info] installing deepspeed (log: ${ds_log})"
  set +e
  python -m pip install --no-cache-dir -U "deepspeed>=0.10.0" 2>&1 | tee "${ds_log}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[warn] deepspeed install failed; falling back to --no_deepspeed (MoE will be disabled)." >&2
    echo "       If you want MoE on A100, install deepspeed with CUDA ops in your environment, then rerun." >&2
    RINGMOE_USE_DEEPSPEED="0"
  fi
fi

# ---------------------------
# Self-check (CUDA + model fwd/bwd)
# ---------------------------

echo "[info] running A100 self-check (CUDA + model fwd/bwd)..."
python -m pytorch_refactor.a100_selfcheck 2>&1 | tee "${ONECLICK_LOGS_DIR}/03b_a100_selfcheck.log"

# ---------------------------
# (Optional) dataset repo clone (metadata/README)
# ---------------------------

RINGMOE_CLONE_DATASETS_REPO="${RINGMOE_CLONE_DATASETS_REPO:-0}"
if [[ "${RINGMOE_CLONE_DATASETS_REPO}" == "1" ]]; then
  RINGMOE_DATASETS_REPO_URL="${RINGMOE_DATASETS_REPO_URL:-https://github.com/HanboBizl/RingMoEDatasets.git}"
  datasets_dir="${WORK_DIR_ABS}/RingMoEDatasets"
  if [[ ! -d "${datasets_dir}/.git" ]]; then
    echo "[info] cloning datasets repo: ${RINGMOE_DATASETS_REPO_URL} -> ${datasets_dir}"
    git clone --depth 1 "${RINGMOE_DATASETS_REPO_URL}" "${datasets_dir}" 2>&1 | tee "${ONECLICK_LOGS_DIR}/04_git_clone_datasets_repo.log"
  else
    echo "[info] datasets repo already exists: ${datasets_dir}"
  fi
  echo "[info] Note: this repo may not include the full dataset bytes; follow its README to download/prepare RingMOSS."
fi

# ---------------------------
# Build data.json (single-modal)
# ---------------------------

if [[ -z "${RINGMOE_DATA_ROOT:-}" ]]; then
  echo "[fail] Please set RINGMOE_DATA_ROOT to your image directory (single-modal)." >&2
  echo "Example:" >&2
  echo "  export RINGMOE_DATA_ROOT=/data/ringmoss/images" >&2
  exit 2
fi

DATA_JSON="${RINGMOE_DATA_JSON:-${DATA_DIR_ABS}/data.json}"
RINGMOE_DATA_EXTS="${RINGMOE_DATA_EXTS:-jpg,jpeg,png,tif,tiff,npy,npz}"
RINGMOE_DATA_LIMIT="${RINGMOE_DATA_LIMIT:-0}"

echo "[info] building data.json from: ${RINGMOE_DATA_ROOT}"
echo "[info] data.json: ${DATA_JSON}"
python - <<PY 2>&1 | tee "${ONECLICK_LOGS_DIR}/05_make_data_json.log"
import json, os
from pathlib import Path

root = Path(${RINGMOE_DATA_ROOT@Q}).expanduser().resolve()
out = Path(${DATA_JSON@Q}).expanduser().resolve()
exts = [e.strip().lower().lstrip(".") for e in ${RINGMOE_DATA_EXTS@Q}.split(",") if e.strip()]
limit = int(${RINGMOE_DATA_LIMIT@Q})

if not root.exists() or not root.is_dir():
    raise SystemExit(f"[fail] RINGMOE_DATA_ROOT is not a directory: {root}")

paths = []
for ext in exts:
    paths.extend(root.rglob(f"*.{ext}"))

paths = sorted({p.resolve() for p in paths})
if limit > 0:
    paths = paths[:limit]

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump([str(p) for p in paths], f, ensure_ascii=False)

print("[info] files:", len(paths))
if len(paths) == 0:
    raise SystemExit("[fail] No files matched. Check RINGMOE_DATA_ROOT / RINGMOE_DATA_EXTS.")
print("[ok] wrote:", str(out))
PY

# ---------------------------
# (Optional) model params download (repro step)
# ---------------------------

RINGMOE_SKIP_MODEL_PARAMS_DOWNLOAD="${RINGMOE_SKIP_MODEL_PARAMS_DOWNLOAD:-0}"
if [[ "${RINGMOE_SKIP_MODEL_PARAMS_DOWNLOAD}" != "1" ]]; then
  out_params="${WORK_DIR_ABS}/model_params/torchvision_swin_v2_t.pt"
  echo "[info] downloading reference model params (torchvision SwinV2-T) -> ${out_params}"
  python pytorch_refactor/download_torchvision_swinv2_t.py --out "${out_params}" 2>&1 | tee "${ONECLICK_LOGS_DIR}/06_download_model_params.log"
else
  echo "[info] RINGMOE_SKIP_MODEL_PARAMS_DOWNLOAD=1; skipping model params download."
fi

# ---------------------------
# Train (single GPU)
# ---------------------------

RINGMOE_INPUT_SIZE="${RINGMOE_INPUT_SIZE:-192}"
RINGMOE_MASK_RATIO="${RINGMOE_MASK_RATIO:-0.6}"
RINGMOE_NUM_WORKERS="${RINGMOE_NUM_WORKERS:-4}"
RINGMOE_EPOCHS="${RINGMOE_EPOCHS:-1}"
RINGMOE_LOG_EVERY="${RINGMOE_LOG_EVERY:-10}"
RINGMOE_SEED="${RINGMOE_SEED:-42}"

RINGMOE_USE_DEEPSPEED="${RINGMOE_USE_DEEPSPEED:-1}"
RINGMOE_MICRO_BATCH="${RINGMOE_MICRO_BATCH:-1}"
RINGMOE_BATCH_SIZE="${RINGMOE_BATCH_SIZE:-1}"
RINGMOE_MOE_EXPERTS="${RINGMOE_MOE_EXPERTS:-8}"
RINGMOE_DISABLE_MOE="${RINGMOE_DISABLE_MOE:-0}"

OUTPUT_DIR="${RINGMOE_OUTPUT_DIR:-${WORK_DIR_ABS}/checkpoints}"
RINGMOE_RESUME_FROM="${RINGMOE_RESUME_FROM:-}"
RINGMOE_RESUME_TAG="${RINGMOE_RESUME_TAG:-}"
RINGMOE_EXPORT_PT="${RINGMOE_EXPORT_PT:-0}"

echo "[info] training output_dir: ${OUTPUT_DIR}"

echo ""
echo "================================================================================"
echo "[STEP] Dry-run (data + shapes)"
echo "================================================================================"
python train_a100.py \
  --data_path "${DATA_JSON}" \
  --no_deepspeed \
  --dry_run \
  --num_workers 0 \
  --input_size "${RINGMOE_INPUT_SIZE}" \
  --mask_ratio "${RINGMOE_MASK_RATIO}" \
  --amp \
  --tf32 \
  --use_checkpoint 2>&1 | tee "${ONECLICK_LOGS_DIR}/10_dry_run.log"

if [[ "${RINGMOE_USE_DEEPSPEED}" == "1" ]]; then
  if ! command -v deepspeed >/dev/null 2>&1; then
    echo "[fail] deepspeed not found (but RINGMOE_USE_DEEPSPEED=1)." >&2
    echo "Fix: ensure pytorch_refactor/requirements.txt installed deepspeed successfully, or set RINGMOE_USE_DEEPSPEED=0." >&2
    exit 2
  fi

  ds_cmd=(
    deepspeed --num_gpus 1 train_a100.py
    --deepspeed_config pytorch_refactor/ds_config.json
    --data_path "${DATA_JSON}"
    --epochs "${RINGMOE_EPOCHS}"
    --micro_batch "${RINGMOE_MICRO_BATCH}"
    --num_workers "${RINGMOE_NUM_WORKERS}"
    --input_size "${RINGMOE_INPUT_SIZE}"
    --mask_ratio "${RINGMOE_MASK_RATIO}"
    --seed "${RINGMOE_SEED}"
    --moe_experts "${RINGMOE_MOE_EXPERTS}"
    --output_dir "${OUTPUT_DIR}"
    --save_every 1
    --log_every "${RINGMOE_LOG_EVERY}"
    --tf32
    --use_checkpoint
  )
  if [[ "${RINGMOE_DISABLE_MOE}" == "1" ]]; then
    ds_cmd+=(--disable_moe)
  fi
  if [[ -n "${RINGMOE_RESUME_FROM}" ]]; then
    ds_cmd+=(--resume_from "${RINGMOE_RESUME_FROM}")
  fi
  if [[ -n "${RINGMOE_RESUME_TAG}" ]]; then
    ds_cmd+=(--resume_tag "${RINGMOE_RESUME_TAG}")
  fi

  echo ""
  echo "================================================================================"
  echo "[STEP] Train (single GPU, DeepSpeed)"
  echo "================================================================================"
  echo "[cmd] ${ds_cmd[*]}"
  "${ds_cmd[@]}" "$@" 2>&1 | tee "${ONECLICK_LOGS_DIR}/11_train_deepspeed.log"

  if [[ "${RINGMOE_EXPORT_PT}" == "1" ]]; then
    last_epoch=$((RINGMOE_EPOCHS - 1))
    if [[ "${last_epoch}" -lt 0 ]]; then
      last_epoch=0
    fi
    tag="epoch_${last_epoch}"
    out_pt="${WORK_DIR_ABS}/model.pt"
    echo "[info] exporting consolidated .pt from DeepSpeed checkpoint (tag=${tag}) -> ${out_pt}"
    export_cmd=(
      python pytorch_refactor/export_pt.py
      --ds_ckpt "${OUTPUT_DIR}"
      --tag "${tag}"
      --out "${out_pt}"
      --moe_experts "${RINGMOE_MOE_EXPERTS}"
      --input_size "${RINGMOE_INPUT_SIZE}"
      --deepspeed_config pytorch_refactor/ds_config.json
      --deepspeed
    )
    if [[ "${RINGMOE_DISABLE_MOE}" == "1" ]]; then
      export_cmd+=(--disable_moe)
    fi
    echo "[cmd] ${export_cmd[*]}"
    "${export_cmd[@]}" 2>&1 | tee "${ONECLICK_LOGS_DIR}/12_export_pt.log"
  fi
else
  echo ""
  echo "================================================================================"
  echo "[STEP] Train (single GPU, no DeepSpeed)"
  echo "================================================================================"
  python train_a100.py \
    --data_path "${DATA_JSON}" \
    --no_deepspeed \
    --epochs "${RINGMOE_EPOCHS}" \
    --batch_size "${RINGMOE_BATCH_SIZE}" \
    --num_workers "${RINGMOE_NUM_WORKERS}" \
    --log_every "${RINGMOE_LOG_EVERY}" \
    --seed "${RINGMOE_SEED}" \
    --input_size "${RINGMOE_INPUT_SIZE}" \
    --mask_ratio "${RINGMOE_MASK_RATIO}" \
    --output_dir "${OUTPUT_DIR}" \
    --amp \
    --tf32 \
    --use_checkpoint \
    "$@" 2>&1 | tee "${ONECLICK_LOGS_DIR}/11_train_nodeepspeed.log"
fi

echo ""
echo "================================================================================"
echo "[SUCCESS] Training completed"
echo "================================================================================"
echo "data.json: ${DATA_JSON}"
echo "checkpoints: ${OUTPUT_DIR}"
echo "logs: ${ONECLICK_LOGS_DIR}"
