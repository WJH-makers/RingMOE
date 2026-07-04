#!/usr/bin/env bash
set -Eeuo pipefail

# If you want to see every executed command, set:
#   export RINGMOE_DEBUG=1
RINGMOE_DEBUG="${RINGMOE_DEBUG:-0}"
RINGMOE_PRINT_ALL_LOG_TAILS="${RINGMOE_PRINT_ALL_LOG_TAILS:-1}"
if [[ "${RINGMOE_DEBUG}" == "1" ]]; then
  set -x
fi

# End-to-end automation for a SINGLE A100 GPU:
# - create Python venv
# - install CUDA-enabled PyTorch + torchvision (best-effort, auto-detect cu118/cu121)
# - download CIFAR-10 dataset + export data.json
# - download reference model parameters (torchvision SwinV2-T)
# - run RingMoE PyTorch refactor training (SimMIM) on 1 GPU (no DeepSpeed)
#
# Usage (Linux):
#   bash one_click_a100_single.sh
#
# Optional overrides:
#   # Put everything on a large filesystem (recommended on clusters/containers):
#   RINGMOE_BASE_DIR=/data/ringmoe_a100
#   # Or override individual paths:
#   RINGMOE_VENV_DIR=.venv_a100
#   RINGMOE_VENV_SYSTEM_SITE_PACKAGES=1  # reuse system/conda site packages (avoid reinstalling torch)
#   RINGMOE_TORCH_CUDA=cu121   # or cu118
#   RINGMOE_WORK_DIR=runs/a100_single_cifar10
#   RINGMOE_DATA_DIR=data/cifar10
#   RINGMOE_SKIP_TORCH_INSTALL=1  # if your site already provides torch/torchvision
#   RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL=25  # disk preflight threshold (default: 25GB)
#   RINGMOE_AUTO_RELOCATE=1  # auto move venv/work/data to a larger mount if disk is too small (default: 1)
#   RINGMOE_TMP_DIR=/data/tmp  # where pip unpacks large wheels (defaults under work_dir)
#   RINGMOE_AUTO_USE_SYSTEM_TORCH=1  # if CUDA torch already exists, create venv with --system-site-packages (default: 1)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

_die_trap() {
  local rc=$?
  local lineno="${1:-unknown}"
  local cmd="${2:-unknown}"
  set +e
  trap - ERR

  echo "" >&2
  echo "[fatal] one_click_a100_single.sh failed (exit=${rc})" >&2
  echo "[fatal] line=${lineno}" >&2
  echo "[fatal] cmd=${cmd}" >&2
  echo "[fatal] repo=${ROOT_DIR}" >&2
  echo "[fatal] venv=${VENV_DIR:-<unset>}" >&2
  echo "[fatal] work_dir=${WORK_DIR_ABS:-${WORK_DIR:-<unset>}}" >&2
  echo "[fatal] data_dir=${DATA_DIR_ABS:-${DATA_DIR:-<unset>}}" >&2
  echo "[fatal] tmp_dir=${TMPDIR:-<unset>}" >&2
  echo "[fatal] torch_home=${TORCH_HOME:-<unset>}" >&2
  echo "[fatal] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "[fatal] env: RINGMOE_GPU_ID=${RINGMOE_GPU_ID:-<unset>} RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES=${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES:-<unset>}" >&2

  echo "" >&2
  echo "[fatal] system:" >&2
  (uname -a 2>/dev/null || true) >&2
  (python3 --version 2>/dev/null || true) >&2
  (command -v python3 >/dev/null 2>&1 && echo "[fatal] python3=$(command -v python3)" || true) >&2
  (command -v python >/dev/null 2>&1 && echo "[fatal] python=$(command -v python)" || true) >&2
  (command -v python >/dev/null 2>&1 && python -m pip -V || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true) >&2
  (command -v nvidia-smi >/dev/null 2>&1 && echo "" && echo "[fatal] nvidia-smi -L:" && nvidia-smi -L || true) >&2

  if declare -F _print_df_report >/dev/null 2>&1; then
    echo "" >&2
    _print_df_report >&2 || true
  fi

  local log_dirs=()
  if [[ -n "${WORK_DIR_ABS:-}" ]] && [[ -d "${WORK_DIR_ABS}/logs" ]]; then
    log_dirs+=("${WORK_DIR_ABS}/logs")
  fi
  if [[ -n "${ONECLICK_LOGS_DIR:-}" ]] && [[ -d "${ONECLICK_LOGS_DIR}" ]]; then
    log_dirs+=("${ONECLICK_LOGS_DIR}")
  fi

  local dir
  for dir in "${log_dirs[@]}"; do
    echo "" >&2
    echo "[fatal] logs directory: ${dir}" >&2
    (ls -la "${dir}" 2>/dev/null || true) >&2

    if [[ "${RINGMOE_PRINT_ALL_LOG_TAILS}" == "1" ]]; then
      local log_file
      for log_file in "${dir}"/*.log; do
        [[ -e "${log_file}" ]] || continue
        echo "" >&2
        echo "[fatal] tail -n 200 ${log_file}:" >&2
        (tail -n 200 "${log_file}" 2>/dev/null || true) >&2
      done
    else
      local latest_log=""
      latest_log="$(ls -1t "${dir}"/*.log 2>/dev/null | head -n1 || true)"
      if [[ -n "${latest_log}" ]]; then
        echo "" >&2
        echo "[fatal] tail -n 200 ${latest_log}:" >&2
        (tail -n 200 "${latest_log}" 2>/dev/null || true) >&2
      fi
    fi
  done

  echo "" >&2
  echo "[fatal] Tip: if this is a disk issue, set: export RINGMOE_BASE_DIR=/data/ringmoe_a100" >&2
  exit "${rc}"
}

trap '_die_trap ${LINENO} "${BASH_COMMAND}"' ERR

_user_set_venv_dir=0
_user_set_work_dir=0
_user_set_data_dir=0
if [[ -n "${RINGMOE_VENV_DIR:-}" ]]; then _user_set_venv_dir=1; fi
if [[ -n "${RINGMOE_WORK_DIR:-}" ]]; then _user_set_work_dir=1; fi
if [[ -n "${RINGMOE_DATA_DIR:-}" ]]; then _user_set_data_dir=1; fi

VENV_DIR="${RINGMOE_VENV_DIR:-${ROOT_DIR}/.venv_a100}"
WORK_DIR="${RINGMOE_WORK_DIR:-runs/a100_single_cifar10}"
DATA_DIR="${RINGMOE_DATA_DIR:-data/cifar10}"

RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL="${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL:-25}"
RINGMOE_AUTO_RELOCATE="${RINGMOE_AUTO_RELOCATE:-1}"

_is_int() { [[ "${1:-}" =~ ^[0-9]+$ ]]; }
if ! _is_int "${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL}"; then
  echo "[warn] RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL is not an int: ${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL}; using 25"
  RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL="25"
fi

_abs_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    echo "${p}"
  else
    echo "${ROOT_DIR}/${p}"
  fi
}

if [[ -n "${RINGMOE_BASE_DIR:-}" ]] && [[ "${_user_set_venv_dir}" == "0" ]] && [[ "${_user_set_work_dir}" == "0" ]] && [[ "${_user_set_data_dir}" == "0" ]]; then
  base_abs="$(_abs_path "${RINGMOE_BASE_DIR}")"
  VENV_DIR="${base_abs}/.venv_a100"
  WORK_DIR="${base_abs}/runs/a100_single_cifar10"
  DATA_DIR="${base_abs}/data/cifar10"
fi

_df_avail_kb() {
  # prints available KB for the filesystem that contains the given path
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
  df -h "$(_abs_path "${WORK_DIR}")" 2>/dev/null || true
  df -h "$(_abs_path "${DATA_DIR}")" 2>/dev/null || true
  df -h "${RINGMOE_TMP_DIR:-${TMPDIR:-/tmp}}" 2>/dev/null || true
  df -h "${HOME:-/}" 2>/dev/null || true
}

_pick_large_base_dir() {
  local req_kb="${1:-0}"
  local candidates=()

  if [[ -n "${RINGMOE_BASE_DIR:-}" ]]; then
    candidates+=("${RINGMOE_BASE_DIR}")
  fi

  # Priority order (more likely to be "big/persistent" on clusters first).
  candidates+=(
    "/data"
    "/workspace"
    "/mnt/data"
    "/scratch"
    "/ssd"
    "/var/tmp"
    "/tmp"
    "${HOME:-}"
  )

  local c
  for c in "${candidates[@]}"; do
    [[ -z "$c" ]] && continue
    [[ -d "$c" ]] || continue
    [[ -w "$c" ]] || continue
    local avail_kb
    avail_kb="$(_df_avail_kb "$c" || true)"
    [[ -z "${avail_kb}" ]] && continue
    if [[ "${req_kb}" -le 0 ]] || [[ "${avail_kb}" -ge "${req_kb}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  echo ""
}

_maybe_auto_relocate_for_disk() {
  if [[ "${RINGMOE_AUTO_RELOCATE}" != "1" ]]; then
    return 0
  fi
  if [[ "${_user_set_venv_dir}" == "1" ]] || [[ "${_user_set_work_dir}" == "1" ]] || [[ "${_user_set_data_dir}" == "1" ]]; then
    return 0
  fi
  if [[ "${RINGMOE_SKIP_TORCH_INSTALL:-}" == "1" ]]; then
    return 0
  fi

  local need_kb
  need_kb="$((RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL * 1024 * 1024))"

  local avail_kb
  avail_kb="$(_df_avail_kb "$(dirname "${VENV_DIR}")" || true)"
  [[ -z "${avail_kb}" ]] && return 0
  if [[ "${avail_kb}" -ge "${need_kb}" ]]; then
    return 0
  fi

  local avail_gb
  avail_gb="$(awk "BEGIN{printf \"%.1f\", ${avail_kb}/1024/1024}")"
  echo "[warn] Low disk space for CUDA torch install on $(dirname "${VENV_DIR}")"
  echo "       Need >= ${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL}GB free, but only ~${avail_gb}GB is available."

  local base
  base="$(_pick_large_base_dir "${need_kb}")"
  if [[ -z "${base}" ]]; then
    echo "[fail] No suitable large writable directory found to auto-relocate." >&2
    _print_df_report
    echo "Fix options:" >&2
    echo "- Point venv/work/data to a larger filesystem, e.g.:" >&2
    echo "    export RINGMOE_BASE_DIR=/data/ringmoe_a100" >&2
    echo "    # or:" >&2
    echo "    export RINGMOE_VENV_DIR=/data/.venv_a100" >&2
    echo "    export RINGMOE_WORK_DIR=/data/runs/a100_single_cifar10" >&2
    echo "    export RINGMOE_DATA_DIR=/data/cifar10" >&2
    echo "- Or (if your site already provides CUDA torch): export RINGMOE_SKIP_TORCH_INSTALL=1" >&2
    exit 3
  fi

  local auto_base
  if [[ -n "${RINGMOE_BASE_DIR:-}" ]] && [[ "${base}" == "${RINGMOE_BASE_DIR}" ]]; then
    auto_base="${base}"
  else
    local repo_base
    repo_base="$(basename "${ROOT_DIR}")"
    auto_base="${base}/ringmoe_a100_${repo_base}"
  fi

  echo "[info] Auto-relocating artifacts to: ${auto_base}"
  VENV_DIR="${auto_base}/.venv_a100"
  WORK_DIR="${auto_base}/runs/a100_single_cifar10"
  DATA_DIR="${auto_base}/data/cifar10"
}

_maybe_auto_relocate_for_disk

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
# Mirror all stdout/stderr into a single log while keeping console output.
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[info] repo: ${ROOT_DIR}"
echo "[info] venv: ${VENV_DIR}"
echo "[info] work_dir: ${WORK_DIR} (${WORK_DIR_ABS})"
echo "[info] data_dir: ${DATA_DIR} (${DATA_DIR_ABS})"
echo "[info] tmp_dir: ${TMPDIR}"
echo "[info] torch_home: ${TORCH_HOME}"
echo "[info] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "[info] ringmoe_gpu_id: ${RINGMOE_GPU_ID}"
echo "[info] ringmoe_respect_cuda_visible_devices: ${RINGMOE_RESPECT_CUDA_VISIBLE_DEVICES}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] nvidia-smi -L:"
  nvidia-smi -L || true
  if [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    echo "[info] selected GPU (${CUDA_VISIBLE_DEVICES}) info:"
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,uuid,memory.total --format=csv,noheader || true
  fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "[warn] Detected active conda env: ${CONDA_PREFIX}"
  echo "       Note: This script will NOT run 'conda deactivate' for you."
  echo "       If you see libtinfo/libstdc++/torch import errors, try running in a clean shell:"
  echo "         - (If you see: 'bash: ... libtinfo.so.6: no version information available', it's usually due to conda libs in LD_LIBRARY_PATH.)"
  echo "         - conda deactivate   (interactive shell command)"
  echo "         - or unset LD_LIBRARY_PATH (if it points to conda libs)"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[fail] python3 not found. Install Python 3.8+ (3.10+ recommended)." >&2
  exit 2
fi

RINGMOE_AUTO_USE_SYSTEM_TORCH="${RINGMOE_AUTO_USE_SYSTEM_TORCH:-1}"
if [[ ! -d "${VENV_DIR}" ]] && [[ -z "${RINGMOE_VENV_SYSTEM_SITE_PACKAGES:-}" ]] && [[ "${RINGMOE_AUTO_USE_SYSTEM_TORCH}" == "1" ]]; then
  has_cuda_torch="$(python3 - <<'PY'
ok = False
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except Exception:
    ok = False
else:
    import torch
    ok = bool(torch.cuda.is_available())
print("1" if ok else "0")
PY
)"
  if [[ "${has_cuda_torch}" == "1" ]]; then
    echo "[info] Detected CUDA-enabled torch/torchvision in current Python; creating venv with --system-site-packages."
    RINGMOE_VENV_SYSTEM_SITE_PACKAGES="1"
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  set +e
  if [[ "${RINGMOE_VENV_SYSTEM_SITE_PACKAGES:-}" == "1" ]]; then
    echo "[info] Creating venv with --system-site-packages (reuse existing site packages like torch)."
    python3 -m venv --system-site-packages "${VENV_DIR}"
  else
    python3 -m venv "${VENV_DIR}"
  fi
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail] Failed to create venv at: ${VENV_DIR}" >&2
    echo "Fix options:" >&2
    echo "- Ubuntu/Debian: sudo apt-get install -y python3-venv" >&2
    echo "- Or set RINGMOE_VENV_DIR to a writable path, or use your existing Python environment." >&2
    exit 2
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --no-cache-dir -U pip wheel setuptools

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
        "This usually means something is shadowing the stdlib module (e.g., a stray importlib.py in your working directory).\n"
        "Fix: remove/rename the conflicting module, then re-run."
    )
PY

detect_cuda_tag() {
  if [[ -n "${RINGMOE_TORCH_CUDA:-}" ]]; then
    echo "${RINGMOE_TORCH_CUDA}"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    # Default to cu121 when CUDA version can't be detected.
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
    # Fall back to cu118 for older drivers.
    echo "cu118"
  fi
}

TORCH_CUDA_TAG="$(detect_cuda_tag)"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

echo "[info] torch cuda tag: ${TORCH_CUDA_TAG}"
echo "[info] torch index url: ${TORCH_INDEX_URL}"

if [[ "${RINGMOE_SKIP_TORCH_INSTALL:-}" != "1" ]]; then
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

  if [[ "${need_install}" == "1" ]]; then
    # Disk preflight: CUDA torch wheels are huge (multi-GB). Fail-fast with a clear remedy.
    need_kb="$((RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL * 1024 * 1024))"
    avail_kb="$(_df_avail_kb "$(dirname "${VENV_DIR}")" || true)"
    if [[ -n "${avail_kb}" ]] && [[ "${avail_kb}" -lt "${need_kb}" ]]; then
      echo "[fail] Not enough free disk space to install CUDA torch wheels into: ${VENV_DIR}" >&2
      echo "       Need >= ${RINGMOE_MIN_FREE_GB_FOR_TORCH_INSTALL}GB free on that filesystem." >&2
      _print_df_report
      echo "Fix options:" >&2
      echo "- Put venv/work/data on a larger filesystem and re-run:" >&2
      echo "    export RINGMOE_BASE_DIR=/data/ringmoe_a100" >&2
      echo "    bash one_click_a100_single.sh" >&2
      echo "- Or reuse an existing CUDA torch from your site/conda:" >&2
      echo "    export RINGMOE_VENV_SYSTEM_SITE_PACKAGES=1" >&2
      echo "    # and/or: export RINGMOE_SKIP_TORCH_INSTALL=1" >&2
      exit 3
    fi

    set +e
    mkdir -p "${WORK_DIR_ABS}/logs" 2>/dev/null || true
    pip_log="${ONECLICK_LOGS_DIR}/00_pip_install_torch.log"
    echo "[info] pip install log: ${pip_log}"
    python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple torch torchvision 2>&1 | tee "${pip_log}"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[fail] Failed to install torch/torchvision from ${TORCH_INDEX_URL}" >&2
      echo "Fix options:" >&2
      echo "- Export RINGMOE_TORCH_CUDA=cu118 or cu121 and re-run." >&2
      echo "- Or set RINGMOE_SKIP_TORCH_INSTALL=1 and rely on your site-provided torch module." >&2
      echo "- Or install torch/torchvision manually using the official PyTorch selector, then re-run this script." >&2
      if [[ -f "${pip_log}" ]] && grep -qiE "No space left on device|Errno 28" "${pip_log}"; then
        echo "- Detected: disk full ([Errno 28] No space left on device)" >&2
        _print_df_report
        echo "  Fix (recommended):" >&2
        echo "  * move venv/work/data to a larger filesystem, e.g.:" >&2
        echo "      export RINGMOE_BASE_DIR=/data/ringmoe_a100" >&2
        echo "      bash one_click_a100_single.sh" >&2
        echo "  * or explicitly:" >&2
        echo "      export RINGMOE_VENV_DIR=/data/.venv_a100" >&2
        echo "      export RINGMOE_WORK_DIR=/data/runs/a100_single_cifar10" >&2
        echo "      export RINGMOE_DATA_DIR=/data/cifar10" >&2
        echo "  Cleanup:" >&2
        echo "  * remove partial venv: rm -rf ${VENV_DIR}" >&2
        echo "  * (optional) clear pip cache: rm -rf ~/.cache/pip (or: python -m pip cache purge)" >&2
      fi
      exit 3
    fi
  else
    echo "[info] torch/torchvision already installed with CUDA; skipping install."
  fi
else
  echo "[info] RINGMOE_SKIP_TORCH_INSTALL=1; skipping torch/torchvision install."
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
if not torch.cuda.is_available():
    raise SystemExit(
        "[fail] torch.cuda.is_available() is False.\n"
        "This usually means you installed CPU-only torch, or the NVIDIA driver is unavailable.\n"
        "Fix: reinstall CUDA-enabled torch/torchvision per the official selector, and ensure `nvidia-smi` works."
    )
print("[info] gpu:", torch.cuda.get_device_name(0))
print("[info] bf16 supported:", torch.cuda.is_bf16_supported())
PY

# Minimal deps for the refactor quickstart.
python -m pip install --no-cache-dir -U numpy pillow 2>&1 | tee "${ONECLICK_LOGS_DIR}/01_pip_install_min_deps.log"

python pytorch_refactor/quickstart_single_a100_cifar10.py \
  --work_dir "${WORK_DIR}" \
  --data_dir "${DATA_DIR}" \
  --clean \
  "$@" 2>&1 | tee "${ONECLICK_LOGS_DIR}/10_quickstart_wrapper.log"
