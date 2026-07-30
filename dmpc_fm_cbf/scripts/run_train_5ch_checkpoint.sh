#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/nuplan_clean/bin/python}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_TRAJ="${N_TRAJ:-100}"
T_STEPS="${T_STEPS:-60}"
LR="${LR:-0.001}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_ROOT}/notebooks/cache/model_vel_5ch_canonical_r.pt}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/notebooks/cache}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python and rerun." >&2
  exit 1
fi

if [[ "${DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

args=(
  "${PROJECT_ROOT}/scripts/train_5ch_checkpoint.py"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --n "${N_TRAJ}"
  --t "${T_STEPS}"
  --lr "${LR}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --checkpoint "${CKPT_PATH}"
  --cache-dir "${CACHE_DIR}"
)

if [[ "${FORCE_REGENERATE}" == "1" ]]; then
  args+=(--force-regenerate)
fi

echo "[Run] ${PYTHON_BIN} ${args[*]}"
exec "${PYTHON_BIN}" "${args[@]}"
