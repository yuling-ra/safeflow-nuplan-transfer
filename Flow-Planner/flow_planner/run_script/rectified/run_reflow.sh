#!/usr/bin/env bash
set -euo pipefail

# This launcher is intentionally explicit: the original FM checkpoint is
# read-only input and all ReFlow artifacts go to a separate directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${SAFEFLOW_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../../../../safeflow-nuplan-transfer/dmpc_fm_cbf" 2>/dev/null && pwd || true)}"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/notebooks/cache/model_vel_5ch_canonical_r.pt}"
DATA="${DATA:-$PROJECT_ROOT/notebooks/cache/car_ring_track_oc_dataset_v1_v0rand_with_vel.npy}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/rectified/checkpoints}"
PAIRS="${PAIRS:-$ARTIFACT_DIR/reflow_round1_pairs.npz}"
OUTPUT="${OUTPUT:-$ARTIFACT_DIR/model_reflow_round1.pt}"
DEVICE="${DEVICE:-cpu}"

if [[ -z "$PROJECT_ROOT" || ! -f "$CHECKPOINT" || ! -f "$DATA" ]]; then
  printf 'Set SAFEFLOW_PROJECT_ROOT to the dmpc_fm_cbf project root.\n' >&2
  exit 2
fi

export PYTHONPATH="$PACKAGE_PARENT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_DIR"

"$PYTHON_BIN" -m rectified_flow.reflow_generate \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --output "$PAIRS" \
  --device "$DEVICE" \
  --cbf \
  --num-segments "${NUM_SEGMENTS:-8}" \
  --ode-method "${ODE_METHOD:-rk4}"

"$PYTHON_BIN" -m rectified_flow.train_reflow \
  --pairs "$PAIRS" \
  --init-checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-32}"

"$PYTHON_BIN" -m rectified_flow.eval_nfe \
  --checkpoint "$OUTPUT" \
  --data "$DATA" \
  --device "$DEVICE" \
  --steps ${EVAL_STEPS:-1 2 4 8 16}
