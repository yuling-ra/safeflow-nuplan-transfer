#!/usr/bin/env bash
set -e

# ============================================================
# 完整 Sweep - 覆盖小中大全部规模
# 从 latent≈150 到 latent≈700 的完整探索
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./results_full_sweep_$(date +%Y%m%d_%H%M%S)}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${OUT}/sweep_${TIMESTAMP}.log"

mkdir -p "$OUT"

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOGFILE"
}

log "=========================================="
log "Full Multi-Head PCA Sweep"
log "=========================================="
log "Data: $DATA"
log "Output: $OUT"
log ""

START_TIME=$(date +%s)

python3 sweep_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --Kq 128 192 256 384 512 \
  --Kdq 64 96 128 192 256 \
  --Ktau 64 96 128 192 256 \
  --latent-triplets \
    "100,25,25" \
    "120,30,30" \
    "140,30,30" \
    "100,40,40" \
    "80,50,50" \
    "200,50,50" \
    "240,60,60" \
    "280,60,60" \
    "250,70,70" \
    "220,80,80" \
    "300,50,50" \
    "280,70,70" \
    "400,50,50" \
    "360,70,70" \
    "320,90,90" \
    "450,75,75" \
    "500,100,100" \
    "480,80,80" \
    "420,100,100" \
    "380,120,120" \
  --filters none \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --num-workers 4 \
  --cpu-threshold 60.0 \
  --memory-threshold 60.0 \
  --batch-size 50 \
  2>&1 | tee -a "$LOGFILE"

ELAPSED=$(($(date +%s) - START_TIME))

log ""
log "=========================================="
log "Full Sweep Complete!"
log "=========================================="
log "Time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
log "Results: $OUT/results_summary_multigroup.json"
log "Log: $LOGFILE"
log "=========================================="

