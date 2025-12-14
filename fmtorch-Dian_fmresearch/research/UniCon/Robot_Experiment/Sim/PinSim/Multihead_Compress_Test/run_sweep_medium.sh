#!/usr/bin/env bash
set -e

# ============================================================
# 中等规模 Sweep (总 latent 200-400)
# 平衡压缩率与重建精度
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./results_medium_latent}

echo "=========================================="
echo "Medium Latent Sweep (latent 200-400)"
echo "=========================================="
echo "Data: $DATA"
echo "Output: $OUT"
echo ""

python3 sweep_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --Kq 192 256 384 \
  --Kdq 96 128 192 \
  --Ktau 96 128 192 \
  --latent-triplets \
    "200,50,50" \
    "240,60,60" \
    "280,60,60" \
    "250,70,70" \
    "220,80,80" \
    "300,50,50" \
    "280,70,70" \
  --filters none \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --num-workers 16 \
  --cpu-threshold 80.0 \
  --memory-threshold 80.0 \
  --batch-size 30

echo ""
echo "=========================================="
echo "Medium Sweep Complete!"
echo "Results: $OUT/results_summary_multigroup.json"
echo "=========================================="

