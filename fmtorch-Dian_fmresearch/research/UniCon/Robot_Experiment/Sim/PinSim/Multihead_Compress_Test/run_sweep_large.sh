#!/usr/bin/env bash
set -e

# ============================================================
# 大规模 Sweep (总 latent 400-700)
# 高保真重建，重视精度
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./results_large_latent}

echo "=========================================="
echo "Large Latent Sweep (latent 400-700)"
echo "=========================================="
echo "Data: $DATA"
echo "Output: $OUT"
echo ""

python3 sweep_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --Kq 256 384 512 \
  --Kdq 128 192 256 \
  --Ktau 128 192 256 \
  --latent-triplets \
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
  --num-workers 18 \
  --cpu-threshold 80.0 \
  --memory-threshold 80.0 \
  --batch-size 40

echo ""
echo "=========================================="
echo "Large Sweep Complete!"
echo "Results: $OUT/results_summary_multigroup.json"
echo "=========================================="

