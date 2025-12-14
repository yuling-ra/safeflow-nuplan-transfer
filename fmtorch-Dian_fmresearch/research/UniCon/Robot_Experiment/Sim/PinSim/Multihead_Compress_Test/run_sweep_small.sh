#!/usr/bin/env bash
set -e

# ============================================================
# 小规模 Sweep (总 latent ≤ 200)
# 快速探索，低计算成本
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./results_small_latent}

echo "=========================================="
echo "Small Latent Sweep (latent ≤ 200)"
echo "=========================================="
echo "Data: $DATA"
echo "Output: $OUT"
echo ""

python3 sweep_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --Kq 128 192 256 \
  --Kdq 64 96 128 \
  --Ktau 64 96 128 \
  --latent-triplets \
    "100,25,25" \
    "120,30,30" \
    "140,30,30" \
    "100,40,40" \
    "80,50,50" \
  --filters none \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --num-workers 12 \
  --cpu-threshold 80.0 \
  --memory-threshold 80.0 \
  --batch-size 20

echo ""
echo "=========================================="
echo "Small Sweep Complete!"
echo "Results: $OUT/results_summary_multigroup.json"
echo "=========================================="

