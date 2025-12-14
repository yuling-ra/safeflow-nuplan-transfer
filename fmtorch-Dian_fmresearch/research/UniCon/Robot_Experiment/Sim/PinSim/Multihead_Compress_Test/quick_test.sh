#!/usr/bin/env bash
set -e

# ============================================================
# 快速测试 - 10个样本验证管道
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./quick_test_results}

echo "=========================================="
echo "Quick Test (10 samples)"
echo "=========================================="
echo "Data: $DATA"
echo "Output: $OUT"
echo ""

# 单个配置测试
python3 compress_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --max-samples 10 \
  --Kq 128 \
  --Kdq 64 \
  --Ktau 64 \
  --rq 100 \
  --rdq 25 \
  --rtau 25 \
  --q-filter none \
  --dq-filter none \
  --tau-filter none \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --save-mapping

echo ""
echo "=========================================="
echo "Quick Test Complete!"
echo "Check: $OUT/"
echo "=========================================="


