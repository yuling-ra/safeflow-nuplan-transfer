#!/usr/bin/env bash
set -e

# ============================================================
# 资源监控测试 - 使用少量配置测试资源管理功能
# ============================================================

DATA=${1:-../Compress_Test/test_complete.npz}
OUT=${2:-./test_resource_monitor_results}

echo "=========================================="
echo "Resource Monitor Test"
echo "=========================================="
echo "Testing resource monitoring with 10 configs"
echo "Data: $DATA"
echo "Output: $OUT"
echo ""

python3 sweep_multigroup_pca.py \
  --data "$DATA" \
  --out "$OUT" \
  --Kq 128 256 \
  --Kdq 64 128 \
  --Ktau 64 128 \
  --latent-triplets \
    "100,25,25" \
    "120,30,30" \
  --filters none \
  --wq 1.0 --wdq 0.25 --wtau 0.1 \
  --num-workers 8 \
  --cpu-threshold 75.0 \
  --memory-threshold 75.0 \
  --batch-size 5 \
  --max-samples 100

echo ""
echo "=========================================="
echo "Resource Monitor Test Complete!"
echo "Results: $OUT/results_summary_multigroup.json"
echo "=========================================="

