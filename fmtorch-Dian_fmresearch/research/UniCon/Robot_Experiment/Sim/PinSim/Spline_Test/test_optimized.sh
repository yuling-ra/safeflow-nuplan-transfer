#!/usr/bin/env bash
# 快速测试优化版扫描（使用少量样本验证）

set -e

DATA="../Compress_Test/test_complete.npz"
OUT="./test_optimized_results"

echo "========================================================================"
echo "优化版扫描 - 快速测试"
echo "========================================================================"
echo "使用 10 个样本快速验证优化策略"
echo ""

# 清理旧结果
rm -rf "$OUT"

# 测试策略1：固定 q 高保真
echo "[测试] 策略1: 固定 q 高保真 + 优化 dq/tau"
python3 sweep_spline_advanced.py \
    --data "$DATA" \
    --out "$OUT/strategy1" \
    --max-samples 10 \
    --Mq 128 \
    --Mdq 16 32 \
    --Mtau 16 32 \
    --deg-q 3 \
    --deg-dq 3 \
    --deg-tau 3 5 \
    --lam-q 1e-6 \
    --lam-dq 1e-6 \
    --lam-tau 1e-6 1e-5 \
    --workers 4 \
    --batch-size 4

echo ""
echo "[结果] 策略1:"
cat "$OUT/strategy1/sweep_stats.json" | grep -A 5 "best_under_1k" || echo "  查看完整结果: $OUT/strategy1/sweep_stats.json"

# 测试策略3：从 q 导数评估 dq
echo ""
echo "[测试] 策略3: 从 q 导数评估 dq"
python3 sweep_spline_advanced.py \
    --data "$DATA" \
    --out "$OUT/strategy3" \
    --max-samples 10 \
    --Mq 128 \
    --Mdq 0 \
    --Mtau 16 32 \
    --deg-q 3 \
    --deg-dq 3 \
    --deg-tau 3 5 \
    --lam-q 1e-6 \
    --lam-dq 1e-6 \
    --lam-tau 1e-6 1e-5 \
    --eval-dq-from-q \
    --workers 4 \
    --batch-size 4

echo ""
echo "[结果] 策略3:"
cat "$OUT/strategy3/sweep_stats.json" | grep -A 5 "best_under_1k" || echo "  查看完整结果: $OUT/strategy3/sweep_stats.json"

# 对比结果
echo ""
echo "========================================================================"
echo "对比分析"
echo "========================================================================"
python3 compare_strategies.py \
    --dirs "$OUT/strategy1" "$OUT/strategy3" \
    --names "策略1:固定q" "策略3:q导数" \
    --output "$OUT/comparison"

echo ""
echo "========================================================================"
echo "测试完成！"
echo "========================================================================"
echo "结果目录: $OUT"
echo ""
echo "下一步:"
echo "  1. 查看对比结果: cat $OUT/comparison/comparison_summary.json"
echo "  2. 运行完整扫描: bash run_sweep_optimized.sh --strategy1"
echo "========================================================================"
