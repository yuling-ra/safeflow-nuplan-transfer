#!/usr/bin/env bash
# 一键运行所有优化策略并生成对比报告

set -e

echo "========================================================================"
echo "B-Spline 压缩优化 - 完整流程"
echo "========================================================================"
echo "本脚本将依次运行所有优化策略，并生成对比分析报告"
echo ""
echo "预计总耗时: ~30-40分钟"
echo "  - 策略1: ~15分钟 (144配置)"
echo "  - 策略2: ~5分钟 (30配置)"
echo "  - 策略3: ~8分钟 (48配置)"
echo "  - 对比分析: ~2分钟"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."
echo ""

# 配置
DATA="../Compress_Test/test_complete.npz"
WORKERS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="./results_full_optimization_${TIMESTAMP}"

# 检查数据文件
if [ ! -f "$DATA" ]; then
    echo "错误: 数据文件不存在: $DATA"
    exit 1
fi

# 创建输出目录
mkdir -p "$BASE_OUT"

# 保存运行日志
LOG_FILE="$BASE_OUT/run_log.txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================================================"
echo "开始时间: $(date)"
echo "数据文件: $DATA"
echo "输出目录: $BASE_OUT"
echo "Workers: $WORKERS"
echo "========================================================================"
echo ""

# ============================================================================
# 策略1: 固定 q 高保真 + 优化 dq/tau
# ============================================================================
echo ""
echo "========================================================================"
echo "[1/3] 策略1: 固定 q 高保真 + 优化 dq/tau"
echo "========================================================================"
echo "配置: Mq=128(固定), Mdq∈{8,16,24,32,48,64}, Mtau∈{8,16,24,32,48,64}"
echo "预计: ~144配置, ~15分钟"
echo ""

START_TIME=$(date +%s)

python3 sweep_spline_advanced.py \
    --data "$DATA" \
    --out "$BASE_OUT/strategy1" \
    --Mq 128 \
    --Mdq 8 16 24 32 48 64 \
    --Mtau 8 16 24 32 48 64 \
    --deg-q 3 5 \
    --deg-dq 3 \
    --deg-tau 3 5 \
    --lam-q 1e-6 \
    --lam-dq 1e-6 \
    --lam-tau 1e-6 1e-5 \
    --workers "$WORKERS" \
    --batch-size 20

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "[策略1] 完成! 耗时: $((ELAPSED/60))分$((ELAPSED%60))秒"

# ============================================================================
# 策略2: 三种预算方案 (~1000维)
# ============================================================================
echo ""
echo "========================================================================"
echo "[2/3] 策略2: 三种预算方案 (~1000维)"
echo "========================================================================"
echo "方案A: Mq=128(896) + Mdq=8(56) + Mtau=6(42) = 994维"
echo "方案B: Mq=120(840) + Mdq=16(112) + Mtau=8(56) = 1008维"
echo "方案C: Mq=110(770) + Mdq=24(168) + Mtau=10(70) = 1008维"
echo "预计: ~30配置, ~5分钟"
echo ""

START_TIME=$(date +%s)

python3 sweep_spline_advanced.py \
    --data "$DATA" \
    --out "$BASE_OUT/strategy2" \
    --Mq 110 120 128 \
    --Mdq 6 8 16 24 \
    --Mtau 6 8 10 16 \
    --deg-q 3 5 \
    --deg-dq 3 \
    --deg-tau 3 5 \
    --lam-q 1e-6 \
    --lam-dq 1e-6 \
    --lam-tau 1e-6 1e-5 \
    --workers "$WORKERS" \
    --batch-size 15

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "[策略2] 完成! 耗时: $((ELAPSED/60))分$((ELAPSED%60))秒"

# ============================================================================
# 策略3: 从 q 导数评估 dq (节省维度)
# ============================================================================
echo ""
echo "========================================================================"
echo "[3/3] 策略3: 从 q 导数评估 dq (节省维度)"
echo "========================================================================"
echo "配置: Mq∈{96,128,160}, Mdq=0(从q导数), Mtau∈{16,32,48,64}"
echo "预计: ~48配置, ~8分钟"
echo ""

START_TIME=$(date +%s)

python3 sweep_spline_advanced.py \
    --data "$DATA" \
    --out "$BASE_OUT/strategy3" \
    --Mq 96 128 160 \
    --Mdq 0 \
    --Mtau 16 32 48 64 \
    --deg-q 3 5 \
    --deg-dq 3 \
    --deg-tau 3 5 \
    --lam-q 1e-6 \
    --lam-dq 1e-6 \
    --lam-tau 1e-6 1e-5 \
    --eval-dq-from-q \
    --workers "$WORKERS" \
    --batch-size 15

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "[策略3] 完成! 耗时: $((ELAPSED/60))分$((ELAPSED%60))秒"

# ============================================================================
# 对比分析
# ============================================================================
echo ""
echo "========================================================================"
echo "对比分析"
echo "========================================================================"

python3 compare_strategies.py \
    --dirs "$BASE_OUT/strategy1" "$BASE_OUT/strategy2" "$BASE_OUT/strategy3" \
    --names "策略1:固定q高保真" "策略2:1k预算方案" "策略3:q导数dq" \
    --output "$BASE_OUT/comparison"

# ============================================================================
# 生成可视化
# ============================================================================
echo ""
echo "========================================================================"
echo "生成可视化图表"
echo "========================================================================"

for strategy in strategy1 strategy2 strategy3; do
    echo "[可视化] $strategy..."
    python3 visualize_results.py --input "$BASE_OUT/$strategy" --output "$BASE_OUT/$strategy" 2>/dev/null || true
done

# ============================================================================
# 生成总结报告
# ============================================================================
echo ""
echo "========================================================================"
echo "生成总结报告"
echo "========================================================================"

REPORT_FILE="$BASE_OUT/FINAL_REPORT.md"

cat > "$REPORT_FILE" << 'EOF'
# B-Spline 压缩优化 - 最终报告

## 📊 测试概况

**运行时间**: 
EOF

echo "- 开始: $(head -1 "$LOG_FILE" | grep "开始时间" | cut -d: -f2-)" >> "$REPORT_FILE"
echo "- 结束: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
**测试配置**:
- 数据集: test_complete.npz
- 总配置数: ~222 (144 + 30 + 48)
- 并行 workers: 8

---

## 🏆 最佳配置汇总

### 策略1: 固定 q 高保真 + 优化 dq/tau

EOF

echo '```json' >> "$REPORT_FILE"
cat "$BASE_OUT/strategy1/sweep_stats.json" | jq '.best_under_1k' >> "$REPORT_FILE" 2>/dev/null || echo "查看: strategy1/sweep_stats.json" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
### 策略2: 三种预算方案 (~1000维)

EOF

echo '```json' >> "$REPORT_FILE"
cat "$BASE_OUT/strategy2/sweep_stats.json" | jq '.best_under_1k' >> "$REPORT_FILE" 2>/dev/null || echo "查看: strategy2/sweep_stats.json" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
### 策略3: 从 q 导数评估 dq

EOF

echo '```json' >> "$REPORT_FILE"
cat "$BASE_OUT/strategy3/sweep_stats.json" | jq '.best_under_1k' >> "$REPORT_FILE" 2>/dev/null || echo "查看: strategy3/sweep_stats.json" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
---

## 📈 对比分析

详细对比见: `comparison/comparison_summary.json`

### 可视化图表

- **Pareto 前沿**: `comparison/pareto_frontier.png`
- **RMSE 分解**: `comparison/rmse_breakdown_comparison.png`

各策略详细曲线:
- 策略1: `strategy1/curve_latent_vs_wrmse.png`
- 策略2: `strategy2/curve_latent_vs_wrmse.png`
- 策略3: `strategy3/curve_latent_vs_wrmse.png`

---

## 💡 推荐配置

基于测试结果，推荐使用：

EOF

# 找到全局最佳配置
python3 << 'PYTHON_SCRIPT' >> "$REPORT_FILE"
import json
import glob

best_overall = None
best_wrmse = float('inf')
best_strategy = ""

for strategy in ['strategy1', 'strategy2', 'strategy3']:
    try:
        with open(f'$BASE_OUT/{strategy}/sweep_stats.json') as f:
            stats = json.load(f)
            if 'best_under_1k' in stats and stats['best_under_1k']:
                wrmse = stats['best_under_1k']['wrmse']
                if wrmse < best_wrmse:
                    best_wrmse = wrmse
                    best_overall = stats['best_under_1k']
                    best_strategy = strategy
    except:
        pass

if best_overall:
    print(f"**最佳策略**: {best_strategy}")
    print(f"")
    print(f"**配置**:")
    print(f"- M: q={best_overall['M']['q']}, dq={best_overall['M']['dq']}, tau={best_overall['M']['tau']}")
    print(f"- degree: q={best_overall['degree']['q']}, dq={best_overall['degree']['dq']}, tau={best_overall['degree']['tau']}")
    print(f"- lambda: q={best_overall['lambda']['q']}, dq={best_overall['lambda']['dq']}, tau={best_overall['lambda']['tau']}")
    print(f"")
    print(f"**性能**:")
    print(f"- 潜变量: {best_overall['latent_total']} 维")
    print(f"- WRMSE: {best_overall['wrmse']:.6f}")
    print(f"- RMSE: q={best_overall['rmse_q']:.6f}, dq={best_overall['rmse_dq']:.6f}, tau={best_overall['rmse_tau']:.6f}")
PYTHON_SCRIPT

cat >> "$REPORT_FILE" << 'EOF'

---

## 📁 文件结构

```
results_full_optimization_TIMESTAMP/
├── strategy1/                    # 策略1结果
│   ├── results_summary.json
│   ├── sweep_stats.json
│   └── curve_*.png
├── strategy2/                    # 策略2结果
│   ├── results_summary.json
│   ├── sweep_stats.json
│   └── curve_*.png
├── strategy3/                    # 策略3结果
│   ├── results_summary.json
│   ├── sweep_stats.json
│   └── curve_*.png
├── comparison/                   # 对比分析
│   ├── comparison_summary.json
│   ├── pareto_frontier.png
│   └── rmse_breakdown_comparison.png
├── run_log.txt                   # 运行日志
└── FINAL_REPORT.md               # 本报告
```

---

## 🎯 下一步

1. **查看详细结果**: 
   ```bash
   cat comparison/comparison_summary.json
   ```

2. **查看可视化**: 
   打开各策略目录下的 PNG 图表

3. **选择最佳配置**: 
   根据 WRMSE、潜变量维度、泛化性能综合考虑

4. **应用到 Flow Matching**: 
   使用最佳配置训练 FM 模型

---

**报告生成时间**: $(date)
EOF

echo ""
echo "报告已生成: $REPORT_FILE"

# ============================================================================
# 最终总结
# ============================================================================
echo ""
echo "========================================================================"
echo "全部完成！"
echo "========================================================================"
echo "结束时间: $(date)"
echo ""
echo "结果目录: $BASE_OUT"
echo ""
echo "生成的文件:"
echo "  - 策略1结果: $BASE_OUT/strategy1/"
echo "  - 策略2结果: $BASE_OUT/strategy2/"
echo "  - 策略3结果: $BASE_OUT/strategy3/"
echo "  - 对比分析: $BASE_OUT/comparison/"
echo "  - 运行日志: $BASE_OUT/run_log.txt"
echo "  - 最终报告: $BASE_OUT/FINAL_REPORT.md"
echo ""
echo "下一步:"
echo "  1. 查看最终报告: cat $BASE_OUT/FINAL_REPORT.md"
echo "  2. 查看对比分析: cat $BASE_OUT/comparison/comparison_summary.json"
echo "  3. 查看可视化图表: 打开 $BASE_OUT/comparison/*.png"
echo ""
echo "========================================================================"
