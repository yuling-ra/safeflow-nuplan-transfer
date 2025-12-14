#!/usr/bin/env bash
# 重新生成所有图表（英文标注）

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <results_directory>"
    echo ""
    echo "Example:"
    echo "  $0 results_full_optimization_20251005_193134"
    exit 1
fi

RESULTS_DIR=$1

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory not found: $RESULTS_DIR"
    exit 1
fi

echo "========================================================================"
echo "Regenerating All Plots with English Labels"
echo "========================================================================"
echo "Results directory: $RESULTS_DIR"
echo ""

# Regenerate individual strategy plots
for strategy in strategy1 strategy2 strategy3; do
    if [ -d "$RESULTS_DIR/$strategy" ]; then
        echo "[Processing] $strategy..."
        python3 visualize_results.py --input "$RESULTS_DIR/$strategy" --output "$RESULTS_DIR/$strategy"
        echo ""
    else
        echo "[Skip] $strategy directory not found"
    fi
done

# Regenerate comparison plots
if [ -d "$RESULTS_DIR/strategy1" ] && [ -d "$RESULTS_DIR/strategy2" ] && [ -d "$RESULTS_DIR/strategy3" ]; then
    echo "[Processing] Comparison plots..."
    python3 compare_strategies.py \
        --dirs "$RESULTS_DIR/strategy1" "$RESULTS_DIR/strategy2" "$RESULTS_DIR/strategy3" \
        --names "Strategy1: Fixed-q-High-Fidelity" "Strategy2: 1k-Budget" "Strategy3: q-Derivative-dq" \
        --output "$RESULTS_DIR/comparison"
    echo ""
fi

echo "========================================================================"
echo "All Plots Regenerated!"
echo "========================================================================"
echo ""
echo "Generated plots:"
echo "  - $RESULTS_DIR/strategy1/curve_*.png"
echo "  - $RESULTS_DIR/strategy2/curve_*.png"
echo "  - $RESULTS_DIR/strategy3/curve_*.png"
echo "  - $RESULTS_DIR/comparison/pareto_frontier.png"
echo "  - $RESULTS_DIR/comparison/rmse_breakdown_comparison.png"
echo ""
echo "Done!"
