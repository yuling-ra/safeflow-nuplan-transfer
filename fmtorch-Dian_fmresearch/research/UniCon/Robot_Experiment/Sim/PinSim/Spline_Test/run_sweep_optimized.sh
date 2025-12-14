#!/usr/bin/env bash
# 优化版 B-Spline 压缩扫描
# 策略：固定 q 高保真 + 重点优化 dq/tau + 探索多种配置

set -e

# 默认参数
DATA="../Compress_Test/test_complete.npz"
OUT="./results_optimized_sweep_$(date +%Y%m%d_%H%M%S)"
WORKERS=8
BATCH_SIZE=20

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

优化策略扫描模式：
    --strategy1   策略1: 固定 q 高保真(Mq=128) + 优化 dq/tau (推荐)
    --strategy2   策略2: 三种预算方案 (~1000维)
    --strategy3   策略3: 从 q 导数评估 dq (节省维度)
    --full        完整扫描: 所有策略组合 (~500配置, 1-2小时)

选项:
    -d, --data PATH       数据文件路径
    -o, --out DIR         输出目录
    -w, --workers N       并行 worker 数量 (默认: 8)
    -b, --batch-size N    批处理大小 (默认: 20)
    -h, --help            显示此帮助信息

示例:
    # 策略1: 固定 q 高保真，优化 dq/tau (~100配置, 15分钟)
    $0 --strategy1

    # 完整扫描: 所有策略 (~500配置, 1-2小时)
    $0 --full --workers 12

    # 自定义数据和输出
    $0 --strategy1 --data /path/to/data.npz --out ./my_results

EOF
}

# 默认策略
STRATEGY="strategy1"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy1)
            STRATEGY="strategy1"
            shift
            ;;
        --strategy2)
            STRATEGY="strategy2"
            shift
            ;;
        --strategy3)
            STRATEGY="strategy3"
            shift
            ;;
        --full)
            STRATEGY="full"
            shift
            ;;
        -d|--data)
            DATA="$2"
            shift 2
            ;;
        -o|--out)
            OUT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查数据文件
if [ ! -f "$DATA" ]; then
    echo "错误: 数据文件不存在: $DATA"
    exit 1
fi

echo "========================================================================"
echo "优化版 B-Spline 压缩扫描"
echo "========================================================================"
echo "策略: $STRATEGY"
echo "数据文件: $DATA"
echo "输出目录: $OUT"
echo "Workers: $WORKERS"
echo "批处理大小: $BATCH_SIZE"
echo "========================================================================"
echo ""

# 根据策略设置参数
case $STRATEGY in
    strategy1)
        echo "[策略1] 固定 q 高保真(Mq=128) + 优化 dq/tau"
        echo "  - Mq: 128 (固定，896维)"
        echo "  - Mdq: 8,16,24,32,48,64 (探索最佳)"
        echo "  - Mtau: 8,16,24,32,48,64 (探索最佳)"
        echo "  - deg_q: 3,5 (对比)"
        echo "  - deg_dq: 3 (标准)"
        echo "  - deg_tau: 3,5 (tau有大瞬态，尝试5次)"
        echo "  - lambda_tau: 1e-6,1e-5 (抑制振铃)"
        echo "  预计配置数: ~144"
        echo ""
        
        python3 sweep_spline.py \
            --data "$DATA" \
            --out "$OUT" \
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
            --batch-size "$BATCH_SIZE"
        ;;
    
    strategy2)
        echo "[策略2] 三种预算方案 (~1000维)"
        echo "  方案A: Mq=128(896) + Mdq=8(56) + Mtau=6(42) = 994维"
        echo "  方案B: Mq=120(840) + Mdq=16(112) + Mtau=8(56) = 1008维"
        echo "  方案C: Mq=110(770) + Mdq=24(168) + Mtau=10(70) = 1008维"
        echo "  预计配置数: ~30"
        echo ""
        
        python3 sweep_spline.py \
            --data "$DATA" \
            --out "$OUT" \
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
            --batch-size "$BATCH_SIZE"
        ;;
    
    strategy3)
        echo "[策略3] 从 q 导数评估 dq (节省维度)"
        echo "  - Mq: 96,128,160 (高保真)"
        echo "  - Mdq: 从 q 导数计算 (0维)"
        echo "  - Mtau: 16,32,48,64 (优化)"
        echo "  - eval-dq-from-q: 启用"
        echo "  预计配置数: ~48"
        echo ""
        
        python3 sweep_spline.py \
            --data "$DATA" \
            --out "$OUT" \
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
            --batch-size "$BATCH_SIZE"
        ;;
    
    full)
        echo "[完整扫描] 所有策略组合"
        echo "  - q: 高保真档 (96,110,120,128,160)"
        echo "  - dq: 全范围探索 (0,8,16,24,32,48,64)"
        echo "  - tau: 全范围探索 (6,8,10,16,24,32,48,64)"
        echo "  - 阶数: 3,5 (全组合)"
        echo "  - lambda: 多档位 (1e-6,1e-5,1e-4)"
        echo "  预计配置数: ~800"
        echo ""
        
        python3 sweep_spline.py \
            --data "$DATA" \
            --out "$OUT" \
            --Mq 96 110 120 128 160 \
            --Mdq 0 8 16 24 32 48 64 \
            --Mtau 6 8 10 16 24 32 48 64 \
            --deg-q 3 5 \
            --deg-dq 3 5 \
            --deg-tau 3 5 \
            --lam-q 1e-6 1e-5 \
            --lam-dq 1e-6 1e-5 \
            --lam-tau 1e-6 1e-5 1e-4 \
            --workers "$WORKERS" \
            --batch-size "$BATCH_SIZE"
        ;;
esac

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "扫描完成！"
    echo "========================================================================"
    echo "结果目录: $OUT"
    echo ""
    echo "下一步:"
    echo "  1. 查看统计: cat $OUT/sweep_stats.json"
    echo "  2. 可视化: python3 visualize_results.py --input $OUT"
    echo "  3. 对比分析: python3 compare_strategies.py --dirs $OUT"
    echo "========================================================================"
else
    echo ""
    echo "错误: 扫描失败"
    exit 1
fi
