#!/usr/bin/env bash
# B-Spline 压缩网格扫描启动脚本
# 支持多种预设模式，自动资源管理

set -e

# 默认参数
DATA="../Compress_Test/test_complete.npz"
OUT="./results_sweep_$(date +%Y%m%d_%H%M%S)"
MODE="medium"
WORKERS=4

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -d, --data PATH       数据文件路径 (默认: ../Compress_Test/test_complete.npz)
    -o, --out DIR         输出目录 (默认: ./results_sweep_TIMESTAMP)
    -m, --mode MODE       预设模式: quick|small|medium|large|full (默认: medium)
    -w, --workers N       并行 worker 数量 (默认: 4)
    -h, --help            显示此帮助信息

预设模式说明:
    quick   - 快速测试 (10样本, 6配置, ~1分钟)
    small   - 小规模扫描 (~32配置, ~5分钟)
    medium  - 中等规模扫描 (~150配置, ~20分钟)
    large   - 大规模扫描 (~288配置, ~45分钟)
    full    - 完整扫描 (~1050配置, ~2小时)

示例:
    # 快速测试
    $0 --mode quick

    # 中等规模扫描，使用8个workers
    $0 --mode medium --workers 8

    # 自定义数据文件和输出目录
    $0 --data /path/to/data.npz --out ./my_results --mode large

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA="$2"
            shift 2
            ;;
        -o|--out)
            OUT="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
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

# 打印配置
echo "========================================================================"
echo "B-Spline 压缩网格扫描"
echo "========================================================================"
echo "数据文件: $DATA"
echo "输出目录: $OUT"
echo "模式: $MODE"
echo "Workers: $WORKERS"
echo "========================================================================"
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "[检查] 验证 Python 依赖..."
python3 -c "import numpy, scipy, sklearn, matplotlib, tqdm, psutil, orjson" 2>/dev/null || {
    echo "警告: 部分依赖缺失，尝试安装..."
    pip3 install -r requirements.txt
}

# 运行扫描
echo ""
echo "[启动] 开始网格扫描..."
echo ""

python3 sweep_spline.py \
    --data "$DATA" \
    --out "$OUT" \
    --mode "$MODE" \
    --workers "$WORKERS"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "扫描完成！"
    echo "========================================================================"
    echo "结果目录: $OUT"
    echo ""
    echo "生成的文件:"
    echo "  - results_summary.json       # 所有配置的结果汇总"
    echo "  - sweep_meta.json            # 扫描元数据"
    echo "  - sweep_stats.json           # 统计信息"
    echo "  - result_Spline_*.json       # 单个配置的详细结果"
    echo ""
    echo "下一步:"
    echo "  1. 查看结果: python3 visualize_results.py --input $OUT"
    echo "  2. 分析最佳配置: cat $OUT/sweep_stats.json"
    echo "========================================================================"
else
    echo ""
    echo "错误: 扫描失败"
    exit 1
fi
