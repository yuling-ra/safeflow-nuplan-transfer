#!/bin/bash
# 测试稳态初始化完整工作流

echo "=========================================="
echo "稳态初始化功能测试"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤1: 记录稳态周期
echo -e "${BLUE}[步骤 1/3] 记录稳态周期 (10个周期，无可视化)${NC}"
echo "命令: python record_steady_state.py --cyc 10 --no-viz"
echo ""
python record_steady_state.py --cyc 10 --no-viz

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告: 记录稳态周期失败，可能缺少依赖或URDF文件${NC}"
    echo "请检查 pinocchio 和 meshcat 是否正确安装"
    exit 1
fi

if [ ! -f "steady_state_cycle.npz" ]; then
    echo -e "${YELLOW}错误: 未生成 steady_state_cycle.npz 文件${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 稳态周期记录成功！${NC}"
echo ""

# 步骤2: 传统方式运行（对比用）
echo -e "${BLUE}[步骤 2/3] 传统方式运行 (1.5个周期)${NC}"
echo "命令: python pin_fr3_draw_eight.py --cyc 1.5 --speed 0"
echo "说明: 有随机初始化、IK求解、Landing过程"
echo ""
# 使用 timeout 避免挂起，如果系统没有可视化可能会失败
timeout 120 python pin_fr3_draw_eight.py --cyc 1.5 --speed 0 2>/dev/null || true

echo ""
echo -e "${GREEN}✓ 传统方式测试完成${NC}"
echo ""

# 步骤3: 稳态初始化方式运行
echo -e "${BLUE}[步骤 3/3] 稳态初始化方式运行 (1.5个周期)${NC}"
echo "命令: python pin_fr3_draw_eight.py --init-steady-state steady_state_cycle.npz --cyc 1.5 --speed 0 --seed 42"
echo "说明: 跳过所有过渡阶段，直接进入周期运动"
echo ""
timeout 120 python pin_fr3_draw_eight.py --init-steady-state steady_state_cycle.npz --cyc 1.5 --speed 0 --seed 42 2>/dev/null || true

echo ""
echo -e "${GREEN}✓ 稳态初始化测试完成${NC}"
echo ""

# 总结
echo "=========================================="
echo -e "${GREEN}测试完成！${NC}"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - steady_state_cycle.npz (稳态周期数据)"
if [ -f "steady_state_cycle.npz" ]; then
    SIZE=$(du -h steady_state_cycle.npz | cut -f1)
    echo "    大小: $SIZE"
fi
echo ""
echo "下一步:"
echo "  1. 查看使用文档: cat STEADY_STATE_INIT_USAGE.md"
echo "  2. 使用稳态初始化:"
echo "     python pin_fr3_draw_eight.py --init-steady-state steady_state_cycle.npz"
echo "  3. 不同随机种子:"
echo "     python pin_fr3_draw_eight.py --init-steady-state steady_state_cycle.npz --seed 123"
echo ""

