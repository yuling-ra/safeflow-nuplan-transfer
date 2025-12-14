# B-Spline 压缩测试框架 - 完整总结

## 🎯 项目目标

在 **≤1000 维潜变量预算**内，将轨迹压缩的 **WRMSE 从 0.26 降至 < 0.01**。

### 当前基线（quick test, N=10）
```
WRMSE: 0.264
  - rmse_q:   0.029 (已经很好)
  - rmse_dq:  0.130 (需要优化)
  - rmse_tau: 0.808 (主要瓶颈) ⚠️
```

### 优化目标
```
WRMSE: < 0.01
  - rmse_q:   < 0.005 (毫米级)
  - rmse_dq:  < 0.02  (10x 改进)
  - rmse_tau: < 0.05  (16x 改进)
潜变量: ≤ 1000 维
```

---

## 📁 文件结构

```
Spline_Test/
├── 核心模块
│   ├── spline_utils.py              # B-Spline 基函数、编码/解码
│   ├── utils.py                     # 数据加载、标准化、评估
│   ├── resource_monitor.py          # 系统资源监控
│   └── requirements.txt             # Python 依赖
│
├── 测试脚本
│   ├── spline_benchmark.py          # 单次基准测试
│   ├── sweep_spline.py              # 标准网格扫描
│   ├── sweep_spline_advanced.py    # 高级扫描（支持多档位λ、q导数）
│   └── visualize_results.py         # 结果可视化
│
├── 启动脚本
│   ├── run_sweep.sh                 # 标准扫描（5种预设模式）
│   ├── run_sweep_optimized.sh      # 优化扫描（4种策略）⭐
│   └── test_optimized.sh            # 快速测试（验证优化）
│
├── 分析工具
│   └── compare_strategies.py        # 多策略对比分析
│
└── 文档
    ├── README.md                    # 完整文档
    ├── QUICK_START.md               # 快速开始
    ├── OPTIMIZATION_GUIDE.md        # 优化指南 ⭐
    └── SUMMARY.md                   # 本文件
```

---

## 🚀 快速开始

### 1️⃣ 快速测试（2分钟验证）

```bash
cd /home/nvidiapc/Flow_ICLR/fmtorch/research/UniCon/Robot_Experiment/Sim/PinSim/Spline_Test

# 测试优化版扫描
bash test_optimized.sh
```

### 2️⃣ 推荐配置（策略1，15分钟）

```bash
# 固定 q 高保真 + 优化 dq/tau
bash run_sweep_optimized.sh --strategy1 --workers 8

# 查看结果
LATEST=$(ls -td results_optimized_sweep_* | head -1)
cat $LATEST/sweep_stats.json | jq '.best_under_1k'
```

### 3️⃣ 完整扫描（1-2小时）

```bash
# 所有策略组合（~800配置）
bash run_sweep_optimized.sh --full --workers 12

# 可视化
LATEST=$(ls -td results_optimized_sweep_* | head -1)
python3 visualize_results.py --input $LATEST
```

---

## 🎨 优化策略详解

### 策略1: 固定 q 高保真 + 优化 dq/tau（推荐）⭐

**思路**: 
- q 已经很好（rmse=0.029），固定在高保真档
- 重点优化 dq 和 tau（当前瓶颈）

**配置**:
```bash
Mq=128 (固定，896维)
Mdq ∈ {8,16,24,32,48,64}
Mtau ∈ {8,16,24,32,48,64}
deg_q ∈ {3,5}
deg_tau ∈ {3,5}  # tau 有大瞬态，尝试5次
lambda_tau ∈ {1e-6,1e-5}  # 抑制振铃
```

**预期**:
- 配置数: ~144
- 耗时: ~15分钟
- WRMSE: < 0.01
- 最佳潜变量: 896 + 224 + 224 = 1344 维

**运行**:
```bash
bash run_sweep_optimized.sh --strategy1
```

---

### 策略2: 三种预算方案 (~1000维)

**思路**: 
- 严格控制在 1000 维预算内
- 对比不同资源分配策略

**方案**:
- **方案A**: `Mq=128(896) + Mdq=8(56) + Mtau=6(42) = 994维`
- **方案B**: `Mq=120(840) + Mdq=16(112) + Mtau=8(56) = 1008维`
- **方案C**: `Mq=110(770) + Mdq=24(168) + Mtau=10(70) = 1008维`

**预期**:
- 配置数: ~30
- 耗时: ~5分钟
- WRMSE: < 0.02

**运行**:
```bash
bash run_sweep_optimized.sh --strategy2
```

---

### 策略3: 从 q 导数评估 dq（节省维度）

**思路**: 
- dq 从 q 的解析导数计算（B'θ）
- 不占用额外维度，更抗 OOD

**配置**:
```bash
Mq ∈ {96,128,160}
Mdq=0  # 从 q 导数计算
Mtau ∈ {16,32,48,64}
eval-dq-from-q: 启用
```

**预期**:
- 配置数: ~48
- 耗时: ~8分钟
- WRMSE: < 0.015
- 最佳潜变量: 896 + 0 + 112 = 1008 维
- 更好的泛化性能

**运行**:
```bash
bash run_sweep_optimized.sh --strategy3
```

---

### 完整扫描: 所有策略组合

**思路**: 
- 探索全部可能的配置空间
- 找到全局最优解

**配置**:
```bash
Mq ∈ {96,110,120,128,160}
Mdq ∈ {0,8,16,24,32,48,64}  # 包含从 q 导数
Mtau ∈ {6,8,10,16,24,32,48,64}
deg ∈ {3,5}  # 全组合
lambda ∈ {1e-6,1e-5,1e-4}  # 多档位
```

**预期**:
- 配置数: ~800
- 耗时: ~1-2小时
- 找到全局最优

**运行**:
```bash
bash run_sweep_optimized.sh --full --workers 12
```

---

## 📊 结果分析

### 查看统计信息
```bash
cat results_optimized_sweep_*/sweep_stats.json
```

### 提取最佳配置（≤1000维）
```bash
python3 -c "
import json
with open('results_optimized_sweep_*/sweep_stats.json') as f:
    stats = json.load(f)
    best = stats['best_under_1k']
    print(f'最佳配置 (≤1000维):')
    print(f'  M: q={best[\"M\"][\"q\"]}, dq={best[\"M\"][\"dq\"]}, tau={best[\"M\"][\"tau\"]}')
    print(f'  degree: q={best[\"degree\"][\"q\"]}, dq={best[\"degree\"][\"dq\"]}, tau={best[\"degree\"][\"tau\"]}')
    print(f'  潜变量: {best[\"latent_total\"]}')
    print(f'  WRMSE: {best[\"wrmse\"]:.6f}')
    print(f'  RMSE: q={best[\"rmse_q\"]:.6f}, dq={best[\"rmse_dq\"]:.6f}, tau={best[\"rmse_tau\"]:.6f}')
"
```

### 可视化对比
```bash
python3 visualize_results.py --input results_optimized_sweep_*/
```

生成的图表：
- `curve_latent_vs_wrmse.png` - 参数量 vs WRMSE
- `curve_latent_vs_rmse_groups.png` - q/dq/tau 分别的误差
- `pareto_frontier.png` - Pareto 前沿
- `rmse_breakdown_comparison.png` - 分组误差对比

### 多策略对比
```bash
python3 compare_strategies.py \
  --dirs results_optimized_sweep_*/ \
  --names "策略1" "策略2" "策略3" \
  --output ./comparison_results
```

---

## 🔑 关键参数说明

### 控制点数 M（决定样条灵活性）

| 组 | 推荐范围 | 说明 |
|---|---------|------|
| **q** | 96-160 | 位置轨迹，需要高保真（毫米级） |
| **dq** | 16-48 | 速度，相对平滑 |
| **tau** | 16-64 | 力矩，曲线复杂，大瞬态 |

### 样条阶数 degree（决定平滑度）

| 阶数 | 连续性 | 适用场景 |
|-----|--------|---------|
| **3** | C² | 标准选择，适合平滑轨迹 |
| **5** | C⁴ | 更平滑，适合 tau 的大瞬态 |

### 正则化 lambda（决定稳定性）

| 值 | 效果 | 适用场景 |
|----|------|---------|
| **1e-6** | 几乎不正则化 | 标准值，追求最高精度 |
| **1e-5** | 轻度正则化 | 抑制端点振铃 |
| **1e-4** | 中度正则化 | 牺牲少许精度换稳定性 |

---

## 💡 推荐配置（基于分析）

### 🥇 配置A: 平衡型（推荐）
```yaml
Mq: 128
Mdq: 32
Mtau: 32
degree:
  q: 3
  dq: 3
  tau: 5
lambda:
  q: 1e-6
  dq: 1e-6
  tau: 1e-5
潜变量: 896 + 224 + 224 = 1344 维
预期 WRMSE: < 0.008
```

### 🥈 配置B: 低维型（1k预算）
```yaml
Mq: 120
Mdq: 16
Mtau: 8
degree:
  q: 3
  dq: 3
  tau: 5
lambda:
  q: 1e-6
  dq: 1e-6
  tau: 1e-5
潜变量: 840 + 112 + 56 = 1008 维
预期 WRMSE: < 0.015
```

### 🥉 配置C: 鲁棒型（OOD友好）
```yaml
Mq: 128
Mdq: 0  # 从 q 导数
Mtau: 16
degree:
  q: 3
  dq: 3
  tau: 5
lambda:
  q: 1e-6
  dq: 1e-6
  tau: 1e-5
潜变量: 896 + 0 + 112 = 1008 维
预期 WRMSE: < 0.012
更好的泛化性能
```

---

## 🛠️ 故障排查

### 问题: 内存不足
**解决方案**:
```bash
# 减少样本数
--max-samples 100

# 减少 workers
--workers 2

# 减小批处理
--batch-size 5
```

### 问题: CPU 占用过高
**解决方案**:
```bash
# 减少并行度
--workers 1

# 使用串行模式
# 资源监控会自动调整
```

### 问题: tau 误差仍然很大
**解决方案**:
```bash
# 增加控制点
--Mtau 64 96 128

# 使用五次样条
--deg-tau 5

# 增加正则化
--lam-tau 1e-5 1e-4
```

---

## 📈 预期改进路径

### 阶段1: 基线（已完成）
```
WRMSE: 0.264
Mq=16, Mdq=16, Mtau=16
潜变量: 336 维
```

### 阶段2: 策略1（推荐首先运行）
```
WRMSE: < 0.01 (目标)
Mq=128, Mdq=32, Mtau=32
潜变量: 1344 维
预计改进: 26x
```

### 阶段3: 策略2（1k预算）
```
WRMSE: < 0.02
Mq=120, Mdq=16, Mtau=8
潜变量: 1008 维
预计改进: 13x
```

### 阶段4: 策略3（鲁棒性）
```
WRMSE: < 0.015
Mq=128, Mdq=0, Mtau=16
潜变量: 1008 维
更好的泛化
```

---

## 🎯 下一步行动

### 1. 立即执行（推荐）
```bash
# 运行策略1
bash run_sweep_optimized.sh --strategy1 --workers 8

# 等待完成（~15分钟）
# 查看结果
LATEST=$(ls -td results_optimized_sweep_* | head -1)
cat $LATEST/sweep_stats.json
```

### 2. 对比分析
```bash
# 运行所有策略
bash run_sweep_optimized.sh --strategy1
bash run_sweep_optimized.sh --strategy2
bash run_sweep_optimized.sh --strategy3

# 对比
python3 compare_strategies.py \
  --dirs results_optimized_sweep_*/ \
  --output ./comparison_results
```

### 3. 完整扫描（计算预算充裕时）
```bash
# 运行完整扫描
bash run_sweep_optimized.sh --full --workers 12

# 可视化
python3 visualize_results.py --input results_optimized_sweep_*/
```

---

## 📚 相关文档

- **README.md** - 完整功能说明
- **QUICK_START.md** - 快速开始指南
- **OPTIMIZATION_GUIDE.md** - 详细优化策略
- **SUMMARY.md** - 本文件（总结）

---

## ✅ 检查清单

- [x] 框架部署完成
- [x] 资源监控集成
- [x] 优化策略设计
- [x] 测试脚本准备
- [ ] 运行策略1测试
- [ ] 分析结果
- [ ] 选择最佳配置
- [ ] 完整扫描（可选）

---

**准备就绪！开始优化吧！** 🚀
