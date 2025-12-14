# B-Spline 压缩优化指南

## 🎯 优化目标

根据初步测试结果分析：
- **当前瓶颈**: `rmse_tau=0.808` (力矩) 和 `rmse_dq=0.130` (速度)
- **优化目标**: 在 ≤1000 维预算内，将 WRMSE 从 0.26 降至 < 0.01
- **策略**: 固定 q 高保真 + 重点优化 dq/tau

## 📊 优化策略

### 策略1: 固定 q 高保真 + 优化 dq/tau（推荐）

**配置**:
- `Mq=128` (固定，896维) - 锁定 q 在毫米级精度
- `Mdq ∈ {8,16,24,32,48,64}` - 探索 dq 最佳配置
- `Mtau ∈ {8,16,24,32,48,64}` - 探索 tau 最佳配置
- `deg_q ∈ {3,5}` - 对比三次/五次样条
- `deg_tau ∈ {3,5}` - tau 有大瞬态，尝试五次
- `lambda_tau ∈ {1e-6,1e-5}` - 抑制端点振铃

**运行**:
```bash
bash run_sweep_optimized.sh --strategy1 --workers 8
```

**预计**:
- 配置数: ~144
- 耗时: ~15分钟
- 最佳潜变量: 896 + 56 + 56 = 1008 维

---

### 策略2: 三种预算方案 (~1000维)

**方案A**: `Mq=128(896) + Mdq=8(56) + Mtau=6(42) = 994维`
**方案B**: `Mq=120(840) + Mdq=16(112) + Mtau=8(56) = 1008维`
**方案C**: `Mq=110(770) + Mdq=24(168) + Mtau=10(70) = 1008维`

**运行**:
```bash
bash run_sweep_optimized.sh --strategy2 --workers 8
```

**预计**:
- 配置数: ~30
- 耗时: ~5分钟
- 对比不同资源分配策略

---

### 策略3: 从 q 导数评估 dq（节省维度）

**配置**:
- `Mq ∈ {96,128,160}` - 高保真 q
- `Mdq=0` - 从 q 解析导数计算 dq（不占额外维度）
- `Mtau ∈ {16,32,48,64}` - 优化 tau
- 更抗 OOD，适合推广到新轨迹

**运行**:
```bash
bash run_sweep_optimized.sh --strategy3 --workers 8
```

**预计**:
- 配置数: ~48
- 耗时: ~8分钟
- 最佳潜变量: 896 + 0 + 112 = 1008 维

---

### 完整扫描: 所有策略组合

**配置**:
- `Mq ∈ {96,110,120,128,160}` - 全范围 q
- `Mdq ∈ {0,8,16,24,32,48,64}` - 包含从 q 导数
- `Mtau ∈ {6,8,10,16,24,32,48,64}` - 全范围 tau
- `deg ∈ {3,5}` - 全组合
- `lambda ∈ {1e-6,1e-5,1e-4}` - 多档位正则化

**运行**:
```bash
bash run_sweep_optimized.sh --full --workers 12
```

**预计**:
- 配置数: ~800
- 耗时: ~1-2小时
- 找到全局最优配置

## 🚀 快速开始

### 1. 推荐流程（策略1）

```bash
cd /home/nvidiapc/Flow_ICLR/fmtorch/research/UniCon/Robot_Experiment/Sim/PinSim/Spline_Test

# 运行策略1
bash run_sweep_optimized.sh --strategy1 --workers 8

# 等待完成后，可视化结果
LATEST=$(ls -td results_optimized_sweep_* | head -1)
python3 visualize_results.py --input $LATEST

# 查看最佳配置
cat $LATEST/sweep_stats.json | grep -A 20 "best_under_1k"
```

### 2. 对比多个策略

```bash
# 运行所有策略
bash run_sweep_optimized.sh --strategy1 --workers 8
bash run_sweep_optimized.sh --strategy2 --workers 8
bash run_sweep_optimized.sh --strategy3 --workers 8

# 对比结果
python3 compare_strategies.py \
  --dirs results_optimized_sweep_*/ \
  --names "策略1:固定q" "策略2:预算方案" "策略3:q导数" \
  --output ./comparison_results
```

### 3. 完整扫描（计算预算充裕时）

```bash
# 运行完整扫描
bash run_sweep_optimized.sh --full --workers 12

# 可视化
LATEST=$(ls -td results_optimized_sweep_* | head -1)
python3 visualize_results.py --input $LATEST
```

## 📈 预期改进

### 当前基线（quick test）
```
WRMSE: 0.264
  - rmse_q:   0.029 (已经很好)
  - rmse_dq:  0.130 (需要优化)
  - rmse_tau: 0.808 (主要瓶颈)
```

### 策略1 预期（Mq=128, Mdq=48, Mtau=48）
```
WRMSE: < 0.01 (目标)
  - rmse_q:   < 0.005 (毫米级)
  - rmse_dq:  < 0.02  (10x 改进)
  - rmse_tau: < 0.05  (16x 改进)
潜变量: 896 + 336 + 336 = 1568 维
```

### 策略2 预期（方案B: Mq=120, Mdq=16, Mtau=8）
```
WRMSE: < 0.02
潜变量: 840 + 112 + 56 = 1008 维 (刚好1k)
```

### 策略3 预期（Mq=128, Mdq=0, Mtau=16）
```
WRMSE: < 0.015
  - rmse_dq: 稍高（从导数），但更鲁棒
潜变量: 896 + 0 + 112 = 1008 维
```

## 🔍 关键参数说明

### 控制点数 M
- **q**: 128 (896维) - 锁定高保真，RMSE_q < 0.005
- **dq**: 16-48 - 平衡精度与维度
- **tau**: 16-48 - tau 曲线复杂，需要更多控制点

### 样条阶数 degree
- **3 (三次)**: 标准选择，C² 连续，适合平滑轨迹
- **5 (五次)**: C⁴ 连续，更平滑，适合 tau 的大瞬态

### 正则化 lambda
- **1e-6**: 标准值，几乎不正则化
- **1e-5**: 轻度正则化，抑制端点振铃
- **1e-4**: 中度正则化，牺牲少许精度换取稳定性

## 📊 结果分析

### 查看统计信息
```bash
cat results_optimized_sweep_*/sweep_stats.json
```

### 提取 ≤1000维 最佳配置
```bash
python3 -c "
import json
with open('results_optimized_sweep_*/sweep_stats.json') as f:
    stats = json.load(f)
    best = stats['best_under_1k']
    print(f'最佳配置 (≤1000维):')
    print(f'  M: q={best[\"M\"][\"q\"]}, dq={best[\"M\"][\"dq\"]}, tau={best[\"M\"][\"tau\"]}')
    print(f'  潜变量: {best[\"latent_total\"]}')
    print(f'  WRMSE: {best[\"wrmse\"]:.6f}')
"
```

### 可视化对比
```bash
python3 visualize_results.py --input results_optimized_sweep_*/
```

生成的图表：
- `curve_latent_vs_wrmse.png` - 参数量 vs WRMSE
- `curve_latent_vs_rmse_groups.png` - q/dq/tau 分别的误差
- `curve_compression_vs_wrmse.png` - 压缩比 vs WRMSE

## 💡 优化建议

### 1. 如果 tau 仍然是瓶颈
- 增加 `Mtau` 到 64 或更高
- 使用 `degree=5` 处理大瞬态
- 尝试 `lambda_tau=1e-5` 或 `1e-4` 抑制振铃

### 2. 如果需要更低维度
- 使用策略3（从 q 导数评估 dq）
- 减少 `Mq` 到 96 或 110
- 权衡精度与维度

### 3. 如果需要更高精度
- 增加 `Mq` 到 160
- 使用 `degree=5` 全组
- 降低 `lambda` 到 `1e-7`

## 🎯 推荐配置（基于分析）

### 配置A: 平衡型（推荐）
```
Mq=128, Mdq=32, Mtau=32
degree: q=3, dq=3, tau=5
lambda: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 896 + 224 + 224 = 1344 维
预期 WRMSE: < 0.008
```

### 配置B: 低维型（1k预算）
```
Mq=120, Mdq=16, Mtau=8
degree: q=3, dq=3, tau=5
lambda: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 840 + 112 + 56 = 1008 维
预期 WRMSE: < 0.015
```

### 配置C: 鲁棒型（OOD友好）
```
Mq=128, Mdq=0 (from q), Mtau=16
degree: q=3, dq=3, tau=5
lambda: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 896 + 0 + 112 = 1008 维
预期 WRMSE: < 0.012
更好的泛化性能
```

## 📝 使用示例

```bash
# 1. 运行推荐配置（策略1）
bash run_sweep_optimized.sh --strategy1

# 2. 查看实时进度
tail -f results_optimized_sweep_*/sweep_meta.json

# 3. 完成后分析
LATEST=$(ls -td results_optimized_sweep_* | head -1)
python3 visualize_results.py --input $LATEST

# 4. 提取最佳配置
cat $LATEST/sweep_stats.json | jq '.best_under_1k'

# 5. 对比多个策略
python3 compare_strategies.py \
  --dirs results_optimized_sweep_*/ \
  --output ./comparison_results
```

---

**祝优化顺利！** 🚀
