# B-Spline 压缩 - 快速参考卡片

## 🚀 一键命令

```bash
# 进入目录
cd /home/nvidiapc/Flow_ICLR/fmtorch/research/UniCon/Robot_Experiment/Sim/PinSim/Spline_Test

# 快速测试（2分钟）
bash test_optimized.sh

# 推荐配置（15分钟）
bash run_sweep_optimized.sh --strategy1 --workers 8

# 完整流程（30-40分钟）
bash RUN_ALL.sh
```

---

## 📊 优化策略速查

| 策略 | 配置数 | 耗时 | 目标 | 命令 |
|-----|--------|------|------|------|
| **策略1** | ~144 | 15分钟 | 固定q高保真 + 优化dq/tau | `bash run_sweep_optimized.sh --strategy1` |
| **策略2** | ~30 | 5分钟 | 三种1k维预算方案 | `bash run_sweep_optimized.sh --strategy2` |
| **策略3** | ~48 | 8分钟 | q导数评估dq（鲁棒） | `bash run_sweep_optimized.sh --strategy3` |
| **完整** | ~800 | 1-2小时 | 全局最优 | `bash run_sweep_optimized.sh --full` |

---

## 🎯 性能目标

### 当前基线
```
WRMSE: 0.264
  rmse_q:   0.029 ✓
  rmse_dq:  0.130 ⚠️
  rmse_tau: 0.808 ❌
```

### 优化目标
```
WRMSE: < 0.01
  rmse_q:   < 0.005
  rmse_dq:  < 0.02
  rmse_tau: < 0.05
潜变量: ≤ 1000 维
```

---

## 🔑 关键参数

### 控制点数 M
- **q**: 128 (896维) - 锁定高保真
- **dq**: 16-48 - 平衡精度与维度
- **tau**: 16-64 - 复杂曲线需要更多

### 样条阶数 degree
- **3**: 标准，C² 连续
- **5**: 更平滑，适合 tau 大瞬态

### 正则化 lambda
- **1e-6**: 标准，最高精度
- **1e-5**: 轻度，抑制振铃
- **1e-4**: 中度，换稳定性

---

## 💡 推荐配置

### 🥇 平衡型（推荐）
```yaml
Mq=128, Mdq=32, Mtau=32
deg: q=3, dq=3, tau=5
λ: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 1344维
预期 WRMSE: < 0.008
```

### 🥈 低维型（1k预算）
```yaml
Mq=120, Mdq=16, Mtau=8
deg: q=3, dq=3, tau=5
λ: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 1008维
预期 WRMSE: < 0.015
```

### 🥉 鲁棒型（OOD）
```yaml
Mq=128, Mdq=0(from q), Mtau=16
deg: q=3, dq=3, tau=5
λ: q=1e-6, dq=1e-6, tau=1e-5
潜变量: 1008维
预期 WRMSE: < 0.012
```

---

## 📈 结果查看

```bash
# 查看统计
cat results_*/sweep_stats.json

# 提取最佳配置
cat results_*/sweep_stats.json | jq '.best_under_1k'

# 可视化
python3 visualize_results.py --input results_*/

# 对比策略
python3 compare_strategies.py --dirs results_*/ --output comparison/
```

---

## 🛠️ 故障排查

| 问题 | 解决方案 |
|-----|---------|
| 内存不足 | `--max-samples 100 --workers 2 --batch-size 5` |
| CPU过高 | `--workers 1` |
| tau误差大 | `--Mtau 64 96 --deg-tau 5 --lam-tau 1e-5` |

---

## 📁 输出文件

```
results_*/
├── results_summary.json       # 所有配置结果
├── sweep_stats.json           # 统计信息 ⭐
├── sweep_meta.json            # 运行元数据
├── curve_latent_vs_wrmse.png  # 主要曲线 ⭐
└── result_Spline_*.json       # 单个配置详情
```

---

## 🔗 文档链接

- **README.md** - 完整功能说明
- **QUICK_START.md** - 快速开始
- **OPTIMIZATION_GUIDE.md** - 详细优化策略
- **SUMMARY.md** - 完整总结
- **本文件** - 快速参考

---

## ⚡ 最快路径

```bash
# 1. 快速测试（验证环境）
bash test_optimized.sh

# 2. 运行推荐策略
bash run_sweep_optimized.sh --strategy1

# 3. 查看结果
LATEST=$(ls -td results_optimized_sweep_* | head -1)
cat $LATEST/sweep_stats.json | jq '.best_under_1k'

# 4. 可视化
python3 visualize_results.py --input $LATEST

# 完成！🎉
```

---

**提示**: 首次运行建议使用 `test_optimized.sh` 验证环境，然后运行 `strategy1` 获得最佳结果。
