# 📑 B-Spline 压缩测试框架 - 文件索引

## 🎯 快速导航

### 我想...

| 目标 | 查看文档 | 运行命令 |
|-----|---------|---------|
| **快速开始** | [QUICK_START.md](QUICK_START.md) | `bash test_optimized.sh` |
| **了解优化策略** | [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) | `bash run_sweep_optimized.sh --strategy1` |
| **查看快速参考** | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | - |
| **完整了解框架** | [SUMMARY.md](SUMMARY.md) | - |
| **技术细节** | [README.md](README.md) | - |
| **部署状态** | [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) | - |
| **一键运行全部** | [RUN_ALL.sh](RUN_ALL.sh) | `bash RUN_ALL.sh` |

---

## 📂 文件分类

### 🎓 文档（7个）

#### 入门文档
- **[QUICK_START.md](QUICK_START.md)** (4.7K) - 快速开始指南 ⭐ 首先阅读
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** (3.6K) - 快速参考卡片 ⭐ 随时查阅

#### 深入文档
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** (6.8K) - 详细优化策略 ⭐ 核心文档
- **[SUMMARY.md](SUMMARY.md)** (9.4K) - 完整总结
- **[README.md](README.md)** (4.9K) - 功能说明

#### 状态文档
- **[DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md)** - 部署完成报告
- **[INDEX.md](INDEX.md)** - 本文件（文件索引）

---

### 🔧 核心模块（4个）

- **[spline_utils.py](spline_utils.py)** (2.7K) - B-Spline 基函数、编码/解码
  - `normalized_times()` - 时间归一化
  - `open_uniform_knots()` - 开区间均匀结
  - `bspline_design_matrix()` - 构造基矩阵
  - `bspline_deriv_design_matrix()` - 导数基矩阵
  - `ridge_pinv()` - 岭回归伪逆
  - `encode_theta()` / `decode_signal()` - 编码/解码

- **[utils.py](utils.py)** (3.4K) - 数据加载、标准化、评估
  - `load_dataset()` - 加载 NPZ 数据
  - `zscore_fit_stats()` / `zscore_apply()` / `zscore_inv()` - Z-score 标准化
  - `rmse()` / `group_rmse()` / `weighted_rmse()` - 评估指标
  - `save_json()` - JSON 保存

- **[resource_monitor.py](resource_monitor.py)** (4.6K) - 系统资源监控
  - `ResourceMonitor` 类 - 资源监控器
  - `check_resources()` - 检查 CPU/内存
  - `get_safe_worker_count()` - 动态调整 workers
  - `wait_for_resources()` - 等待资源恢复
  - `print_system_info()` - 打印系统信息

- **[requirements.txt](requirements.txt)** (93B) - Python 依赖
  ```
  numpy>=1.22
  scipy>=1.10
  scikit-learn>=1.2
  matplotlib>=3.7
  orjson>=3.9
  tqdm>=4.66
  psutil>=5.9
  ```

---

### 🧪 测试脚本（4个）

- **[spline_benchmark.py](spline_benchmark.py)** (12K) - 单次基准测试
  - `run_spline_compression()` - 运行单次压缩-重建
  - 支持网格扫描（M × degree）
  - 生成详细结果 JSON

- **[sweep_spline.py](sweep_spline.py)** (13K) - 标准网格扫描
  - `run_sweep()` - 运行网格扫描
  - `worker_task()` - 并行工作任务
  - 支持 5 种预设模式（quick/small/medium/large/full）
  - 资源监控 + 批处理

- **[sweep_spline_advanced.py](sweep_spline_advanced.py)** (17K) - 高级扫描 ⭐
  - `run_spline_compression_advanced()` - 高级压缩（支持 q 导数）
  - 支持多档位 lambda 扫描
  - 支持从 q 导数评估 dq（Mdq=0）
  - 自动找到 ≤1000维 最佳配置

- **[visualize_results.py](visualize_results.py)** (7.4K) - 结果可视化
  - `plot_params_vs_rmse()` - 参数量 vs RMSE 曲线
  - `plot_compression_ratio()` - 压缩比 vs RMSE
  - `print_summary()` - 打印结果摘要
  - 生成 PNG 图表

---

### 🚀 启动脚本（4个）

- **[run_sweep.sh](run_sweep.sh)** (3.7K) - 标准扫描
  - 5 种预设模式: quick/small/medium/large/full
  - 自动检查依赖
  - 生成结果摘要

- **[run_sweep_optimized.sh](run_sweep_optimized.sh)** (6.7K) - 优化扫描 ⭐
  - **策略1**: 固定 q 高保真 + 优化 dq/tau (~144配置)
  - **策略2**: 三种预算方案 (~30配置)
  - **策略3**: 从 q 导数评估 dq (~48配置)
  - **完整**: 全局最优 (~800配置)

- **[test_optimized.sh](test_optimized.sh)** (2.5K) - 快速测试
  - 使用 10 个样本验证
  - 测试策略1 和策略3
  - 生成对比分析
  - 约 2 分钟完成

- **[RUN_ALL.sh](RUN_ALL.sh)** (12K) - 一键运行全部 ⭐
  - 依次运行所有策略
  - 自动生成对比分析
  - 生成可视化图表
  - 生成最终报告
  - 约 30-40 分钟完成

---

### 📊 分析工具（1个）

- **[compare_strategies.py](compare_strategies.py)** (7.9K) - 多策略对比
  - `plot_pareto_frontier()` - Pareto 前沿
  - `plot_rmse_breakdown()` - RMSE 分解对比
  - `print_comparison_table()` - 对比表格
  - 生成对比摘要 JSON

---

### 📄 参考文档（1个）

- **[spline_sweep：不同密度_阶数的样条压缩重建（含曲线图）.md](spline_sweep：不同密度_阶数的样条压缩重建（含曲线图）.md)** (15K)
  - 原始设计文档
  - 方法说明
  - 示例代码

---

## 🎯 使用场景

### 场景1: 我是新手，想快速开始
```bash
# 1. 阅读快速开始
cat QUICK_START.md

# 2. 运行快速测试
bash test_optimized.sh

# 3. 查看结果
cat test_optimized_results/strategy1/sweep_stats.json
```

### 场景2: 我想获得最佳结果
```bash
# 1. 阅读优化指南
cat OPTIMIZATION_GUIDE.md

# 2. 运行策略1（推荐）
bash run_sweep_optimized.sh --strategy1 --workers 8

# 3. 可视化
LATEST=$(ls -td results_optimized_sweep_* | head -1)
python3 visualize_results.py --input $LATEST
```

### 场景3: 我想对比所有策略
```bash
# 1. 运行完整流程
bash RUN_ALL.sh

# 2. 查看最终报告
cat results_full_optimization_*/FINAL_REPORT.md
```

### 场景4: 我想自定义配置
```bash
# 1. 查看技术文档
cat README.md

# 2. 使用高级扫描脚本
python3 sweep_spline_advanced.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./my_custom_results \
  --Mq 128 \
  --Mdq 32 \
  --Mtau 32 \
  --deg-q 3 \
  --deg-tau 5 \
  --lam-tau 1e-5 \
  --workers 8
```

---

## 📈 优化策略速查

| 策略 | 脚本 | 配置数 | 耗时 | 目标 WRMSE |
|-----|------|--------|------|-----------|
| **策略1** | `run_sweep_optimized.sh --strategy1` | ~144 | 15分钟 | < 0.01 |
| **策略2** | `run_sweep_optimized.sh --strategy2` | ~30 | 5分钟 | < 0.02 |
| **策略3** | `run_sweep_optimized.sh --strategy3` | ~48 | 8分钟 | < 0.015 |
| **完整** | `run_sweep_optimized.sh --full` | ~800 | 1-2小时 | 全局最优 |
| **全部** | `RUN_ALL.sh` | ~222 | 30-40分钟 | 对比分析 |

---

## 🔑 关键概念

### 控制点数 M
- 决定样条的灵活性
- 越大越精细，但占用更多维度
- 推荐: q=128, dq=16-48, tau=16-64

### 样条阶数 degree
- 决定曲线的平滑度
- 3: C² 连续（标准）
- 5: C⁴ 连续（更平滑，适合 tau）

### 正则化 lambda
- 决定拟合的稳定性
- 1e-6: 标准（最高精度）
- 1e-5: 轻度（抑制振铃）
- 1e-4: 中度（换稳定性）

### 从 q 导数评估 dq
- Mdq=0 时自动启用
- 使用 B'θ 计算 dq
- 不占额外维度
- 更抗 OOD

---

## 📊 输出文件结构

```
results_*/
├── results_summary.json           # 所有配置结果
├── sweep_stats.json               # 统计信息 ⭐
│   ├── wrmse_min/max/mean/median
│   ├── latent_min/max/mean
│   └── best_under_1k              # ≤1000维最佳配置 ⭐
├── sweep_meta.json                # 运行元数据
├── result_Spline_*.json           # 单个配置详情
├── curve_latent_vs_wrmse.png      # 主要曲线 ⭐
├── curve_latent_vs_rmse_groups.png # 分组曲线
└── curve_compression_vs_wrmse.png  # 压缩比曲线
```

---

## 🛠️ 工具函数速查

### 数据处理
```python
from utils import load_dataset, zscore_apply, zscore_inv
dataset = load_dataset('data.npz', max_samples=10)
q, dq, tau = dataset['q'], dataset['dq'], dataset['tau']
```

### B-Spline 编码
```python
from spline_utils import *
t = normalized_times(T)
B = bspline_design_matrix(t, M=128, degree=3)
P = ridge_pinv(B, lam=1e-6)
theta = encode_theta(P, q[0])  # (M, J)
q_rec = decode_signal(B, theta)  # (T, J)
```

### 资源监控
```python
from resource_monitor import ResourceMonitor
monitor = ResourceMonitor(cpu_threshold=85.0)
safe_workers = monitor.get_safe_worker_count(requested=8)
```

---

## 📞 获取帮助

### 命令行帮助
```bash
# 标准扫描
bash run_sweep.sh --help

# 优化扫描
bash run_sweep_optimized.sh --help

# Python 脚本
python3 sweep_spline_advanced.py --help
```

### 文档帮助
- 快速问题: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- 优化问题: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
- 技术问题: [README.md](README.md)
- 故障排查: [README.md](README.md) 的故障排查章节

---

## ✅ 检查清单

### 首次使用
- [ ] 阅读 [QUICK_START.md](QUICK_START.md)
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 运行快速测试: `bash test_optimized.sh`
- [ ] 查看结果: `cat test_optimized_results/strategy1/sweep_stats.json`

### 正式运行
- [ ] 阅读 [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
- [ ] 选择策略（推荐策略1）
- [ ] 运行扫描: `bash run_sweep_optimized.sh --strategy1`
- [ ] 可视化结果: `python3 visualize_results.py --input results_*/`
- [ ] 选择最佳配置

### 完整分析
- [ ] 运行全部策略: `bash RUN_ALL.sh`
- [ ] 查看最终报告: `cat results_full_optimization_*/FINAL_REPORT.md`
- [ ] 对比分析: `cat results_full_optimization_*/comparison/comparison_summary.json`
- [ ] 应用到 Flow Matching

---

**准备就绪！** 选择您的使用场景，开始优化吧！🚀
