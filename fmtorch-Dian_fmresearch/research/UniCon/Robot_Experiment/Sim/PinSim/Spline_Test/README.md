# B-Spline 压缩基准测试框架

基于 B-Spline（样条）的轨迹压缩与重建测试框架，支持网格扫描、资源监控、并行处理。

## 📋 功能特性

- ✅ **B-Spline 编码-解码**: 使用开区间均匀结样条，岭回归拟合
- ✅ **多组独立压缩**: q / dq / tau 三组分别处理
- ✅ **网格扫描**: 遍历不同控制点数 M 与样条阶数 p
- ✅ **资源监控**: 实时监控 CPU/内存，防止系统过载
- ✅ **并行处理**: 支持多进程并行，自动批处理
- ✅ **预设模式**: quick/small/medium/large/full 五种预设
- ✅ **可视化**: 自动生成参数量 vs RMSE 曲线图

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试（1分钟）

```bash
bash run_sweep.sh --mode quick
```

### 3. 中等规模扫描（~20分钟）

```bash
bash run_sweep.sh --mode medium --workers 8
```

### 4. 可视化结果

```bash
python3 visualize_results.py --input ./results_sweep_TIMESTAMP
```

## 📊 预设模式

| 模式 | 样本数 | 配置数 | 预计耗时 | 说明 |
|------|--------|--------|----------|------|
| `quick` | 10 | ~6 | ~1分钟 | 快速验证 |
| `small` | 全部 | ~32 | ~5分钟 | 小规模扫描 |
| `medium` | 全部 | ~150 | ~20分钟 | 中等规模（推荐） |
| `large` | 全部 | ~288 | ~45分钟 | 大规模扫描 |
| `full` | 全部 | ~1050 | ~2小时 | 完整扫描 |

## 🔧 自定义参数

### 使用 `spline_benchmark.py`（单次运行）

```bash
python3 spline_benchmark.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results \
  --Mq 16 32 48 64 \
  --Mdq 16 32 \
  --Mtau 16 32 \
  --deg-q 3 5 \
  --deg-dq 3 \
  --deg-tau 3 \
  --lam-q 1e-6 \
  --wq 1.0 --wdq 0.25 --wtau 0.1
```

### 使用 `sweep_spline.py`（网格扫描 + 并行）

```bash
python3 sweep_spline.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results \
  --mode medium \
  --workers 8 \
  --batch-size 15
```

## 📁 输出文件

```
results_sweep_TIMESTAMP/
├── results_summary.json          # 所有配置的结果汇总
├── sweep_meta.json                # 扫描元数据
├── sweep_stats.json               # 统计信息（最佳配置、WRMSE分布等）
├── result_Spline_*.json           # 单个配置的详细结果
├── curve_latent_vs_wrmse.png      # 参数量 vs WRMSE 曲线
├── curve_latent_vs_rmse_groups.png # 分组 RMSE 曲线
└── curve_compression_vs_wrmse.png  # 压缩比 vs WRMSE
```

## 🧮 方法说明

### B-Spline 编码-解码流程

1. **时间归一化**: 将轨迹时间归一到 `[0, 1]`
2. **Z-score 标准化**: 每组（q/dq/tau）独立标准化
3. **构造基矩阵**: 开区间均匀结 B-Spline 基函数 `B(t)`
4. **岭回归编码**: `θ = (B^T B + λI)^{-1} B^T y`
5. **解码重建**: `ŷ = B θ`
6. **反标准化**: 恢复原始尺度
7. **评估误差**: 计算 RMSE、加权 RMSE

### 关键参数

- **M (控制点数)**: 决定样条的灵活性，越大越精细
- **p (阶数)**: 样条次数，3=三次（常用），5=五次（更平滑）
- **λ (正则化)**: 岭回归参数，防止过拟合，默认 `1e-6`

## 🔍 资源监控

框架内置资源监控，自动：
- 监控 CPU/内存使用率
- 动态调整并行 worker 数量
- 等待资源恢复（避免系统崩溃）
- 分批处理（避免内存溢出）

### 手动调整资源阈值

编辑 `sweep_spline.py`:

```python
monitor = ResourceMonitor(
    cpu_threshold=85.0,      # CPU 阈值 (%)
    memory_threshold=85.0,   # 内存阈值 (%)
)
```

## 📈 结果分析

### 查看统计信息

```bash
cat results_sweep_TIMESTAMP/sweep_stats.json
```

### 查看最佳配置

```bash
python3 -c "
import json
with open('results_sweep_TIMESTAMP/results_summary.json') as f:
    results = json.load(f)
best = sorted(results, key=lambda x: x['metrics']['wrmse'])[0]
print(f\"最佳配置: M={best['M']}, deg={best['degree']}, WRMSE={best['metrics']['wrmse']:.6f}\")
"
```

### 可视化对比

```bash
python3 visualize_results.py --input results_sweep_TIMESTAMP
```

## 🛠️ 故障排查

### 问题: 内存不足

**解决方案**:
1. 减少 `--max-samples` 样本数
2. 减少 `--workers` 数量
3. 减小 `--batch-size` 批处理大小
4. 使用更小的预设模式（如 `small`）

### 问题: CPU 占用过高

**解决方案**:
1. 减少 `--workers` 数量
2. 降低资源阈值（编辑 `sweep_spline.py`）
3. 使用串行模式 `--workers 1`

### 问题: 扫描中断

**解决方案**:
- 中间结果已保存在 `results_summary_partial.json`
- 可以从中断处继续（手动调整配置列表）

## 📚 参考文献

- B-Spline 理论: de Boor, C. (1978). *A Practical Guide to Splines*
- 岭回归: Hoerl & Kennard (1970). *Ridge Regression*
- 开区间均匀结: Piegl & Tiller (1997). *The NURBS Book*

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License
