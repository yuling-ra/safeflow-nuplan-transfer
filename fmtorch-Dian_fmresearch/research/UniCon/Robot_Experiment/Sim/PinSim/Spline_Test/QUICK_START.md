# B-Spline 压缩测试 - 快速开始指南

## 🎯 一键运行

### 1. 快速测试（1分钟验证）

```bash
bash run_sweep.sh --mode quick
```

这将：
- 使用 10 个样本
- 测试 6 个配置
- 约 1 分钟完成
- 输出到 `./results_sweep_TIMESTAMP/`

### 2. 推荐配置（20分钟完整测试）

```bash
bash run_sweep.sh --mode medium --workers 8
```

这将：
- 使用全部样本
- 测试 ~150 个配置
- 约 20 分钟完成
- 自动并行处理（8 workers）

### 3. 查看结果

```bash
# 自动找到最新结果目录
LATEST=$(ls -td results_sweep_* | head -1)
python3 visualize_results.py --input $LATEST
```

## 📊 预期输出

### 终端输出示例

```
======================================================================
系统信息
======================================================================
CPU 核心数: 16
CPU 使用率: 12.3%
内存总量: 62.5 GB
可用内存: 48.2 GB
内存使用率: 22.9%
======================================================================

[数据] 加载 ../Compress_Test/test_complete.npz...
[数据] 形状: N=2040, T=7853, J=7

[扫描] 总配置数: 150
======================================================================

[批次 1/15] 处理配置 1-10/150
  完成: 10/10 配置
  累计: 10/150 配置

...

======================================================================
扫描完成
======================================================================
总配置数: 150
成功: 150
失败: 0
耗时: 00:18:42

WRMSE 统计:
  最小: 0.001234
  最大: 0.045678
  平均: 0.012345
  中位数: 0.010987

最佳配置 (Top 5):
1. M=(96,48,48) deg=(3,3,3) | WRMSE=0.001234 | Latent=1344 | 压缩比=125.67x
2. M=(128,48,48) deg=(3,3,3) | WRMSE=0.001456 | Latent=1568 | 压缩比=107.65x
...
```

### 生成的文件

```
results_sweep_20251005_143022/
├── results_summary.json          # 所有结果（可编程分析）
├── sweep_stats.json               # 统计摘要
├── sweep_meta.json                # 运行元数据
├── curve_latent_vs_wrmse.png      # 主要曲线图
├── curve_latent_vs_rmse_groups.png # 分组曲线图
├── curve_compression_vs_wrmse.png  # 压缩比曲线
└── result_Spline_*.json           # 单个配置详情
```

## 🔧 常见调整

### 调整数据文件

```bash
bash run_sweep.sh \
  --data /path/to/your/data.npz \
  --mode medium
```

### 调整并行度

```bash
# 使用更多 workers（如果 CPU 核心多）
bash run_sweep.sh --mode medium --workers 12

# 使用更少 workers（如果系统资源紧张）
bash run_sweep.sh --mode medium --workers 2
```

### 自定义输出目录

```bash
bash run_sweep.sh \
  --out ./my_custom_results \
  --mode medium
```

## 📈 结果解读

### 查看最佳配置

```bash
cat results_sweep_*/sweep_stats.json | grep -A 5 "wrmse_min"
```

### 提取 Top 10 配置

```bash
python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    results = json.load(f)
top10 = sorted(results, key=lambda x: x['metrics']['wrmse'])[:10]
for i, r in enumerate(top10, 1):
    print(f\"{i}. M={r['M']}, WRMSE={r['metrics']['wrmse']:.6f}\")
" results_sweep_*/results_summary.json
```

### 对比不同阶数

可视化工具会自动按阶数分组绘制曲线，查看：
- `curve_latent_vs_wrmse.png` - 不同阶数的性能对比
- `curve_latent_vs_rmse_groups.png` - q/dq/tau 分别的误差

## ⚠️ 注意事项

### 1. 数据文件格式

确保 NPZ 文件包含：
- `q_log`: (N, T, 7) - 关节位置
- `dq_log`: (N, T, 7) - 关节速度
- `tau_log`: (N, T, 7) - 关节力矩

### 2. 系统资源

- **quick 模式**: 任何配置都可运行
- **medium 模式**: 建议 ≥16GB 内存
- **large/full 模式**: 建议 ≥32GB 内存

### 3. 运行时间

实际时间取决于：
- CPU 核心数
- 数据集大小（N 和 T）
- 并行 worker 数量

## 🆘 遇到问题？

### 内存不足

```bash
# 使用更少样本
bash run_sweep.sh --mode quick  # 只用 10 个样本
```

### CPU 占用过高

```bash
# 减少并行度
bash run_sweep.sh --mode medium --workers 2
```

### 扫描中断

中间结果已保存在 `results_summary_partial.json`，可以查看已完成的配置。

## 📚 下一步

1. **分析结果**: 使用 `visualize_results.py` 生成图表
2. **选择最佳配置**: 根据 WRMSE 和压缩比权衡
3. **导出映射**: 使用最佳配置训练后续模型
4. **对比其他方法**: 与 DCT-PCA、VAE 等方法对比

## 💡 提示

- 首次运行建议使用 `--mode quick` 验证环境
- `medium` 模式适合大多数场景
- 如需发论文级别的完整扫描，使用 `--mode full`
- 可视化工具会自动高亮最佳配置

---

**祝测试顺利！** 🎉
