# 完整数据集压缩测试 - 启动指南

## 测试概览

本次测试将对全部 2040 个样本进行三阶段压缩评测，总计 **58 组配置**：

### 阶段 1: PCA 大扫描 (32 组)
- K = [32, 48, 64, 96]
- r = [8, 12, 16, 24, 32, 48, 64, 96]
- 预计时间: ~10-15 分钟

### 阶段 2: VAE 小网格 (18 组)
- K = [48, 64]
- r = [16, 32, 48]
- arch = [2x512]
- beta = [0.0, 0.0001, 0.001]
- epochs = 30
- 预计时间: ~30-45 分钟

### 阶段 3: VAE 精炼 (8 组)
- K = [48, 64]
- r = [32, 48]
- arch = [3x1024]
- beta = [0.0, 0.001]
- epochs = 60
- 预计时间: ~45-60 分钟

**总预计时间**: 约 1.5-2 小时

## 快速启动

### 方法 1: 一键启动（推荐）

```bash
./run_complete_sweep.sh
```

### 方法 2: 后台运行（推荐用于远程服务器）

```bash
nohup ./run_complete_sweep.sh > sweep.out 2>&1 &
```

然后可以通过以下命令监控进度：

```bash
# 查看实时日志
tail -f complete_sweep/sweep_*.log

# 或查看输出
tail -f sweep.out
```

### 方法 3: 使用 screen/tmux（推荐用于长时间运行）

```bash
# 使用 screen
screen -S compress_test
./run_complete_sweep.sh
# Ctrl+A, D 分离会话
# screen -r compress_test 重新连接

# 使用 tmux
tmux new -s compress_test
./run_complete_sweep.sh
# Ctrl+B, D 分离会话
# tmux attach -t compress_test 重新连接
```

## 监控进度

### 实时查看日志

```bash
tail -f complete_sweep/sweep_*.log
```

### 查看当前阶段

```bash
ls -lh complete_sweep/
```

你会看到：
- `stage1_pca_sweep/` - 阶段 1 完成后出现
- `stage2_vae_grid/` - 阶段 2 完成后出现
- `stage3_vae_refine/` - 阶段 3 完成后出现

### 查看部分结果

即使测试未完成，你也可以查看已完成阶段的结果：

```bash
# 查看阶段 1 结果
cat complete_sweep/stage1_pca_sweep/results_summary.json | python3 -m json.tool | head -50

# 查看阶段 2 结果
cat complete_sweep/stage2_vae_grid/results_summary.json | python3 -m json.tool | head -50
```

## 测试完成后

测试完成后，你将获得以下文件：

```
complete_sweep/
├── sweep_YYYYMMDD_HHMMSS.log          # 完整日志
├── all_results_combined.json          # 所有 58 组结果汇总
├── ranking_report.txt                 # Top 20 配置排名报告
├── stage1_pca_sweep/
│   ├── results_summary.json
│   └── result_PCA_*.json (32 个)
├── stage2_vae_grid/
│   ├── results_summary.json
│   ├── result_VAE_*.json (18 个)
│   └── vae_*.pt (18 个模型)
└── stage3_vae_refine/
    ├── results_summary.json
    ├── result_VAE_*.json (8 个)
    └── vae_*.pt (8 个模型)
```

### 查看最佳配置

```bash
cat complete_sweep/ranking_report.txt
```

这会显示 Top 20 配置，包括：
- WRMSE（主要评价指标）
- 各组件 RMSE（q, dq, tau）
- 压缩率
- 来源阶段

### 快速分析

```python
import json
import pandas as pd

# 读取所有结果
with open('complete_sweep/all_results_combined.json') as f:
    results = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame([
    {
        'method': r['method'],
        'K': r['K'],
        'r': r['latent'],
        'wrmse': r['metrics']['wrmse'],
        'rmse_q': r['metrics']['rmse_q'],
        'rmse_dq': r['metrics']['rmse_dq'],
        'rmse_tau': r['metrics']['rmse_tau'],
        'compression_ratio': (2040 * 7853 * 21) / (2040 * r['latent']),
        'stage': r['stage']
    }
    for r in results
])

# 按 WRMSE 排序
df_sorted = df.sort_values('wrmse')
print(df_sorted.head(10))

# 按压缩率分组统计
print("\n压缩率 vs WRMSE:")
df['ratio_group'] = pd.cut(df['compression_ratio'], bins=[0, 500, 1000, 2000, 5000, 10000])
print(df.groupby('ratio_group')['wrmse'].describe())
```

## 测试策略说明

### 为什么分三个阶段？

1. **阶段 1 (PCA)**: 快速建立 baseline，确定合理的 K 和 r 范围
2. **阶段 2 (VAE 小)**: 用轻量模型快速探索 VAE 性能
3. **阶段 3 (VAE 精)**: 用高容量模型验证最优配置

### 选择标准

根据你的需求：

**重建质量优先**：
- 选择 WRMSE 最小的配置
- VAE 首选 beta=0.0（纯重建，无正则化）
- 较大的 r 值（32-64）

**压缩率优先**：
- 选择较小的 r 值（8-16）
- 同等误差下选更小的 r
- PCA 作为 baseline（训练快、确定性）

**平衡方案**：
- r=16-32
- VAE beta=0.001（轻微正则化，潜空间更平滑）
- K=48-64

## 中断与恢复

如果测试中断，你可以：

### 方法 1: 继续运行未完成的阶段

根据已完成的阶段手动运行后续阶段：

```bash
# 如果阶段 1 完成，运行阶段 2
python3 compress_benchmark.py \
  --data test_complete.npz \
  --out complete_sweep/stage2_vae_grid \
  --methods VAE \
  --K 48 64 --latent 16 32 48 \
  --vae-arch 2x512 --beta 0.0 0.0001 0.001 \
  --epochs 30 --batch-size 128 --lr 1e-3

# 然后运行阶段 3
python3 compress_benchmark.py \
  --data test_complete.npz \
  --out complete_sweep/stage3_vae_refine \
  --methods VAE \
  --K 48 64 --latent 32 48 \
  --vae-arch 3x1024 --beta 0.0 0.001 \
  --epochs 60 --batch-size 128 --lr 1e-3
```

### 方法 2: 只运行特定配置

如果你发现某些配置特别感兴趣，可以单独运行：

```bash
python3 compress_benchmark.py \
  --data test_complete.npz \
  --out complete_sweep/custom_test \
  --methods PCA \
  --K 64 --latent 32
```

## 系统资源

### GPU 内存要求

- 阶段 1 (PCA): CPU only, ~4GB RAM
- 阶段 2 (2x512): ~2-4GB GPU 内存
- 阶段 3 (3x1024): ~4-8GB GPU 内存

如果 GPU 内存不足，可以：
1. 减小 `--batch-size`（128 → 64 或 32）
2. 运行时设置 `CUDA_VISIBLE_DEVICES=-1` 强制使用 CPU

### CPU 运行

如果没有 GPU：

```bash
CUDA_VISIBLE_DEVICES=-1 ./run_complete_sweep.sh
```

预计时间会增加 3-5 倍。

## 故障排除

### 错误: CUDA out of memory

```bash
# 方法 1: 减小 batch size
# 编辑 run_complete_sweep.sh，将 --batch-size 128 改为 64

# 方法 2: 使用 CPU
CUDA_VISIBLE_DEVICES=-1 ./run_complete_sweep.sh
```

### 错误: 数据文件找不到

确保 `test_complete.npz` 在当前目录：

```bash
ls -lh test_complete.npz
```

### 测试很慢

这是正常的！全数据集测试需要时间。可以：
1. 先运行快速测试验证流程：`python3 compress_benchmark.py --quick-test`
2. 使用 GPU 加速
3. 在后台运行，不影响其他工作

## 下一步

测试完成后，根据 `ranking_report.txt` 选择最佳配置，然后：

1. **可视化对比**: 随机采样轨迹，绘制原始 vs 重建
2. **误差分析**: 分析不同关节、不同时间段的误差分布
3. **部署应用**: 将最佳模型用于数据压缩/存储
4. **论文撰写**: 整理结果表格和图表

祝测试顺利！有问题随时查看日志文件。 