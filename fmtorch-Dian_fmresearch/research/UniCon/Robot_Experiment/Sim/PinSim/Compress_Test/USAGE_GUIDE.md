# 压缩测试框架使用指南

## 概览

本测试框架用于评估机器人轨迹数据（q, dq, tau）的压缩与重建性能，支持：
- **DCT→PCA**: 离散余弦变换 + 主成分分析
- **DCT→VAE**: 离散余弦变换 + 变分自编码器

## 快速开始（推荐）

### 1分钟快速验证管道

```bash
python compress_benchmark.py --quick-test
```

这会用 10 个样本快速测试整个管道，确保代码正常工作。

**输出示例：**
```
PCA K= 48 r=  8 | WRMSE=1.056522 | EVR=0.9987
VAE K= 48 r=  8 2x256   β=0.0010 | WRMSE=1.817397
```

## 完整测试流程

### 场景 1: 全数据集基准测试

测试所有 2040 个样本，完整的参数网格：

```bash
# 使用脚本
bash run_benchmark.sh

# 或直接运行 Python
python compress_benchmark.py \
  --data test_complete.npz \
  --out ./results_full \
  --methods PCA VAE \
  --K 48 64 \
  --latent 16 32 64 \
  --vae-arch 2x512 3x1024 \
  --beta 0.0 0.001 \
  --epochs 60
```

**预计时间**: ~30-60分钟（取决于GPU）

### 场景 2: 只测试 PCA（快速）

```bash
python compress_benchmark.py \
  --methods PCA \
  --K 32 48 64 96 128 \
  --latent 8 16 24 32 48 64 96 128
```

**预计时间**: ~5分钟

### 场景 3: VAE 架构对比

```bash
python compress_benchmark.py \
  --methods VAE \
  --K 64 \
  --latent 32 \
  --vae-arch 2x256 2x512 3x512 3x1024 4x2048 \
  --beta 0.001 \
  --epochs 100
```

### 场景 4: β-VAE 超参数扫描

```bash
python compress_benchmark.py \
  --methods VAE \
  --K 64 \
  --latent 32 \
  --vae-arch 3x1024 \
  --beta 0.0 0.0001 0.0005 0.001 0.005 0.01 0.05 \
  --epochs 80
```

### 场景 5: 不同权重配置对比

需要多次运行：

```bash
# 方案A: 平等权重
python compress_benchmark.py --wq 1.0 --wdq 1.0 --wtau 1.0 --out ./results_equal_weights

# 方案B: 重视关节位置
python compress_benchmark.py --wq 2.0 --wdq 0.5 --wtau 0.2 --out ./results_q_priority

# 方案C: 重视关节速度
python compress_benchmark.py --wq 1.0 --wdq 1.0 --wtau 0.1 --out ./results_dq_priority
```

## 参数调优建议

### DCT 截断系数 K

- **K=32**: 极限压缩，适合存储优先场景
- **K=48**: 平衡压缩率与精度（**推荐起点**）
- **K=64**: 高精度重建
- **K=96-128**: 接近无损重建

**选择策略**: 先用 K=48 测试，观察 RMSE，然后向上/向下调整

### 潜变量维度 r

- **r=8-16**: 超高压缩，适合大规模数据集索引
- **r=16-32**: 平衡点（**推荐**）
- **r=32-64**: 高保真重建
- **r>64**: 逐渐接近 K×21 维，压缩率降低

**选择策略**: r ≈ K/3 到 K/2 是常见的起点

### VAE 架构

格式: `层数x隐藏维度`

- **2x256**: 最轻量，快速实验
- **2x512**: 标准配置（**推荐起点**）
- **3x1024**: 高容量，适合复杂数据
- **4x2048**: 最大容量，训练慢

**选择策略**: 
- 小数据集（<100样本）: 2x256
- 中等数据集（100-1000）: 2x512
- 大数据集（>1000）: 3x1024

### β-VAE 超参数

- **β=0.0**: 无 KL 正则，接近普通 AE
- **β=0.0001-0.001**: 轻微正则（**推荐**）
- **β=0.01-0.1**: 强正则，潜空间更平滑但重建误差增大

**选择策略**: 从 0.001 开始，观察 WRMSE，向上/向下调整

## 结果分析

### 查看汇总结果

```bash
cat results/results_summary.json | python -m json.tool
```

### 关键指标解读

- **wrmse**: 加权总 RMSE（主要指标）
  - < 0.5: 优秀
  - 0.5-1.0: 良好
  - 1.0-2.0: 可接受
  - > 2.0: 需要调参

- **rmse_q**: 关节位置 RMSE（单位：弧度）
  - < 0.1: 优秀
  - 0.1-0.5: 良好

- **rmse_dq**: 关节速度 RMSE（单位：弧度/秒）
  - < 0.2: 优秀
  - 0.2-0.5: 良好

- **rmse_tau**: 关节力矩 RMSE（单位：牛米）
  - < 2.0: 优秀
  - 2.0-5.0: 良好

- **explained_variance_ratio** (仅PCA): 解释方差比例
  - > 0.99: 优秀
  - 0.95-0.99: 良好
  - < 0.95: 可能需要增加 r 或 K

### 对比 PCA vs VAE

一般规律：
- **PCA**: 
  - ✅ 训练快（秒级）
  - ✅ 确定性
  - ✅ 线性变换，易解释
  - ❌ 可能在非线性数据上表现不佳

- **VAE**:
  - ✅ 非线性，可能更好地捕捉复杂模式
  - ✅ 生成能力
  - ❌ 训练慢（分钟级）
  - ❌ 需要调参
  - ❌ 随机性（不同运行结果略有差异）

## 常见问题

### Q: 我应该用多少个样本测试？

**A**: 
- 快速验证: 10-50 样本
- 开发调参: 100-500 样本
- 最终基准: 全部 2040 样本

### Q: PCA 报错 "n_components must be between..."

**A**: 确保 `latent < min(样本数, K×21)`。对于小样本测试，减小 latent 值。

### Q: VAE 训练很慢怎么办？

**A**: 
1. 减少 `--epochs`（从 60 → 30）
2. 使用更小的 `--vae-arch`（3x1024 → 2x512）
3. 增大 `--batch-size`（128 → 256）
4. 确保使用 GPU（代码会自动检测）

### Q: 如何选择最优配置？

**A**: 按以下优先级排序：
1. **WRMSE** 越小越好
2. **压缩率** = (N×T×21) / (N×r) 越大越好
3. **训练时间** 越快越好（PCA优势）

绘制 WRMSE vs 压缩率曲线，找帕累托前沿。

### Q: 权重参数 wq/wdq/wtau 如何设置？

**A**: 取决于应用场景：
- **轨迹回放**: wq=1.0, wdq=0.25, wtau=0.1（默认，重视位置）
- **动力学建模**: wq=1.0, wdq=1.0, wtau=0.5（重视速度）
- **力控制**: wq=0.5, wdq=0.5, wtau=1.0（重视力矩）

## 高级技巧

### 并行测试多个配置

使用 GNU parallel：

```bash
parallel --jobs 4 python compress_benchmark.py --K {} --latent 32 --out ./results_K{} ::: 32 48 64 96
```

### 绘制压缩率-误差曲线

```python
import json
import matplotlib.pyplot as plt
import numpy as np

with open('results/results_summary.json') as f:
    results = json.load(f)

pca_results = [r for r in results if r['method'] == 'PCA']
vae_results = [r for r in results if r['method'] == 'VAE']

# 提取数据
pca_ratios = [r['shapes']['D'] / r['latent'] for r in pca_results]
pca_wrmse = [r['metrics']['wrmse'] for r in pca_results]

plt.scatter(pca_ratios, pca_wrmse, label='PCA')
plt.xlabel('压缩率')
plt.ylabel('WRMSE')
plt.legend()
plt.grid(True)
plt.show()
```

### 批量评估不同数据集

```bash
for dataset in data1.npz data2.npz data3.npz; do
  python compress_benchmark.py --data $dataset --out ./results_$(basename $dataset .npz)
done
```

## 输出文件说明

- `results_summary.json`: 所有配置的汇总（用于后续分析）
- `result_*.json`: 每个配置的详细结果
- `run_meta.json`: 运行元数据（参数、数据形状等）
- `vae_*.pt`: VAE 模型权重（可用于后续推理）

## 下一步

测试完成后，根据结果：

1. **选择最优配置**: 综合考虑 WRMSE、压缩率、训练时间
2. **部署应用**: 使用保存的 PCA 或 VAE 模型进行压缩/重建
3. **可视化**: 随机采样几条轨迹，绘制原始 vs 重建对比图
4. **集成**: 将压缩管道集成到数据收集/训练流程中

祝测试顺利！ 