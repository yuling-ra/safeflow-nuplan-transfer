# 机器人轨迹压缩基准测试

基于 DCT→PCA 和 DCT→VAE 的统一压缩评测框架。

## 项目结构

```
Compress_Test/
├── test_complete.npz          # 输入数据 (2040条轨迹, 形状: q_log/dq_log/tau_log = 2040×7853×7)
├── utils.py                   # 工具函数: DCT变换、评估指标、数据加载
├── models_vae.py              # VAE模型定义
├── compress_benchmark.py      # 主测试脚本
├── run_benchmark.sh           # 完整测试启动脚本
├── requirements.txt           # Python依赖
└── README.md                  # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试（5个样本）

测试管道是否正常工作：

```bash
python compress_benchmark.py --quick-test
```

这将：
- 只使用 10 个样本
- 简化配置：K=[48], latent=[8], arch=['2x256'], beta=[0.001]
- 训练 10 epochs
- 输出到 `./quick_test_results/`

预计运行时间：< 1 分钟

### 3. 完整测试

#### 方法 A: 使用脚本

```bash
bash run_benchmark.sh
```

#### 方法 B: 直接运行 Python

```bash
python compress_benchmark.py \
  --data test_complete.npz \
  --out ./results \
  --methods PCA VAE \
  --K 48 64 \
  --latent 16 32 64 \
  --vae-arch 2x512 3x1024 \
  --beta 0.0 0.001 \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-3
```

## 参数说明

### 数据相关
- `--data`: 输入 npz 文件路径（默认: `test_complete.npz`）
- `--out`: 输出目录（默认: `./results`）

### 方法选择
- `--methods`: 测试方法列表（默认: `PCA VAE`）

### DCT 参数
- `--K`: DCT 截断频率系数个数（默认: `48 64`）

### 降维参数
- `--latent`: 潜在空间维度（默认: `16 32 64`）

### VAE 参数
- `--vae-arch`: VAE 架构（格式: `层数x隐藏维度`，默认: `2x512 3x1024`）
- `--beta`: β-VAE KL 权重（默认: `0.0 0.001`）
- `--epochs`: 训练轮数（默认: `60`）
- `--batch-size`: 批大小（默认: `128`）
- `--lr`: 学习率（默认: `1e-3`）
- `--dropout`: Dropout 概率（默认: `0.0`）

### 加权设置
- `--wq`: 关节位置权重（默认: `1.0`）
- `--wdq`: 关节速度权重（默认: `0.25`）
- `--wtau`: 关节力矩权重（默认: `0.1`）

### 其他
- `--seed`: 随机种子（默认: `42`）
- `--quick-test`: 快速测试模式（5个样本）

## 输出说明

### 文件结构

```
results/
├── results_summary.json              # 所有配置的汇总结果
├── run_meta.json                     # 运行元数据
├── result_PCA_K48_r16_*.json         # 每个配置的详细结果
├── result_PCA_K48_r32_*.json
├── result_VAE_K48_r16_2by512_*.json
├── vae_K48_r16_2by512.pt             # VAE 模型权重
└── ...
```

### 评估指标

每个结果包含：
- `wrmse`: 加权 RMSE（总体）
- `rmse_q`: 关节位置 RMSE
- `rmse_dq`: 关节速度 RMSE
- `rmse_tau`: 关节力矩 RMSE
- `rmse_all`: 未加权总 RMSE
- `explained_variance_ratio`: (仅 PCA) 解释方差比例

## 数据流程

1. **加载数据**: 从 `test_complete.npz` 读取 `q_log`, `dq_log`, `tau_log`
2. **组合**: 拼接为 `(N, T, 21)` 形状，顺序为 `[q(7) | dq(7) | tau(7)]`
3. **标准化**: Z-score 归一化每个通道
4. **加权**: 对 q/dq/tau 应用不同权重（可选）
5. **DCT**: 时间轴做 DCT-II，截断到 K 个系数 → `(N, K, 21)`
6. **展平**: `(N, K, 21)` → `(N, K×21)`
7. **降维**: PCA 或 VAE 降维到 r 维
8. **重建**: 逆操作恢复到原始空间
9. **评估**: 计算各项误差指标

## 压缩率计算

原始数据大小：`N × T × 21` (2040 × 7853 × 21 ≈ 337M floats)

压缩后大小：
- **DCT+PCA**: `N × r` (2040 × 16 ≈ 32K floats) → **压缩率 10,500:1**
- **DCT+VAE**: `N × r + 模型参数` 

## 示例结果

```
PCA K= 48 r= 16 | WRMSE=0.012345 | EVR=0.9876
PCA K= 48 r= 32 | WRMSE=0.008901 | EVR=0.9912
VAE K= 48 r= 16 2x512   β=0.0010 | WRMSE=0.011234
VAE K= 48 r= 32 2x512   β=0.0010 | WRMSE=0.009012
```

## 高级用法

### 自定义参数网格

```bash
python compress_benchmark.py \
  --data test_complete.npz \
  --out ./custom_sweep \
  --methods PCA \
  --K 32 48 64 96 128 \
  --latent 8 16 24 32 48 64 \
  --wq 1.0 --wdq 0.5 --wtau 0.2
```

### 只测试 VAE

```bash
python compress_benchmark.py \
  --methods VAE \
  --K 64 \
  --latent 32 \
  --vae-arch 3x1024 4x2048 \
  --beta 0.0001 0.001 0.01 \
  --epochs 100
```

### 在不同权重配置下测试

需要多次运行，每次使用不同的 `--wq/--wdq/--wtau` 参数。

## 故障排除

### 依赖问题

如果 `orjson` 安装失败，可以修改 `utils.py` 使用标准 `json` 库：

```python
import json
def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
```

### 内存不足

如果内存不足，可以：
1. 减少 `--K` 值
2. 减少 `--latent` 值
3. 使用 `--quick-test` 模式
4. 分批运行不同配置

### GPU 不可用

代码会自动检测并回退到 CPU。VAE 训练在 CPU 上会较慢，建议：
1. 减少 `--epochs`
2. 使用更小的 `--vae-arch`（如 `2x256`）

## 引用

基于论文中描述的 DCT+PCA/VAE 压缩方法。详见：`dct_pca_vs_vae_压缩基准测试（_2040_7853_21_）.md` 