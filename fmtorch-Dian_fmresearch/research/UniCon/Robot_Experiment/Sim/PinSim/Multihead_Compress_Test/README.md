# 多头 PCA 压缩基准测试

q / dq / τ 三组分别做 DCT→PCA→IDCT 的独立压缩重建

## 特性

- **多头架构**: 每组（q, dq, tau）独立的 DCT 截断 + PCA 降维
- **灵活配置**: 可分别设置 K 和 r 参数
- **预滤波**: 支持 none / EMA / Savitzky-Golay / Butterworth
- **映射导出**: 可保存 PCA 参数用于后续推理
- **网格搜索**: 自动化 sweep 多种配置
- **多进程加速**: 利用多核 CPU 并行处理，显著提升速度

## 项目结构

```
Multihead_Compress_Test/
├── utils.py                      # 工具函数
├── filters.py                    # 预滤波器
├── compress_multigroup_pca.py    # 单配置测试
├── sweep_multigroup_pca.py       # 网格搜索
├── quick_test.sh                 # 快速测试（10样本）
├── run_sweep_small.sh            # 小规模 sweep (latent≤200)
├── run_sweep_medium.sh           # 中规模 sweep (latent 200-400)
├── run_sweep_large.sh            # 大规模 sweep (latent 400-700)
├── run_sweep_full.sh             # 完整 sweep (全部规模)
└── README.md                     # 本文档
```

## 快速开始

### 1. 快速测试（推荐第一步）

验证管道是否正常工作：

```bash
bash quick_test.sh
```

这会用 10 个样本测试一个配置（K=(128,64,64), r=(100,25,25)）。

**预计时间**: < 30 秒

### 2. 运行 Sweep

根据你的需求选择不同规模：

#### 小规模 (latent ≤ 200)

```bash
bash run_sweep_small.sh
```

- **配置数**: ~45 组
- **预计时间**: ~10-15 分钟
- **适用**: 快速探索，低计算成本

#### 中等规模 (latent 200-400)

```bash
bash run_sweep_medium.sh
```

- **配置数**: ~63 组
- **预计时间**: ~15-25 分钟
- **适用**: 平衡压缩率与精度

#### 大规模 (latent 400-700)

```bash
bash run_sweep_large.sh
```

- **配置数**: ~72 组
- **预计时间**: ~20-30 分钟
- **适用**: 高保真重建

#### 完整 Sweep (全部规模)

```bash
bash run_sweep_full.sh
```

- **配置数**: ~500 组
- **预计时间**: ~2-3 小时
- **适用**: 完整评估，找到最优配置

### 3. 自定义配置

```bash
python compress_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./custom_results \
  --Kq 384 --Kdq 192 --Ktau 192 \
  --rq 400 --rdq 70 --rtau 70 \
  --save-mapping
```

## 参数说明

### DCT 截断系数 K

- `--Kq`: q 组的 DCT 系数个数（默认: 256）
- `--Kdq`: dq 组的 DCT 系数个数（默认: 128）
- `--Ktau`: tau 组的 DCT 系数个数（默认: 128）

**建议**: 给 q 更多带宽，因为关节位置最重要

### PCA 潜变量维度 r

- `--rq`: q 组的潜维度（默认: 400）
- `--rdq`: dq 组的潜维度（默认: 50）
- `--rtau`: tau 组的潜维度（默认: 50）

**约束**: r < min(N, K×7)，其中 N 是样本数

### 预滤波器

- `--q-filter`, `--dq-filter`, `--tau-filter`: 各组的滤波规格
  - `none`: 不滤波（默认）
  - `ema:9`: EMA 滤波，窗口=9
  - `sg:21,3`: Savitzky-Golay，窗口=21，多项式阶=3
  - `butter:0.05,4`: Butterworth 低通，截止频率=0.05（Nyquist分数），阶数=4

**建议**: 高保真重建优先用 `none`

### 权重

- `--wq`: q 的 WRMSE 权重（默认: 1.0）
- `--wdq`: dq 的 WRMSE 权重（默认: 0.25）
- `--wtau`: tau 的 WRMSE 权重（默认: 0.1）

## 输出说明

### 文件结构

```
results_xxx/
├── results_summary_multigroup.json   # 所有配置汇总（按WRMSE排序）
├── result_MHPCA_K(...)_r(...).json   # 每个配置的详细结果
├── mapping_MHPCA_K(...)_r(...).npz   # PCA 映射参数（如果 --save-mapping）
└── sweep_YYYYMMDD_HHMMSS.log         # 运行日志（完整sweep）
```

### 评估指标

- `wrmse`: 加权总 RMSE（主要指标）
- `rmse_q`: 关节位置 RMSE（rad）
- `rmse_dq`: 关节速度 RMSE（rad/s）
- `rmse_tau`: 关节力矩 RMSE（Nm）
- `explained_variance_ratio`: 各组 PCA 解释方差比例

### 查看结果

```bash
# 查看汇总
cat results_xxx/results_summary_multigroup.json | python -m json.tool | head -100

# Top 10 配置会自动打印在控制台
```

## 配置网格详情

### 小规模 Sweep

```
K_q ∈ {128, 192, 256}
K_dq, K_τ ∈ {64, 96, 128}

Latent triplets (rq, rdq, rtau):
  (100, 25, 25)  → total = 150
  (120, 30, 30)  → total = 180
  (140, 30, 30)  → total = 200
  (100, 40, 40)  → total = 180
  (80, 50, 50)   → total = 180
```

### 中等规模 Sweep

```
K_q ∈ {192, 256, 384}
K_dq, K_τ ∈ {96, 128, 192}

Latent triplets:
  (200, 50, 50)  → total = 300
  (240, 60, 60)  → total = 360
  (280, 60, 60)  → total = 400
  (250, 70, 70)  → total = 390
  (220, 80, 80)  → total = 380
  (300, 50, 50)  → total = 400
  (280, 70, 70)  → total = 420
```

### 大规模 Sweep

```
K_q ∈ {256, 384, 512}
K_dq, K_τ ∈ {128, 192, 256}

Latent triplets:
  (400, 50, 50)   → total = 500
  (360, 70, 70)   → total = 500
  (320, 90, 90)   → total = 500
  (450, 75, 75)   → total = 600
  (500, 100, 100) → total = 700
  (480, 80, 80)   → total = 640
  (420, 100, 100) → total = 620
  (380, 120, 120) → total = 620
```

## 性能优化建议

### 内存

- 完整数据集（2040样本）: ~4-8 GB RAM
- 如果内存不足，使用 `--max-samples` 限制样本数

### 速度

- PCA 计算主要取决于 N 和 K×J
- 大 K 值会显著增加计算时间
- 建议先用小规模 sweep 找到合理范围，再精炼

### 并行加速 + 智能资源管理 ⚡

代码已内置多进程支持和智能资源监控，**防止系统冻结**。

```bash
python sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results \
  --num-workers 16 \
  --cpu-threshold 80.0 \
  --memory-threshold 80.0 \
  --batch-size 30 \
  ...
```

**核心安全特性**:
- ✅ 实时监控 CPU 和内存使用率
- ✅ 动态调整并发数防止过载
- ✅ 分批处理避免一次性启动过多进程
- ✅ 资源过载时自动暂停或终止
- ✅ 批次间自动休息让系统恢复

**性能提升**: 在 32 核系统上，使用 12-18 个 workers 可将速度提升 **10-15 倍**！

**推荐配置**:
- 小规模测试: 8-12 workers, batch-size 20
- 大规模测试: 16-18 workers, batch-size 30-50
- 系统负载高时: 降低 threshold 到 70-75%

详细说明见: **[RESOURCE_MANAGEMENT.md](RESOURCE_MANAGEMENT.md)**

## 与单头 PCA 对比

| 方法 | 优势 | 劣势 |
|------|------|------|
| **单头** | 简单，参数少 | q/dq/tau 共享带宽，q精度受限 |
| **多头** | 可给q更多资源，精度更高 | 参数更多，需要调优 |

**预期提升**: 多头架构在相同总 latent 下，`rmse_q` 应显著低于单头

## 故障排除

### 错误: n_components must be...

**原因**: r 超过了最大成分数

**解决**: 
- 减小 r 值
- 增加样本数（去掉 `--max-samples`）

### 滤波器错误

**原因**: 滤波器参数格式错误

**解决**: 检查格式，例如 `sg:21,3`（逗号分隔）

### 内存不足

**解决**:
1. 使用 `--max-samples` 限制样本
2. 减小 K 值
3. 分批运行不同配置

## 下一步

测试完成后：

1. **分析结果**: 查看 `results_summary_multigroup.json`
2. **选择最优**: 根据 WRMSE 和压缩率选择
3. **可视化**: 类似单头的可视化脚本
4. **部署**: 使用保存的映射进行在线压缩

## 高级用法

### 加入预滤波对比

```bash
python sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results_with_filters \
  --filters none ema:9 sg:21,3 \
  --latent-triplets "400,50,50" "360,70,70" \
  ...
```

### 导出所有映射

在 sweep 命令中加 `--save-mapping`

祝测试顺利！

