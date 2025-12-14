# 多头 PCA 测试 - 快速启动指南

## 一键启动命令

### 快速验证（推荐第一步）

```bash
cd /home/nvidiapc/Flow_ICLR/fmtorch/research/UniCon/Robot_Experiment/Sim/PinSim/Multihead_Compress_Test
bash quick_test.sh
```

**作用**: 用 10 个样本测试管道是否正常  
**时间**: < 30 秒

---

### 小规模 Sweep (latent ≤ 200)

```bash
bash run_sweep_small.sh
```

**配置数**: ~45 组  
**时间**: ~10-15 分钟  
**适用**: 快速探索，找到合理范围

---

### 中等规模 Sweep (latent 200-400)

```bash
bash run_sweep_medium.sh
```

**配置数**: ~63 组  
**时间**: ~15-25 分钟  
**适用**: 平衡压缩率与精度

---

### 大规模 Sweep (latent 400-700)

```bash
bash run_sweep_large.sh
```

**配置数**: ~72 组  
**时间**: ~20-30 分钟  
**适用**: 高保真重建，最大化精度

---

### 完整 Sweep (全部规模)

```bash
bash run_sweep_full.sh
```

**配置数**: ~500 组  
**时间**: ~2-3 小时  
**适用**: 完整评估，覆盖所有可能性

---

## 后台运行（推荐用于长时间测试）

```bash
# 使用 nohup
nohup bash run_sweep_full.sh > full_sweep.log 2>&1 &

# 监控进度
tail -f full_sweep.log

# 或查看具体结果目录
ls -lh results_full_sweep_*/
```

---

## 查看结果

### 实时监控

```bash
# 查看已生成的结果数量
ls results_*/result_MHPCA_*.json | wc -l

# 查看最新结果
ls -lt results_*/result_MHPCA_*.json | head -5
```

### 查看汇总

```bash
# 完成后查看 Top 10
cat results_*/results_summary_multigroup.json | python -m json.tool | head -200
```

---

## 自定义配置示例

### 测试特定配置

```bash
python compress_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./test_config_1 \
  --Kq 384 --Kdq 192 --Ktau 192 \
  --rq 400 --rdq 70 --rtau 70 \
  --save-mapping
```

### 自定义网格

```bash
python sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./custom_sweep \
  --Kq 256 384 512 \
  --Kdq 128 192 \
  --Ktau 128 192 \
  --latent-triplets "350,75,75" "400,100,100" "450,125,125" \
  --filters none
```

---

## 预期结果

### 与单头 PCA 对比

基于单头测试的最佳结果（K=96, r=64）:
- WRMSE: ~1.06
- RMSE_q: ~0.59

**多头预期**（相同总 latent=500）:
- WRMSE: **< 0.8** (预计降低 20-30%)
- RMSE_q: **< 0.4** (预计降低 30-40%)

原因: 给 q 更多 K 和 r，显著提升位置重建精度

---

## 推荐测试流程

### 第 1 步: 快速验证

```bash
bash quick_test.sh
```

确保代码运行无误

### 第 2 步: 小规模探索

```bash
bash run_sweep_small.sh
```

快速找到合理的 K 和 r 范围

### 第 3 步: 精细调优

根据小规模结果，选择一个规模继续：

```bash
# 如果小规模表现好，试中等规模
bash run_sweep_medium.sh

# 如果需要更高精度，试大规模
bash run_sweep_large.sh
```

### 第 4 步: 完整评估（可选）

```bash
bash run_sweep_full.sh
```

只在有充足时间时运行

---

## 故障排除

### 问题 1: 找不到数据文件

**现象**: `FileNotFoundError: test_complete.npz`

**解决**: 检查数据文件路径
```bash
ls -lh ../Compress_Test/test_complete.npz
```

如果不在默认位置，修改脚本第一行：
```bash
DATA=/path/to/your/test_complete.npz
```

### 问题 2: 内存不足

**现象**: `MemoryError` 或系统变慢

**解决**: 
1. 先用快速测试验证: `bash quick_test.sh`
2. 减小 K 值范围
3. 分批运行不同规模

### 问题 3: orjson 未安装

**现象**: `ModuleNotFoundError: No module named 'orjson'`

**解决**:
```bash
pip install orjson
# 或
pip install -r requirements.txt
```

代码有 fallback 到标准 json，但 orjson 更快

---

## 输出说明

### 目录结构

```
Multihead_Compress_Test/
├── results_small_latent/              # 小规模结果
│   ├── results_summary_multigroup.json
│   ├── result_MHPCA_*.json (多个)
│   └── mapping_MHPCA_*.npz (如果加了 --save-mapping)
├── results_medium_latent/             # 中等规模结果
├── results_large_latent/              # 大规模结果
└── results_full_sweep_YYYYMMDD_HHMMSS/  # 完整sweep结果
    ├── results_summary_multigroup.json
    ├── sweep_YYYYMMDD_HHMMSS.log
    └── result_MHPCA_*.json (500+个)
```

### 关键文件

- `results_summary_multigroup.json`: **最重要**，所有配置按 WRMSE 排序
- `result_MHPCA_K(...)_r(...).json`: 每个配置的详细结果
- `mapping_MHPCA_*.npz`: PCA 映射参数（用于后续推理）

---

## 下一步

测试完成后：

1. **查看最佳配置**: 
   ```bash
   cat results_*/results_summary_multigroup.json | python -m json.tool | head -100
   ```

2. **对比单头结果**: 比较 WRMSE 和 RMSE_q

3. **可视化**: 创建类似单头的可视化脚本

4. **部署应用**: 使用最佳配置的映射进行在线压缩

---

## 预计时间表

### 单进程运行

| 任务 | 时间 | 输出 |
|------|------|------|
| 快速测试 | < 1 分钟 | 1 个配置 |
| 小规模 | 10-15 分钟 | ~45 个配置 |
| 中等规模 | 15-25 分钟 | ~63 个配置 |
| 大规模 | 20-30 分钟 | ~72 个配置 |
| 完整 | 2-3 小时 | ~500 个配置 |

### 多进程加速 (6-8 workers)

| 任务 | 时间 | 加速比 |
|------|------|--------|
| 小规模 | **2-3 分钟** | 5-7x |
| 中等规模 | **3-5 分钟** | 5-7x |
| 大规模 | **4-6 分钟** | 5-7x |
| 完整 | **20-30 分钟** | 5-7x |

**注意**: 所有脚本已默认启用多进程（6-8 workers），享受加速！

---

## 多进程控制

### 查看 CPU 核数

```bash
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"
```

### 自定义 worker 数量

编辑脚本，修改 `--num-workers` 参数：

```bash
# 例如：使用 4 个 workers
python3 sweep_multigroup_pca.py \
  ... \
  --num-workers 4
```

### 性能监控

运行时会显示实时进度和速率：

```
[15/45] ✓ K(256,128,128)_r(200,50,50)_f(none)
    Progress: 15/45 (33.3%) | Rate: 0.52 cfg/s | ETA: 9.6 min
```

建议从小规模开始，根据结果决定是否继续！

祝测试顺利！🚀

