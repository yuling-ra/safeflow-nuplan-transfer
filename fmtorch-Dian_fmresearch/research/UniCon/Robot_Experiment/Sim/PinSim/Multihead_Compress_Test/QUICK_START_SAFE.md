# 快速开始指南 (安全版)

## 🛡️ 防冻结机制已部署

现在所有测试都集成了智能资源监控，**不会再冻结系统**。

## 第一步：测试资源监控

```bash
# 检查系统状态
python3 resource_monitor.py

# 运行快速测试 (10个配置，验证功能)
./test_resource_monitor.sh
```

预期输出：
```
======================================================================
System Information
======================================================================
CPU Cores: 32
CPU Usage: X.X%
Memory Available: XX.X GB
...
```

## 第二步：小规模测试

```bash
# 保守配置，确保稳定 (60个配置，约 3-5 分钟)
./run_sweep_small.sh
```

配置:
- **Workers**: 12 (38% 核心)
- **Batch Size**: 20 (每批20个配置)
- **CPU Threshold**: 80%
- **Memory Threshold**: 80%

## 第三步：根据需求选择规模

### 选项 A: 中等规模测试
```bash
# 144 个配置，约 10-15 分钟
./run_sweep_medium.sh
```
- Workers: 16, Batch: 30

### 选项 B: 大规模测试
```bash
# 270 个配置，约 20-30 分钟
./run_sweep_large.sh
```
- Workers: 18, Batch: 40

### 选项 C: 完整测试
```bash
# 600 个配置，约 40-60 分钟
./run_sweep_full.sh
```
- Workers: 18, Batch: 50

## 实时监控

在另一个终端运行，实时查看系统状态：

```bash
# 方式1: watch 命令
watch -n 1 'echo "CPU:" && mpstat 1 1 | tail -1 && echo "Memory:" && free -h | grep Mem'

# 方式2: htop (更直观)
htop
```

## 安全特性

测试运行时会显示：

```
[42/200] ✓ MHPCA_Kq256_Kdq128_Ktau128_r400-50-50_none
    Progress: 42/200 (21.0%) | Rate: 2.15 cfg/s | ETA: 12.3 min
    System: CPU 72.3%, Memory 65.8%    ← 实时资源监控
```

如果资源过载：
```
[Resource Monitor] ⚠ WARNING: CPU usage too high: 85.2% > 80.0%
[Resource Monitor] Consider reducing --num-workers or using --batch-size
```

如果持续过载（2分钟），会自动终止并保存已完成的结果。

## 手动调整参数

如果默认配置对你的系统不合适：

```bash
python3 sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./custom_results \
  --Kq 256 384 \
  --Kdq 128 \
  --Ktau 128 \
  --latent-triplets "400,50,50" "360,70,70" "320,90,90" \
  --filters none \
  --num-workers 12 \           # ← 降低并发数
  --cpu-threshold 75.0 \        # ← 更敏感的阈值
  --memory-threshold 75.0 \
  --batch-size 15               # ← 更小的批处理
```

## 系统负载参考

| 当前系统状态 | 推荐配置 |
|------------|---------|
| 空闲 (<30% 负载) | `--num-workers 18 --batch-size 50` |
| 正常 (30-60% 负载) | `--num-workers 16 --batch-size 40` |
| 繁忙 (60-80% 负载) | `--num-workers 12 --batch-size 30` |
| 高负载 (>80% 负载) | `--num-workers 8 --batch-size 20` |

## 后台运行

对于长期任务，建议后台运行：

```bash
# 使用 nohup
nohup ./run_sweep_full.sh > sweep.log 2>&1 &

# 查看进度
tail -f sweep.log

# 查看系统负载
htop
```

## 故障排查

### Q: 测试仍然让系统卡顿？

**A**: 降低资源使用：
```bash
# 更保守的配置
--num-workers 6 --batch-size 10 --cpu-threshold 70.0
```

### Q: 测试频繁暂停？

**A**: 
1. 关闭其他占用资源的程序
2. 放宽阈值：`--cpu-threshold 85.0 --memory-threshold 85.0`
3. 在系统负载低时运行

### Q: 测试速度太慢？

**A**: 提高资源使用（确保系统有余量）：
```bash
--num-workers 20 --batch-size 60 --cpu-threshold 90.0
```

## 检查结果

测试完成后：

```bash
cd results_[timestamp]/
cat results_summary_multigroup.json | head -50  # 查看最佳配置
```

## 详细文档

- 资源管理详解: [RESOURCE_MANAGEMENT.md](RESOURCE_MANAGEMENT.md)
- 完整使用指南: [README.md](README.md)
- 启动指南: [START_GUIDE.md](START_GUIDE.md)

## 关键要点

✅ 从 `test_resource_monitor.sh` 开始  
✅ 使用 `htop` 实时监控  
✅ 先运行 `run_sweep_small.sh` 验证稳定性  
✅ 根据系统负载选择合适的规模  
✅ 资源不足时会自动降级或终止  
✅ 所有进度和结果都会实时保存  

祝测试顺利！🚀

