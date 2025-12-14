# 系统资源管理指南

## 概述

为防止系统冻结，我们集成了智能资源监控和动态调配系统。主要功能：

1. **实时资源监控**: 监控 CPU 和内存使用率
2. **动态 Worker 调整**: 根据系统负载自动调整并发数
3. **分批处理**: 避免一次性启动过多进程
4. **自动降级**: 检测到资源过载时自动暂停/终止
5. **渐进式冷却**: 批次间自动休息让系统恢复

## 核心特性

### 1. 资源阈值保护

```bash
--cpu-threshold 80.0      # CPU 使用率 > 80% 触发警告
--memory-threshold 80.0   # 内存使用率 > 80% 触发警告
```

**默认值**: 80% (保守设置，确保系统响应)

### 2. 动态 Worker 数调整

系统会在启动时评估当前负载，自动调整 worker 数量：

```
请求 18 workers → 系统负载 60% → 批准 18 workers
请求 18 workers → 系统负载 85% → 降级到 12 workers
请求 18 workers → 系统负载 95% → 降级到 6 workers
```

### 3. 分批处理模式

```bash
--batch-size 20   # 每批处理 20 个配置，完成后休息再继续
```

**优势**:
- 避免一次性创建过多进程导致系统卡死
- 批次间自动冷却，让系统资源恢复
- 出错时只影响当前批次，已完成批次的结果已保存

**推荐配置**:
- 小规模测试 (< 50 configs): `--batch-size 20`
- 中等规模 (50-200 configs): `--batch-size 30`
- 大规模测试 (> 200 configs): `--batch-size 50`

### 4. 实时监控与反馈

测试运行时会显示：

```
[42/200] ✓ MHPCA_Kq256_Kdq128_Ktau128_r400-50-50_none
    Progress: 42/200 (21.0%) | Rate: 2.15 cfg/s | ETA: 12.3 min
    System: CPU 72.3%, Memory 65.8%

[Resource Monitor] ⚠ WARNING: CPU usage too high: 85.2% > 80.0%
[Resource Monitor] Consider reducing --num-workers or using --batch-size
```

### 5. 自动终止机制

当系统资源持续过载超过 2 分钟，测试会**自动终止**并保存已完成的结果：

```
[Resource Monitor] System overloaded: Memory usage too high: 89.3% > 80.0%
[Resource Monitor] Waiting for resources...
[Resource Monitor] Timeout waiting for resources. Aborting.
Status: ⚠ ABORTED - Partial results saved
```

## 使用建议

### 首次测试 (推荐)

使用保守配置，确保系统稳定：

```bash
./run_sweep_small.sh
# 使用: 12 workers, batch-size 20, 阈值 80%
```

### 系统负载低时

可以适当放宽限制：

```bash
python3 sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results \
  --Kq 256 384 512 \
  --Kdq 128 192 \
  --Ktau 128 192 \
  --latent-triplets "400,50,50" "360,70,70" "320,90,90" \
  --filters none \
  --num-workers 18 \
  --cpu-threshold 85.0 \
  --memory-threshold 85.0 \
  --batch-size 40
```

### 系统负载高时

使用更保守的配置：

```bash
python3 sweep_multigroup_pca.py \
  --data ../Compress_Test/test_complete.npz \
  --out ./results \
  --Kq 256 384 \
  --Kdq 128 \
  --Ktau 128 \
  --latent-triplets "400,50,50" "360,70,70" \
  --filters none \
  --num-workers 8 \
  --cpu-threshold 70.0 \
  --memory-threshold 70.0 \
  --batch-size 10
```

### 在后台运行长期任务

```bash
nohup ./run_sweep_full.sh > sweep_full.log 2>&1 &
# 使用 htop 或 top 实时监控系统状态
```

## 阈值设置指南

| 系统状态 | CPU阈值 | 内存阈值 | Worker数 | Batch大小 |
|---------|---------|---------|---------|----------|
| 空闲 (<30% 负载) | 90% | 90% | 18 | 50 |
| 正常 (30-60% 负载) | 85% | 85% | 16 | 40 |
| 繁忙 (60-80% 负载) | 80% | 80% | 12 | 30 |
| 高负载 (>80% 负载) | 75% | 75% | 8 | 20 |

## 测试资源监控功能

运行快速测试验证资源管理：

```bash
./test_resource_monitor.sh
```

这会运行 10 个配置，测试：
- 资源监控是否正常工作
- 批处理是否正确执行
- 警告和终止机制是否触发

## 手动监控

在另一个终端运行：

```bash
# 实时监控 CPU/内存
watch -n 1 'echo "=== CPU ===" && mpstat 1 1 && echo "=== Memory ===" && free -h'

# 或使用 htop (更直观)
htop
```

## 故障排查

### 问题: 系统仍然卡顿

**解决方案**:
1. 降低 worker 数: `--num-workers 6`
2. 减小批处理: `--batch-size 10`
3. 提高阈值敏感度: `--cpu-threshold 70.0`
4. 限制数据量: `--max-samples 500`

### 问题: 测试频繁暂停

**原因**: 阈值设置过于严格

**解决方案**:
1. 放宽阈值: `--cpu-threshold 90.0 --memory-threshold 90.0`
2. 关闭其他占用资源的程序
3. 在系统负载低时运行测试

### 问题: 测试速度太慢

**解决方案**:
1. 提高 worker 数: `--num-workers 20`
2. 增大批处理: `--batch-size 0` (禁用分批，一次性处理所有)
3. 但要确保系统有足够资源

## 技术细节

### ResourceMonitor 类

核心监控逻辑位于 `resource_monitor.py`:

```python
monitor = ResourceMonitor(
    cpu_threshold=80.0,      # CPU 阈值
    memory_threshold=80.0,   # 内存阈值
    check_interval=1.0       # 检查间隔(秒)
)

# 动态调整 worker 数
safe_workers = monitor.get_safe_worker_count(requested=18)

# 检查资源是否安全
safe, msg = monitor.check_resources()

# 等待资源恢复
success = monitor.wait_for_resources(timeout=120)
```

### 检查频率

- 批次开始前: 完整检查
- 任务进行中: 每 3 个配置检查一次
- 进度显示: 每 5 个配置显示系统状态

### 冷却策略

批次间休息 2 秒，让系统：
- 释放已完成进程的资源
- 刷新文件系统缓存
- 降低 CPU 温度
- 回收内存碎片

## 最佳实践

1. **首次运行**: 使用 `test_resource_monitor.sh` 验证功能
2. **小批量测试**: 先用 `run_sweep_small.sh` 测试
3. **监控系统**: 使用 `htop` 或 `watch` 实时监控
4. **渐进扩大**: 确认稳定后再运行大规模测试
5. **后台运行**: 长期任务使用 `nohup` 或 `screen`/`tmux`
6. **定期检查**: 查看日志确认没有频繁警告

## 性能预期

| Worker数 | 批处理 | 配置/秒 | 100配置耗时 | 系统影响 |
|---------|-------|---------|-----------|---------|
| 6 | 10 | 1.2 | 83秒 | 低 |
| 12 | 20 | 2.3 | 43秒 | 中 |
| 18 | 30 | 3.2 | 31秒 | 高 |

**注**: 实际性能取决于：
- 数据集大小
- K 和 r 参数
- CPU 型号
- 内存速度
- 其他系统负载

## 总结

通过智能资源管理，我们实现了：

✅ **系统不再冻结**: 自动检测和响应过载  
✅ **稳定可靠**: 批处理确保进度可恢复  
✅ **灵活配置**: 根据系统状态调整参数  
✅ **实时反馈**: 清晰的进度和资源状态  
✅ **安全终止**: 资源耗尽时保护系统  

建议从保守配置开始，逐步优化到适合你系统的最佳参数。

