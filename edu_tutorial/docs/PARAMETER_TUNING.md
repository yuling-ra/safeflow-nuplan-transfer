# 🎛️ Flow Matching + CBF 参数调优完全指南

## 目录
1. [参数总览](#参数总览)
2. [时间门控参数](#时间门控参数)
3. [CBF强度参数](#cbf强度参数)
4. [ODE求解器参数](#ode求解器参数)
5. [常见场景调参](#常见场景调参)
6. [高级调参技巧](#高级调参技巧)

---

## 参数总览

### 📊 参数重要性排序

| 优先级 | 参数 | 默认值 | 范围 | 影响 |
|--------|------|--------|------|------|
| ⭐⭐⭐ | SUPPRESS_FRAC | 0.50 | [0, 1] | CBF激活时刻 |
| ⭐⭐⭐ | MASK_GATE | 1.0 | [0, 1] | CBF全局强度 |
| ⭐⭐ | ODE_CBF_PASSES | 5 | [1, 20] | 约束严格程度 |
| ⭐⭐ | RAMP_FRAC | 0.80 | [0, 1] | CBF完全激活时刻 |
| ⭐ | MAX_CORR_NORM | 1.9 | [0.5, 5] | 单次修正限制 |
| ⭐ | ODE_CBF_MARGIN | 0.0 | [0, 0.1] | 安全边界 |

---

## 时间门控参数

### 🕐 SUPPRESS_FRAC - CBF开始时刻

**定义**: 在这个时间比例之前，CBF完全关闭（g(t)=0）

**数学**: 
$$f(t) = 0, \quad t < \text{SUPPRESS\_FRAC} \times T$$

**默认值**: `0.50` (前50%时间)

**调参指南**:

| 值 | 效果 | 适用场景 |
|---|---|---|
| 0.2-0.3 | 很早激活CBF | 轨迹经常穿过障碍物 |
| 0.4-0.5 | 较早激活（推荐） | 平衡性能和安全 |
| 0.6-0.7 | 较晚激活 | 轨迹过于保守 |
| 0.8-0.9 | 很晚激活 | 只在最后时刻避障 |

**实验示例**:
```python
# 测试不同的SUPPRESS_FRAC
for frac in [0.3, 0.5, 0.7]:
    cfg.SUPPRESS_FRAC = frac
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    # 保存并比较
```

**症状诊断**:
- ❌ 轨迹穿障碍 → 降低至0.3-0.4
- ❌ 轨迹太弯曲 → 提高至0.6-0.7
- ✅ 轨迹自然且安全 → 保持当前值

---

### 🕐 RAMP_FRAC - CBF完全激活时刻

**定义**: 在这个时间比例之后，CBF完全激活（g(t)=1）

**数学**:
$$f(t) = 1, \quad t > \text{RAMP\_FRAC} \times T$$

**默认值**: `0.80` (80%后)

**约束**: 必须 > SUPPRESS_FRAC

**调参指南**:

**过渡平滑度** = RAMP_FRAC - SUPPRESS_FRAC

| 差值 | 过渡类型 | 适用场景 |
|------|---------|----------|
| 0.1 | 快速切换 | 需要明确的阶段划分 |
| 0.2-0.3 | 平滑过渡（推荐） | 自然的轨迹演化 |
| 0.4-0.5 | 缓慢过渡 | 极其平滑的修正 |

**常用组合**:
```python
# 组合1: 早期激活，平滑过渡
cfg.SUPPRESS_FRAC = 0.3
cfg.RAMP_FRAC = 0.6  # 差值0.3

# 组合2: 中期激活，平衡（推荐）
cfg.SUPPRESS_FRAC = 0.5
cfg.RAMP_FRAC = 0.8  # 差值0.3

# 组合3: 晚期激活，快速
cfg.SUPPRESS_FRAC = 0.7
cfg.RAMP_FRAC = 0.85  # 差值0.15
```

---

### 💪 MASK_GATE - CBF全局强度

**定义**: 整体控制CBF的强度系数

**数学**:
$$g(t) = \text{MASK\_GATE} \times f(t)$$

**默认值**: `1.0` (完全启用)

**范围**: [0, 1]

**调参指南**:

| 值 | CBF强度 | 适用场景 |
|---|---------|----------|
| 0.0 | 关闭 | Debug或对比实验 |
| 0.3-0.5 | 弱 | 轻微修正，保持流畅 |
| 0.7-0.8 | 中等 | 平衡安全和自然度 |
| 0.9-1.0 | 强（推荐） | 最大安全保障 |

**实验**:
```python
# 测试不同强度
for gate in [0.5, 0.7, 1.0]:
    cfg.MASK_GATE = gate
    traj = generate_trajectory(flow_model, scaler, cfg, use_cbf=True)
    # 计算违反约束的程度
    violation = sum_violating_h(traj, ELLIPSES)
    print(f"MASK_GATE={gate}: violation={violation:.4f}")
```

---

## CBF强度参数

### 🔁 ODE_CBF_PASSES - 投影迭代次数

**定义**: 每次ODE评估时，CBF投影的迭代次数

**默认值**: `5`

**范围**: [1, 20]

**调参指南**:

| 值 | 约束严格程度 | 计算时间 | 适用场景 |
|---|------------|----------|----------|
| 1-2 | 很松 | 很快 | Debug/快速测试 |
| 3-5 | 中等（推荐） | 快 | 一般使用 |
| 7-10 | 严格 | 中等 | 复杂障碍物 |
| 15-20 | 非常严格 | 慢 | 极端安全要求 |

**经验法则**:
- 障碍物数量 × 2 = 建议passes
- 3个椭圆 → 5-6 passes

**代码示例**:
```python
# 自适应passes
num_obstacles = len(ELLIPSES)
cfg.ODE_CBF_PASSES = max(3, min(num_obstacles * 2, 10))
```

---

### 📏 ODE_CBF_MAX_CORR_NORM - 最大修正范数

**定义**: 限制单次速度修正的最大L2范数

**默认值**: `1.9`

**范围**: [0.5, 5.0]

**作用**: 防止过度修正导致轨迹扭曲

**调参指南**:

| 值 | 修正力度 | 轨迹特征 | 适用场景 |
|---|---------|---------|----------|
| 0.5-1.0 | 保守 | 平滑，可能违规 | 追求自然度 |
| 1.5-2.0 | 平衡（推荐） | 较平滑，安全 | 一般使用 |
| 2.5-3.5 | 激进 | 可能抖动，很安全 | 高安全要求 |
| > 4.0 | 极端 | 明显扭曲 | 不推荐 |

**症状诊断**:
- ❌ 轨迹抖动 → 降低至1.0-1.5
- ❌ 仍然穿障碍 → 提高至2.5-3.0
- ✅ 平滑且安全 → 保持当前值

---

### 🛡️ ODE_CBF_MARGIN - 安全边界

**定义**: 额外的安全距离（单位：米）

**数学**:
$$h_{\text{eff}}(x) = h(x) - \text{MARGIN}$$

**默认值**: `0.0` (精确边界)

**范围**: [0, 0.1]

**调参指南**:

| 值 | 安全距离 | 适用场景 |
|---|---------|----------|
| 0.0 | 无额外距离（推荐） | 精确控制 |
| 0.01-0.02 | 1-2cm | 略微保守 |
| 0.03-0.05 | 3-5cm | 明显保守 |
| > 0.05 | > 5cm | 过于保守 |

**何时增加MARGIN?**
- 模型预测有误差
- 物理系统有延迟
- 需要额外安全裕度

---

## ODE求解器参数

### ⏱️ T_SPAN - 时间离散化

**定义**: 从t=0到t=1的时间点序列

**默认**: `torch.linspace(0., 1., 100)` (100个点)

**调参指南**:

| 点数 | 轨迹平滑度 | 计算时间 | 适用场景 |
|------|-----------|----------|----------|
| 20-50 | 粗糙 | 很快 | 快速测试 |
| 80-120 | 平滑（推荐） | 适中 | 一般使用 |
| 200-500 | 非常平滑 | 慢 | 高质量可视化 |
| > 1000 | 极致平滑 | 很慢 | 特殊需求 |

**代码**:
```python
# 快速测试
cfg.T_SPAN = torch.linspace(0., 1., 50)

# 高质量
cfg.T_SPAN = torch.linspace(0., 1., 200)
```

---

### 🔧 SOLVER_METHOD - 求解器算法

**可选值**:
- `'euler'`: 一阶Euler方法
- `'rk4'`: 4阶Runge-Kutta
- `'dopri5'`: 5阶Dormand-Prince（推荐）
- `'dopri8'`: 8阶（更精确但慢）

**对比**:

| 方法 | 精度 | 速度 | 稳定性 | 推荐度 |
|------|-----|------|--------|--------|
| euler | 低 | 最快 | 差 | ❌ |
| rk4 | 中 | 快 | 好 | ✅ |
| dopri5 | 高 | 适中 | 很好 | ⭐⭐⭐ |
| dopri8 | 很高 | 慢 | 极好 | ⭐ |

**选择建议**:
- 日常使用: `dopri5`
- 快速测试: `rk4`
- 高精度: `dopri8`

---

### 📊 SOLVER_TOLERANCE - 求解精度

**定义**: ODE求解的误差容忍度（atol和rtol）

**默认**: `1e-5`

**调参指南**:

| 值 | 精度 | 速度 | 适用场景 |
|---|-----|------|----------|
| 1e-3 | 低 | 很快 | 快速原型 |
| 1e-4 | 中 | 快 | 一般测试 |
| 1e-5 | 高（推荐） | 适中 | 生产使用 |
| 1e-6, 1e-7 | 很高 | 慢 | 高精度需求 |

**经验**:
- 降低tolerance → 更精确但更慢
- 提高tolerance → 更快但可能不稳定

---

## 常见场景调参

### 场景1: 轨迹穿过障碍物 🚫

**症状**: 生成的轨迹与椭圆相交

**解决方案**（按优先级）:

1. **降低SUPPRESS_FRAC**
```python
cfg.SUPPRESS_FRAC = 0.3  # 从0.5降低
# 让CBF更早激活
```

2. **增加ODE_CBF_PASSES**
```python
cfg.ODE_CBF_PASSES = 10  # 从5增加
# 更严格的约束执行
```

3. **增大ODE_CBF_MARGIN**
```python
cfg.ODE_CBF_MARGIN = 0.02  # 从0.0增加
# 增加安全距离
```

4. **确认MASK_GATE**
```python
cfg.MASK_GATE = 1.0  # 确保完全开启
```

**验证**:
```python
# 生成轨迹后检查
traj_tensor = torch.from_numpy(traj.T).to(cfg.DEVICE)
h_min = min_h_over_all(traj_tensor, ELLIPSES)
print(f"Min h value: {h_min:.4f}")
# 应该 > 0
```

---

### 场景2: 轨迹过于保守 🐌

**症状**: 轨迹过度弯曲，远离障碍物，不自然

**解决方案**:

1. **提高SUPPRESS_FRAC**
```python
cfg.SUPPRESS_FRAC = 0.7  # 从0.5提高
# 延迟CBF激活
```

2. **降低ODE_CBF_PASSES**
```python
cfg.ODE_CBF_PASSES = 3  # 从5降低
# 减轻约束强度
```

3. **降低MAX_CORR_NORM**
```python
cfg.ODE_CBF_MAX_CORR_NORM = 1.0  # 从1.9降低
# 限制修正幅度
```

4. **降低MASK_GATE**
```python
cfg.MASK_GATE = 0.7  # 从1.0降低
# 减弱全局强度
```

---

### 场景3: 计算太慢 🐢

**症状**: 生成一条轨迹耗时过长

**解决方案**:

1. **减少T_SPAN点数**
```python
cfg.T_SPAN = torch.linspace(0., 1., 50)  # 从100降低
```

2. **降低ODE_CBF_PASSES**
```python
cfg.ODE_CBF_PASSES = 3  # 从5降低
```

3. **提高SOLVER_TOLERANCE**
```python
cfg.SOLVER_TOLERANCE = 1e-4  # 从1e-5提高
```

4. **切换求解器**
```python
cfg.SOLVER_METHOD = 'rk4'  # 从dopri5切换
```

5. **确认使用GPU**
```python
print(f"Device: {cfg.DEVICE}")
# 应该显示 cuda
```

---

### 场景4: 轨迹抖动 📈

**症状**: 轨迹有明显折线或不平滑

**解决方案**:

1. **增加T_SPAN点数**
```python
cfg.T_SPAN = torch.linspace(0., 1., 200)  # 从100增加
```

2. **降低SOLVER_TOLERANCE**
```python
cfg.SOLVER_TOLERANCE = 1e-6  # 从1e-5降低
```

3. **降低MAX_CORR_NORM**
```python
cfg.ODE_CBF_MAX_CORR_NORM = 1.0  # 限制修正
```

4. **平滑过渡区间**
```python
cfg.SUPPRESS_FRAC = 0.4
cfg.RAMP_FRAC = 0.9  # 拉大差值
```

---

## 高级调参技巧

### 🎯 技巧1: 二分搜索找最优值

```python
# 找最优SUPPRESS_FRAC
def test_suppress_frac(frac):
    cfg.SUPPRESS_FRAC = frac
    trajs = [generate_trajectory(...) for _ in range(10)]
    # 计算指标: 安全性 + 自然度
    safety = sum(min_h(t) > 0 for t in trajs) / 10
    smoothness = -sum(compute_roughness(t) for t in trajs) / 10
    return safety + smoothness

best_frac = binary_search(test_suppress_frac, 0.2, 0.8)
```

### 🎯 技巧2: 网格搜索

```python
# 2D网格搜索
results = {}
for suppress in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for mask_gate in [0.5, 0.7, 0.9, 1.0]:
        cfg.SUPPRESS_FRAC = suppress
        cfg.MASK_GATE = mask_gate
        trajs = generate_multiple(...)
        results[(suppress, mask_gate)] = evaluate(trajs)

# 找最佳组合
best_params = max(results, key=results.get)
```

### 🎯 技巧3: 自适应passes

```python
# 根据障碍物数量自动调整
cfg.ODE_CBF_PASSES = len(ELLIPSES) * 2
```

### 🎯 技巧4: 分阶段调参

```python
# 阶段1: 先确保安全（不管自然度）
cfg.MASK_GATE = 1.0
cfg.SUPPRESS_FRAC = 0.3
cfg.ODE_CBF_PASSES = 10
# 测试是否完全安全

# 阶段2: 在安全基础上优化自然度
cfg.SUPPRESS_FRAC += 0.1
cfg.ODE_CBF_PASSES -= 2
# 逐步放松约束，直到达到平衡
```

### 🎯 技巧5: 记录实验

```python
# 保存实验记录
experiment_log = {
    'params': vars(cfg),
    'results': {
        'safety_rate': safety_rate,
        'avg_smoothness': smoothness,
        'computation_time': time_cost
    },
    'notes': 'Increased SUPPRESS_FRAC to reduce conservatism'
}

import json
with open('experiments.json', 'a') as f:
    json.dump(experiment_log, f)
    f.write('\n')
```

---

## 参数组合推荐

### 🌟 配置1: 平衡型（推荐）
```python
cfg.MASK_GATE = 1.0
cfg.SUPPRESS_FRAC = 0.50
cfg.RAMP_FRAC = 0.80
cfg.ODE_CBF_PASSES = 5
cfg.ODE_CBF_MAX_CORR_NORM = 1.9
cfg.ODE_CBF_MARGIN = 0.0
cfg.SOLVER_METHOD = 'dopri5'
cfg.SOLVER_TOLERANCE = 1e-5
```

### 🛡️ 配置2: 安全优先
```python
cfg.MASK_GATE = 1.0
cfg.SUPPRESS_FRAC = 0.30  # 更早激活
cfg.RAMP_FRAC = 0.70
cfg.ODE_CBF_PASSES = 10   # 更严格
cfg.ODE_CBF_MAX_CORR_NORM = 2.5
cfg.ODE_CBF_MARGIN = 0.02  # 额外距离
```

### 🎨 配置3: 自然度优先
```python
cfg.MASK_GATE = 0.7       # 降低强度
cfg.SUPPRESS_FRAC = 0.70   # 更晚激活
cfg.RAMP_FRAC = 0.85
cfg.ODE_CBF_PASSES = 3    # 较少迭代
cfg.ODE_CBF_MAX_CORR_NORM = 1.0  # 限制修正
cfg.ODE_CBF_MARGIN = 0.0
```

### ⚡ 配置4: 速度优先
```python
cfg.T_SPAN = torch.linspace(0., 1., 50)  # 减少点数
cfg.ODE_CBF_PASSES = 3
cfg.SOLVER_METHOD = 'rk4'  # 更快的求解器
cfg.SOLVER_TOLERANCE = 1e-4  # 降低精度
```

---

## 总结

### 🎓 调参原则

1. **优先级**: 时间门控 > CBF强度 > ODE参数
2. **渐进式**: 一次只改一个参数
3. **记录**: 保存每次实验的结果
4. **对比**: 用基准配置作为参照
5. **理解**: 知道为什么这样调

### 🎯 快速诊断表

| 问题 | 首选参数 | 调整方向 |
|------|---------|---------|
| 穿障碍 | SUPPRESS_FRAC | ⬇️ 降低 |
| 太保守 | SUPPRESS_FRAC | ⬆️ 提高 |
| 太慢 | T_SPAN, PASSES | ⬇️ 降低 |
| 抖动 | MAX_CORR_NORM, T_SPAN | ⬇️ 或 ⬆️ |

---

最后更新: 2025-01-09
版本: 1.0

