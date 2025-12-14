# 相位单位修复说明

## 🐛 问题描述

**发现的 Bug**: 代码中混用了"时间"和"相位"的概念，导致单位不一致：
- `TH` 变量被当作**相位**（弧度）使用
- `theta0` 和 IK 着陆阶段却按**时间** `t` (使用 `omega*θ`) 计算
- 导致约束 `TH[-1] - TH[0] = 1.2*2π` (弧度) 与 `TH[0] = theta0` (秒?) 单位冲突

## ✅ 解决方案

统一使用**纯相位** θ（弧度），彻底移除时间相关的 `omega` 乘法。

## 🔧 具体修改

### 1. `nearest_phase()` 函数 (第 84-93 行)

#### 修改前（错误）
```python
def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
    thetas = np.linspace(0.0, 2.0*np.pi/omega, K, endpoint=False)  # ❌ 时间范围
    y  = cy + Ay*np.sin(omega*thetas)                              # ❌ omega*t
    z  = cz + Az*np.sin(2.0*omega*thetas + phi)                    # ❌ omega*t
    i  = np.argmin((y - y0)**2 + (z - z0)**2)
    return float(thetas[i])  # 返回时间 t (秒)
```

#### 修改后（正确）
```python
def nearest_phase(y0, z0, cy, cz, Ay, Az, phi, K=1200):
    """
    搜索相位 θ ∈ [0, 2π) 使 (y(θ), z(θ)) 离 (y0, z0) 最近
    返回: 相位 θ (弧度)
    """
    thetas = np.linspace(0.0, 2.0*np.pi, K, endpoint=False)  # ✅ 相位范围 [0,2π)
    y  = cy + Ay*np.sin(thetas)                              # ✅ 纯相位
    z  = cz + Az*np.sin(2.0*thetas + phi)                    # ✅ 纯相位
    i  = np.argmin((y - y0)**2 + (z - z0)**2)
    return float(thetas[i])  # 返回相位 θ (弧度)
```

**关键变化**:
- 移除参数 `omega`
- 搜索范围改为 `[0, 2π)` (相位)
- 直接使用 `thetas` 而非 `omega*thetas`

### 2. IK 着陆阶段 (第 229-243 行)

#### 修改前（错误）
```python
theta0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, omega, phi)  # ❌ 传入 omega
y0 = cy + Ay*np.sin(omega*theta0_guess)                                # ❌ omega*θ
z0 = cz + Az*np.sin(2*omega*theta0_guess + phi)                        # ❌ omega*θ
vy0= Ay*omega*np.cos(omega*theta0_guess)                               # ❌ 混乱
vz0= 2*Az*omega*np.cos(2*omega*theta0_guess + phi)                     # ❌ 混乱
```

#### 修改后（正确）
```python
# 找到最近的相位 θ（弧度）
theta0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, phi)  # ✅ 不传 omega

# 构建着陆目标（纯相位）
y0 = float(cy + Ay*np.sin(theta0_guess))                         # ✅ 纯相位
z0 = float(cz + Az*np.sin(2*theta0_guess + phi))                 # ✅ 纯相位

# 期望速度：用平均相位速度
theta_dot_avg = (laps * 2.0*np.pi) / T_total  # rad/s
vy0 = Ay * np.cos(theta0_guess) * theta_dot_avg
vz0 = 2 * Az * np.cos(2*theta0_guess + phi) * theta_dot_avg
```

**关键变化**:
- `theta0_guess` 现在是纯相位（弧度）
- 位置计算直接用 `sin(θ)` 而非 `sin(ω·t)`
- 速度用链式法则：`dy/dt = (dy/dθ) · (dθ/dt)`
  - `dy/dθ = Ay·cos(θ)`
  - `dθ/dt = θ̇_avg = (laps·2π) / T_total`

### 3. 着陆后相位估计 (第 263 行)

#### 修改前（错误）
```python
theta0 = nearest_phase(p0_land[1], p0_land[2], cy, cz, Ay, Az, omega, phi)  # ❌
```

#### 修改后（正确）
```python
theta0 = nearest_phase(p0_land[1], p0_land[2], cy, cz, Ay, Az, phi)  # ✅
print(f"[init] landed. |p-x*|~{...}, theta0={theta0:.3f} rad")  # ✅ 明确单位
```

### 4. Argparse 描述 (第 185-189 行)

#### 修改前
```python
parser = argparse.ArgumentParser(description="Panda FR3 OCP (CasADi) for 3-lap figure-8")  # ❌ 旧描述
parser.add_argument("--N", type=int, default=300, help="discretization nodes")  # ❌ 不清晰
```

#### 修改后
```python
parser = argparse.ArgumentParser(description="Panda FR3 OCP (CasADi+Spline) for 1.2-lap figure-8")  # ✅ 正确圈数
parser.add_argument("--N", type=int, default=300, help="IK initial guess discretization nodes")  # ✅ 明确用途
```

## 📊 单位一致性验证

### 修复前（错误）
```
nearest_phase 返回: t (秒)
theta0_guess = t (秒)
y0 = cy + Ay*sin(ω·t)  # ω·t 是相位（弧度）
TH[0] = t (秒？)       # ❌ 单位混乱
TH[-1] - TH[0] = 1.2*2π (弧度)  # ❌ 不一致！
```

### 修复后（正确）
```
nearest_phase 返回: θ (弧度)
theta0_guess = θ (弧度)
y0 = cy + Ay*sin(θ)    # θ 是相位（弧度）
TH[0] = θ₀ (弧度)      # ✅ 单位统一
TH[-1] - TH[0] = 1.2*2π (弧度)  # ✅ 一致！
```

## 🔍 物理意义

### 相位 θ vs 时间 t

**相位** θ (弧度):
- 描述在周期运动中的"位置"
- 范围: [0, 2π) 对应一个完整周期
- Figure-8: y(θ) = cy + Ay·sin(θ), z(θ) = cz + Az·sin(2θ+φ)

**时间** t (秒):
- 实际经过的物理时间
- 与相位的关系: θ(t) = ∫θ̇(t)dt
- 对于均匀运动: θ = ω·t (ω 是角速度)

### 为什么要用相位参数化？

1. **几何约束自然**: 直接在曲线上定义约束
2. **单调性简单**: θ 单调递增即可
3. **终点明确**: θ_N = θ_0 + laps·2π
4. **解耦速度**: 相位增量可以非均匀（加速/减速）

## 🧪 验证方法

```python
import numpy as np

# 加载结果
data = np.load('eight_ocp_dataset.npz')
theta_knots = data['knots_theta']

# 检查相位范围（应该在合理的弧度范围内）
print(f"θ₀ = {theta_knots[0]:.3f} rad")
print(f"θₙ = {theta_knots[-1]:.3f} rad")
print(f"Δθ = {theta_knots[-1] - theta_knots[0]:.3f} rad")
print(f"Expected Δθ = {1.2 * 2 * np.pi:.3f} rad")

# 应该看到：
# θ₀ ≈ 0-6 rad (取决于着陆位置)
# Δθ ≈ 7.540 rad (= 1.2 × 2π)
# θ 单调递增
```

## 📝 总结

### 修复前的问题
- ❌ 单位混乱（时间 vs 相位）
- ❌ `nearest_phase` 返回时间但用作相位
- ❌ 约束不一致
- ❌ 物理意义不明确

### 修复后的优势
- ✅ 单位统一（全部使用弧度）
- ✅ 物理意义清晰
- ✅ 约束一致
- ✅ 易于理解和调试

### 核心原则
**使用相位 θ（弧度）进行轨迹参数化，时间 t 仅用于计算采样率和速度推导。**

## 🎯 后续使用建议

1. **检查相位范围**: 确保 θ ∈ [θ₀, θ₀+1.2·2π]
2. **验证单调性**: θₖ₊₁ > θₖ (所有结点)
3. **计算时间轴**: t = (θ-θ₀) / θ̇_avg (近似)
4. **相位速度**: θ̇(t) = (laps·2π) / T_total (平均)

---

**修复日期**: 2025-01-11  
**影响**: 核心算法正确性  
**重要性**: ⭐⭐⭐⭐⭐ (关键修复)





