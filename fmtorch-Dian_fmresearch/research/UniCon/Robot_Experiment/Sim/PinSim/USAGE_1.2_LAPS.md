# 使用指南：1.2 圈配置 (128 结点)

## 📋 配置概览

| 参数 | 值 | 说明 |
|------|-----|------|
| **轨迹时长** | 13.612 秒 | 固定 (6806 × 0.002s) |
| **圈数** | 1.2 圈 | figure-8 路径 |
| **结点数 K** | 128 | 优化变量 |
| **决策变量** | 2688 个 | 7×128×3 (q,dq,ddq) |
| **输出样本** | 6806 个 | 固定样本数 |
| **采样率** | 500 Hz | dt = 0.002s |
| **样条阶数** | 5 阶 | 五次 Hermite (C²) |

## 🚀 基本使用

### 快速运行
```bash
# 使用默认配置（推荐）
python pin_fr3_draw_eight_ocp.py
```

### 自定义参数
```bash
# 自定义输出文件和可视化速度
python pin_fr3_draw_eight_ocp.py --save my_trajectory.npz --speed 2.5

# 使用不同的随机种子
python pin_fr3_draw_eight_ocp.py --seed 42

# 自定义 IK 初始猜测的节点数（不影响优化）
python pin_fr3_draw_eight_ocp.py --N 300
```

## 📊 参数详解

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--speed` | 2.5 | MeshCat 可视化慢放倍数 |
| `--N` | 300 | IK 初始猜测的节点数（不影响优化）|
| `--seed` | 0 | 随机种子 |
| `--save` | `eight_ocp_dataset.npz` | 输出文件名 |

### 固定参数（在代码中）

#### 轨迹参数
```python
# 第 217-218 行
laps = 1.2       # 圈数
T_total = 13.612 # 总时长 (s)
```

#### 结点数
```python
# 第 263 行
K = 128  # 样条结点数
```

#### 配点数
```python
# 第 308 行
colloc_s = [0.211324865405187, 0.788675134594813]  # 2 点 Gauss
```

#### 权重
```python
# 第 302-305 行
w_e, w_x, w_tau = 1e5, 5e3, 5.0  # 路径、平面、扭矩
w_dv, w_da = 1.0, 5.0             # 结点平滑
w_dth = 1.0                       # 相位均匀
```

## 📈 性能特征

### NLP 问题规模
- **决策变量**: 2688 个 (7×128×3 + 128)
- **等式约束**: ~1000 个 (动力学 + 边界)
- **不等式约束**: ~1800 个 (扭矩限制 + 单调性)
- **总约束**: ~2800 个

### 计算时间估计
| 阶段 | 时间 (估计) |
|------|------------|
| IK 初始化 | ~5 秒 |
| NLP 构建 | ~3 秒 |
| IPOPT 求解 | ~3-6 分钟 |
| 样条插值 | ~15 秒 |
| 扭矩计算 | ~15 秒 |
| **总计** | **~4-7 分钟** |

### 内存使用
- **峰值内存**: ~300 MB
- **输出文件**: ~11 MB (npz 格式)

## 📁 输出数据

### NPZ 文件结构
```python
data = np.load('eight_ocp_dataset.npz')

# 致密轨迹 (6806 × 维度)
t = data['t']           # 时间 (6806,)
q = data['q']           # 关节位置 (6806, 7)
dq = data['dq']         # 关节速度 (6806, 7)
ddq = data['ddq']       # 关节加速度 (6806, 7)
tau = data['tau']       # 关节力矩 (6806, 7)
ee = data['ee']         # 末端位置 (6806, 3)
ee_des = data['ee_des'] # 期望末端位置 (6806, 3)
theta = data['theta']   # 相位参数 (6806,)

# 结点数据 (128 × 维度)
knots_q = data['knots_q']       # (128, 7)
knots_dq = data['knots_dq']     # (128, 7)
knots_ddq = data['knots_ddq']   # (128, 7)
knots_theta = data['knots_theta'] # (128,)

# 元数据
sample_rate = data['sample_rate']  # 500.0 Hz
x_plane = data['x_plane']          # X 平面位置
fig8_params = data['fig8_params']  # [cy, cz, Ay, Az, phi]
```

### 验证输出
```python
import numpy as np

# 加载数据
data = np.load('eight_ocp_dataset.npz')

# 验证采样率
dt_actual = data['t'][1] - data['t'][0]
fs_actual = 1.0 / dt_actual
print(f"Sampling rate: {fs_actual:.2f} Hz")  # 应该是 500.00 Hz

# 验证时长
T_actual = data['t'][-1] - data['t'][0]
print(f"Duration: {T_actual:.3f} s")  # 应该是 13.612 s

# 验证样本数
print(f"Samples: {len(data['t'])}")  # 应该是 6806

# 验证圈数
theta_range = data['knots_theta'][-1] - data['knots_theta'][0]
laps_actual = theta_range / (2 * np.pi)
print(f"Laps: {laps_actual:.2f}")  # 应该是 1.20
```

## 🎨 可视化

### MeshCat 可视化
运行脚本后会自动打开浏览器显示 3D 可视化：
```
[viz] MeshCat ON
URL: http://127.0.0.1:7000/static/
```

### 生成图表
脚本会自动生成三个图表：
1. **YZ 平面轨迹** - 实际 vs 期望
2. **关节力矩曲线** - 7 个关节
3. **跟踪误差** - 末端位置误差随时间变化

## 🔧 调整参数

### 增加精度（更多结点）
```python
# 修改第 263 行
K = 256  # 从 128 增加到 256
# 决策变量: 2688 → 5376
# 求解时间: 4-7分钟 → 8-15分钟
```

### 减少计算时间（更少结点）
```python
# 修改第 263 行
K = 64  # 从 128 减少到 64
# 决策变量: 2688 → 1344
# 求解时间: 4-7分钟 → 2-4分钟
# 注意: 精度会略微下降
```

### 调整相位均匀性
```python
# 修改第 304 行
w_dth = 0.5   # 更宽松（允许更多变速）
w_dth = 1.0   # 默认（平衡）
w_dth = 2.0   # 更严格（更接近匀速）
```

### 增加配点精度
```python
# 修改第 308 行
# 从 2 点升级到 3 点 Gauss-Legendre
colloc_s = [0.112701665379258, 0.5, 0.887298334620742]
# 约束数量: ~2800 → ~4200
# 求解时间: +20-30%
# 动力学精度: 更高
```

## 📝 常见问题

### Q1: 为什么输出是 6806 样本而不是其他数字？
**A**: 这是为了与其他版本对齐。6806 × 0.002s = 13.612s，正好是 500Hz 采样。

### Q2: 可以改变圈数吗？
**A**: 可以，修改第 217 行的 `laps` 和第 218 行的 `T_total`。
```python
laps = 2.0       # 改为 2 圈
T_total = 20.0   # 相应调整时长
```

### Q3: K=128 是否过多？
**A**: 对于 1.2 圈 13.6 秒的轨迹，128 个结点是合理的（每段约 0.107 秒）。如果觉得求解慢，可以降到 K=64。

### Q4: IPOPT 不收敛怎么办？
**A**: 尝试以下方法：
1. 增加迭代次数：修改第 350 行 `ipopt.max_iter: 2000` → `3000`
2. 放松容差：修改第 350 行 `ipopt.tol: 1e-4` → `1e-3`
3. 减少结点数：K=128 → K=64

### Q5: 如何获得更平滑的扭矩？
**A**: 增加平滑权重：
```python
w_dv = 2.0   # 从 1.0 增加到 2.0（速度平滑）
w_da = 10.0  # 从 5.0 增加到 10.0（加速度平滑）
```

## 🎯 推荐工作流

### 1. 快速测试（K=64）
```bash
python pin_fr3_draw_eight_ocp.py --save test_k64.npz
# 修改代码: K = 64
# 时间: ~2-4 分钟
```

### 2. 标准运行（K=128，默认）
```bash
python pin_fr3_draw_eight_ocp.py --save standard_k128.npz
# 时间: ~4-7 分钟
```

### 3. 高精度（K=256）
```bash
python pin_fr3_draw_eight_ocp.py --save highres_k256.npz
# 修改代码: K = 256
# 时间: ~8-15 分钟
```

## 📊 与其他配置对比

| 配置 | 圈数 | 时长(s) | K | 变量数 | 求解时间 |
|------|------|---------|---|--------|----------|
| 旧版 (3圈) | 3.0 | 31.42 | 64 | 1344 | 2-5分钟 |
| **当前 (1.2圈)** | **1.2** | **13.612** | **128** | **2688** | **4-7分钟** |
| 快速版 | 1.2 | 13.612 | 64 | 1344 | 2-4分钟 |
| 高精度版 | 1.2 | 13.612 | 256 | 5376 | 8-15分钟 |

## 🔬 技术细节

### 五次 Hermite 样条
- **连续性**: C² (位置、速度、加速度连续)
- **段数**: K-1 = 127 段
- **每段长度**: h = 13.612/127 ≈ 0.107 秒
- **配点/段**: 2 个 (Gauss-Legendre)
- **总配点**: 127 × 2 = 254 个

### 动力学约束
在每个配点处强制：
```
τ(s) = M(q(s)) · ẍ(s) + h(q(s), ẋ(s))
-τ_max ≤ τ(s) ≤ τ_max
```

### 相位约束
```
θ₀ = nearest_phase(初始位置)
θₙ = θ₀ + 1.2 × 2π
θₖ₊₁ ≥ θₖ (单调)
```

## 🎓 参考

- **Hermite 插值理论**: Numerical Recipes, Chapter 3
- **直接配点法**: Optimal Control Theory by Donald E. Kirk
- **Gauss-Legendre 配点**: https://en.wikipedia.org/wiki/Gaussian_quadrature

## 📞 支持

如有问题，请检查：
1. CasADi 和 Pinocchio 版本是否正确
2. URDF 文件路径是否正确
3. IPOPT 求解器是否安装
4. 内存是否充足（至少 4GB 可用）

---

**版本**: v1.2 (2025-01-11)  
**配置**: 1.2 圈，128 结点，500Hz 采样

