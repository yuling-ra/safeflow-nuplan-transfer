# Panda FR3 OCP with Quintic Hermite Spline 使用说明

## 概述

`pin_fr3_draw_eight_ocp_spline.py` 是升级版的最优控制程序，使用**五次 Hermite 样条参数化**来优化 Panda 机器人的"8"字轨迹跟踪。

### 主要改进

1. **决策变量大幅减少**：从原来的 6000+ 个节点降到 64 个结点（knots）
   - 变量数：7×64×3 ≈ 1344 个（q, dq, ddq 在每个结点）
   - 原方法：7×300×5 = 10500 个变量

2. **C² 连续性**：五次 Hermite 样条确保位置、速度、加速度在整条轨迹上连续

3. **配点法约束**：在每段样条的 Gauss 配点上强制动力学和路径误差

4. **消除扭矩变量**：通过表达式 `tau = M(q)a + h(q,v)` 直接计算，减少变量

5. **500Hz 输出**：生成致密轨迹（2ms 间隔），与 PD 控制版本对齐

## 基本用法

```bash
python pin_fr3_draw_eight_ocp_spline.py
```

## 可选参数

```bash
python pin_fr3_draw_eight_ocp_spline.py --K 64 --speed 2.5 --seed 0 --save my_trajectory.npz
```

### 参数说明

- `--K`: 样条结点数量（默认 64）
  - 更多结点 = 更高精度，但优化更慢
  - 推荐范围：32-128

- `--speed`: MeshCat 可视化慢放倍数（默认 2.5）

- `--seed`: 随机种子（默认 0）

- `--save`: 输出数据文件名（默认 "eight_ocp_spline_dataset.npz"）

## 输出数据格式

生成的 `.npz` 文件包含：

### 致密轨迹（500Hz）
- `t`: 时间数组
- `q`: 关节位置 (N × 7)
- `dq`: 关节速度 (N × 7)
- `ddq`: 关节加速度 (N × 7)
- `tau`: 关节力矩 (N × 7)
- `ee`: 末端实际位置 (N × 3)
- `ee_des`: 末端期望位置 (N × 3)
- `theta`: 相位参数 (N,)

### 结点数据（优化结果）
- `knots_t`: 结点时间 (K,)
- `knots_q`: 结点关节位置 (K × 7)
- `knots_dq`: 结点关节速度 (K × 7)
- `knots_ddq`: 结点关节加速度 (K × 7)
- `knots_theta`: 结点相位 (K,)

### 参数
- `x_plane`, `fig8_params`: "8"字轨迹参数

## 与原版本对比

| 特性 | 原版 (Euler) | 样条版 (Hermite) |
|------|-------------|-----------------|
| 决策变量 | ~10,500 | ~1,344 |
| 连续性 | C⁰ | C² |
| 求解时间 | ~5-10 分钟 | ~2-5 分钟 |
| 输出采样率 | 自定义 | 500Hz 固定 |
| 动力学约束 | 每个节点 | 配点处 |

## 调参建议

### 结点数 `--K`
- **32**: 快速原型，精度较低
- **64** (默认): 平衡精度和速度
- **128**: 高精度，求解较慢

### 权重调整（在代码中）
```python
w_e     = 1e5     # 路径法向误差权重
w_x     = 5e3     # X 平面软约束
w_tau   = 5.0     # 扭矩 L2 范数
w_dv    = 1.0     # 结点速度平滑
w_da    = 5.0     # 结点加速度平滑
```

### 配点数（在代码中）
```python
# 当前使用 2 点 Gauss 配点
colloc_s = [0.211324865405187, 0.788675134594813]

# 如需更高精度，可升级到 3 点：
# colloc_s = [0.112701665379258, 0.5, 0.887298334620742]
```

## 依赖环境

与原版相同：
```bash
conda activate casadi_env  # 或你的 CasADi 环境
# 需要: pinocchio[meshcat]>=2.7.0, casadi, meshcat, numpy, matplotlib
```

## 故障排除

### IPOPT 不收敛
- 尝试增加迭代次数：修改 `ipopt.max_iter` (默认 3000)
- 减少结点数：`--K 32`
- 调整初始猜测：检查 IK landing 是否成功

### 轨迹有振荡
- 增加平滑权重：`w_dv`, `w_da`
- 增加配点数量（代码内修改）

### 内存不足
- 减少结点数：`--K 32` 或 `--K 48`

## 示例

生成高精度轨迹：
```bash
python pin_fr3_draw_eight_ocp_spline.py --K 128 --save high_precision.npz
```

快速测试：
```bash
python pin_fr3_draw_eight_ocp_spline.py --K 32 --speed 1.0
```

## 技术细节

### 五次 Hermite 样条插值

对于段 `[t_k, t_{k+1}]`，参数 `s ∈ [0,1]`：

```
q(s) = H₀₀(s)·q_k + H₁₀(s)·h·v_k + H₂₀(s)·h²·a_k
     + H₀₁(s)·q_{k+1} + H₁₁(s)·h·v_{k+1} + H₂₁(s)·h²·a_{k+1}
```

其中 `H_ij(s)` 是五次 Hermite 基函数，保证：
- q(0)=q_k, q(1)=q_{k+1}
- v(0)=v_k, v(1)=v_{k+1}  
- a(0)=a_k, a(1)=a_{k+1}

### 配点法

在每段的 Gauss 配点 `s ∈ {0.21, 0.79}` 处强制：
- 动力学约束：`τ = M(q)a + h(q,v)`
- 扭矩限制：`|τ| ≤ τ_max`
- 路径跟踪代价

这比在结点处约束更准确，且不增加变量数。

## 参考

- 原版 Euler 方法：`pin_fr3_draw_eight_ocp.py`
- PD 控制版本：`pin_fr3_draw_eight.py`

