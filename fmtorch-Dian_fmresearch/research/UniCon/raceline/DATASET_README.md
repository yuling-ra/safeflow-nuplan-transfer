# Nürburgring OCP Trajectory Dataset

## 1. 概述

本数据集包含在 Nürburgring 赛道地图上通过最优控制问题 (OCP) 生成的大量车辆轨迹。数据集的宗旨在为基于学习的运动规划和控制策略（如模仿学习、流匹配模型等）提供高质量的专家数据。

- **轨迹总数**: 10,180
- **数据来源**: 由以下四个 `npz` 文件合并而成：
  - `ocp_dataset_incremental.npz` (2010 trajectories)
  - `ocp_dataset_incremental_99743ee3.npz` (2670 trajectories)
  - `ocp_dataset_inverse_2a1131b3.npz` (3500 trajectories)
  - `ocp_dataset_inverse_6dc436f8.npz` (2000 trajectories)
- **物理环境**: 所有轨迹都在一个 `585x514` 像素的 2D 栅格地图上生成。

## 2. 数据生成逻辑

轨迹数据主要通过两种不同的 OCP 求解脚本生成，它们在采样策略和优化目标上略有不同，从而提供了更丰富的数据多样性。

### 2.1 "Incremental" 方法 (`make_dataset.py`)

- **核心逻辑**: 在赛道地图上预定义的起始和结束矩形区域内随机采样起点和终点。
- **参考线**: 连接起点和终点在主参考赛道线上的投影点，并重新采样成固定长度的路径作为 OCP 的参考。
- **优化目标**: 使用一个较为基础的成本函数，主要惩罚偏离参考路径的误差、控制量的大小和平滑度。
- **特点**: 生成的轨迹覆盖范围广，但可能不是最激进或最优的驾驶行为。

### 2.2 "Inverse" 方法 (`run_inverse_batch.py`)

- **核心逻辑**: 采用更复杂的 **分阶段 OCP (Staged-OCP)** 求解策略。
- **参考线**: 与 "Incremental" 类似，但对参考线的处理和初始状态的生成更为精细。
- **优化目标**: 成本函数非常复杂，引入了更多项，例如：
  - **进度项 (Progress Term)**: 鼓励轨迹尽可能沿着参考线前进。
  - **目标吸引 (Goal Attraction)**: 在轨迹的末端施加一个强大的吸引力，确保能精确到达目标点。
  - **高级权重调度**: 在不同的优化阶段，动态调整各项成本（如安全性、路径跟踪、平滑度）的权重。
  - **Warm Start**: 使用 Pure Pursuit 和曲率前馈等方法生成一个高质量的初始控制序列，大大提高了求解效率和成功率。
- **特点**: 生成的轨迹更接近专家级的驾驶行为，动态性能和控制平滑性更好。这也解释了为什么验证脚本会提示 `inverse` 数据集的控制量标准差（`accel` std）显著高于 `incremental` 数据集，因为它探索了更极限的控制输入。

## 3. 数据结构 (DataLoader 设计参考)

数据集以单个 `.npz` 格式文件存储，可以通过 `numpy.load()` 读取。内部包含多个键 (key)，每个键对应一个 `numpy` 数组。所有轨迹的长度都已统一。

- **`N`**: 表示轨迹总数 (10,180)。
- **`T_states`**: 状态序列的长度 (101)。
- **`T_actions`**: 控制序列的长度 (100)。

---

**`states`**
- **形状**: `(N, T_states, 4)` 即 `(10180, 101, 4)`
- **数据类型**: `float64`
- **维度说明**:
  - `dim 0`: 轨迹索引
  - `dim 1`: 时间步
  - `dim 2`: 车辆状态 `[x, y, theta, velocity]`
    - `x` (pix): 车辆质心 x 坐标
    - `y` (pix): 车辆质心 y 坐标
    - `theta` (rad): 航向角
    - `velocity` (pix/s): 速度
- **DataLoader 用途**: 这是模型需要学习和预测的核心数据。对于流匹配模型，这是 `y` (目标)；对于自回归模型，这是序列输入和输出。

---

**`actions`**
- **形状**: `(N, T_actions, 2)` 即 `(10180, 100, 2)`
- **数据类型**: `float64`
- **维度说明**:
  - `dim 0`: 轨迹索引
  - `dim 1`: 时间步 (注意: 比状态少一维)
  - `dim 2`: 控制输入 `[acceleration, steering_angle]`
    - `acceleration` (pix/s²): 加速度
    - `steering_angle` (rad): 转向角
- **DataLoader 用途**: 可以作为模型的条件输入，或在某些任务中作为预测目标。

---

**`x0s`**
- **形状**: `(N, 4)` 即 `(10180, 4)`
- **数据类型**: `float64`
- **维度说明**: 每条轨迹的初始状态 `[x, y, theta, velocity]`，等同于 `states[:, 0, :]`。
- **DataLoader 用途**: 作为模型的初始条件输入，例如在 ODE/SDE 求解器中设置 `x(t_0)`。

---

**`starts` / `ends`**
- **形状**: `(N, 2)` 即 `(10180, 2)`
- **数据类型**: `float64`
- **维度说明**: 每条轨迹在 OCP 求解时设定的起点和终点 `[x, y]`。
- **DataLoader 用途**: 可用作高级条件输入，例如训练一个目标导向的策略模型 `pi(a | s, goal)`。

---

**`refs`**
- **形状**: `(N, T_states, 2)` 即 `(10180, 101, 2)`
- **数据类型**: `float64`
- **维度说明**: OCP 求解时使用的参考路径 `[x, y]`。
- **DataLoader 用途**: 可以作为模型的强引导条件，帮助模型学习遵循特定路径。

---

**`vmaxs`**
- **形状**: `(N,)` 即 `(10180,)`
- **数据类型**: `float64`
- **维度说明**: 每条轨迹段对应的最大速度限制 (pix/s)。
- **DataLoader 用途**: 可作为条件输入，帮助模型理解不同路段的动态约束。

## 4. 数据统计摘要

以下是整个合并后数据集的关键物理量的统计信息：

| 物理量 | 均值 (Mean) | 标准差 (Std Dev) | 范围 (Range) |
| :--- | :---: | :---: | :---: |
| **加速度** | 2.239 | 12.052 | `[-27.1, 34.3]` |
| **转向角** | -0.001 | 0.060 | `[-0.51, 0.61]` |
| **速度** | 49.239 | 12.566 | `[0.0, 75.6]` |

## 5. 使用示例

```python
import numpy as np

# 加载数据集
dataset_path = 'path/to/your/combined_dataset.npz'
data = np.load(dataset_path)

# 访问数据
states = data['states']
actions = data['actions']
initial_states = data['x0s']

print(f"数据集总轨迹数: {states.shape[0]}")
print(f"单条轨迹状态维度: {states.shape[1:]}")
print(f"单条轨迹动作维度: {actions.shape[1:]}")

# 获取第一条轨迹
trajectory_0_states = states[0]  # Shape: (101, 4)
trajectory_0_actions = actions[0] # Shape: (100, 2)
``` 