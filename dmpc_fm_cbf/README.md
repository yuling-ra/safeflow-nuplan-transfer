# DMPC + Flow Matching + CBF Framework

多车轨迹规划框架，结合分布式模型预测控制(DMPC)、Flow Matching和控制屏障函数(CBF)。

## 核心架构

```
1. 数据集生成: Bicycle dynamics + Optimal Control → 训练轨迹
2. Canonical Frame: 世界坐标 → 标准坐标系 (SE(2) equivariance)
3. Flow Matching训练: 1D UNet学习速度场 (conditional OT path)
4. ODE Rollout + CBF: torchdiffeq积分 + CBF安全投影
5. DMPC控制: K条候选轨迹采样 → 评估 → 最优执行
```

## 安装

```bash
cd dmpc_fm_cbf
pip install -e .

# 完整安装（包括torchdiffeq）
pip install -e ".[full]"
```

## 快速开始

```python
import numpy as np
from dmpc_fm_cbf import SimpleUNet1D, set_seed
from dmpc_fm_cbf.controllers import two_car_fm_dmpc
from dmpc_fm_cbf.training import FlowMatchingTrainer, TrainConfig

# 1. 创建模型
model = SimpleUNet1D(
    cond_dim=3,  # [dist_to_goal, obs_x, obs_y]
    hidden_dim=128,
)

# 2. 训练 (假设已有数据)
# trainer = FlowMatchingTrainer(model, TrainConfig())
# trainer.train(dataset)

# 3. 双车仿真
xsA, xsB = two_car_fm_dmpc(
    xA0=[-1, -1], gA=[1, 1],
    xB0=[1, -1], gB=[-1, 1],
    model=model,
    dt=0.1,
    H=15,
)
```

## 模块结构

```
dmpc_fm_cbf/
├── __init__.py          # 主入口
├── models.py            # 神经网络 (SimpleUNet1D, VelocityMLP)
├── cbf.py               # 控制屏障函数
├── canonical.py         # 坐标变换
├── ode_solver.py        # ODE积分器
├── controllers.py       # DMPC控制器
├── dataset.py           # 数据集生成
├── training.py          # 训练工具
└── utils.py             # 辅助函数
```

## 主要功能

### Flow Matching
- `SimpleUNet1D`: 1D U-Net 用于序列预测
- `AffinePath`: 线性插值路径 x_t = (1-t)x_0 + t*x_1
- `FlowMatchingTrainer`: 训练循环

### CBF 安全保障
- `cbf_project_velocity_world`: 速度场投影
- 支持静态障碍物（椭圆/圆形）
- 支持动态车间避障（压力自适应增益）

### DMPC 控制器
- `two_car_fm_dmpc`: FM + CBF 完整版
- `two_car_pure_dmpc`: 纯DMPC基线
- `two_car_dmpc_with_cbf`: DMPC + CBF执行层

### Canonical Frame
- `canonicalize`: 世界坐标 → 标准坐标
- `uncanonicalize`: 标准坐标 → 世界坐标
- 实现SE(2)等变性，提高泛化能力

## 参数调优建议

### CBF参数
- `agent_safe_radius`: 安全距离（建议0.5-0.8）
- `gamma_min`: 最小增益（建议1.0-6.0）
- `w_p`: 压力权重（建议5.0-10.0）

### DMPC参数
- `H`: 预测视野（建议10-20）
- `K`: 候选数量（建议3-5）
- `speed_scale_B`: B车让行比例（建议0.3-0.5）

### 训练参数
- `batch_size`: 16-64
- `lr`: 1e-4 to 1e-3
- `num_epochs`: 50-200

## Ablation实验

包含三种对比方法：
1. **DMPC + FM + CBF**: 完整版（推荐）
2. **Pure DMPC**: 纯MPC基线
3. **DMPC + CBF**: MPC + 执行层CBF

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.20
- torchdiffeq >= 0.2 (可选，用于高精度ODE)
- matplotlib (可视化)

## 致谢

基于Anthropic Flow Matching和CBF相关研究。
