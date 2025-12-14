# mpc_cbf/config_single.py
import numpy as np

# 仿真与时间设置
sim_time = 200        # 仿真步数
Ts = 0.1              # 采样时间
T_horizon = 20        # MPC 预测步长

# 机器人初始状态 x0 = [x, y, theta]^T
x0 = np.array([[ -4.0],
               [  0.0],
               [  0.0]])   # 朝向正 x 轴

# 目标点 (setpoint 控制)
control_type = "setpoint"
goal = np.array([[ 4.0],
                 [ 0.0],
                 [ 0.0]])

# 速度限制
v_limit = 0.8           # m/s
omega_limit = 1.0       # rad/s

# cost 权重
Q = np.diag([10.0, 10.0, 1.0])   # 状态误差
R = np.diag([0.1, 0.1])          # 控制输入

# 机器人半径 + 安全距离
r = 0.3
safety_dist = 0.2

# CBF 参数
gamma = 0.5   # 离散 CBF 中的 (1 - gamma)

# 是否使用 CBF / 直接距离约束
controller = "MPC-CBF"  # 或 "MPC-DC"

# 障碍物配置
static_obstacles_on = True
moving_obstacles_on = False

# 静态障碍物列表: [(x_obs, y_obs, r_obs), ...]
obs = [
    (0.0, 0.0, 0.4),
]

# 如果你之后要测移动障碍，可以改为 True 并配置 moving_obs
moving_obs = []   # 这里只用不到
