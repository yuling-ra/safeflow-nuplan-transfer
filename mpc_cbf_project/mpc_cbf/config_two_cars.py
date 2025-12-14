# mpc_cbf/config_two_cars.py
import numpy as np

# 仿真设置
sim_time = 200
Ts = 0.1
T_horizon = 20

# 两个小车的初始状态
# 车1在左边往右，车2在右边往左
x1_0 = np.array([[-4.0], [0.0], [0.0]])
x2_0 = np.array([[ 4.0], [0.0], [np.pi]])   # 朝向负 x 轴

x0 = np.vstack([x1_0, x2_0])  # 6x1

# 每一辆车的目标（互换位置）
control_type = "setpoint"
goal1 = np.array([[ 4.0], [0.0], [0.0]])
goal2 = np.array([[-4.0], [0.0], [np.pi]])
goal = np.vstack([goal1, goal2])  # 6x1

# 速度限制
v_limit = 0.8
omega_limit = 1.0

# cost 权重（对两个车分别套 Q_single）
Q_single = np.diag([10.0, 10.0, 1.0])
Q = np.block([
    [Q_single,           np.zeros((3, 3))],
    [np.zeros((3, 3)),   Q_single]
])  # 6x6

R_single = np.diag([0.1, 0.1])
R = np.block([
    [R_single,           np.zeros((2, 2))],
    [np.zeros((2, 2)),   R_single]
])  # 4x4

# 机器人几何 + 安全参数
r1 = 0.3
r2 = 0.3
safety_dist = 0.2
gamma = 0.5
controller = "MPC-CBF"

# 障碍物配置：可以复用单车的静态障碍，也可以先关掉
static_obstacles_on = False
moving_obstacles_on = False
obs = []
moving_obs = []
