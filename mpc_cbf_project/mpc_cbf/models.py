# mpc_cbf/models.py
import numpy as np
import do_mpc
from casadi import SX, cos, sin

def get_B_unicycle(x):
    """
    x: casadi SX [3x1] (x, y, theta)
    返回 B(x) [3x2]，输入是 [v, omega]^T
    """
    a = 1e-9  # 保证相对阶为1的小常数
    B = SX.zeros(3, 2)
    theta = x[2]
    B[0, 0] = cos(theta)
    B[0, 1] = -a * sin(theta)
    B[1, 0] = sin(theta)
    B[1, 1] = a * cos(theta)
    B[2, 1] = 1.0
    return B

def build_single_car_model(config):
    """
    离散时间单车模型:
    x_{k+1} = x_k + B(x_k) u_k Ts
    """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # 状态 x ∈ R^3
    _x = model.set_variable(var_type='_x', var_name='x', shape=(3, 1))

    # 控制输入 u ∈ R^2
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))

    # 动力学
    B = get_B_unicycle(_x)
    x_next = _x + B @ _u * config.Ts
    model.set_rhs('x', x_next)

    # cost 表达式
    if config.control_type == "setpoint":
        X_err = _x - config.goal
    else:
        raise NotImplementedError("Only 'setpoint' control is implemented in this template.")

    cost_expr = (X_err.T @ config.Q @ X_err)[0, 0]
    model.set_expression('cost', cost_expr)

    # 静态 / 动态障碍需要的 tvp 参数（这里先只支持 moving obstacles）
    if config.moving_obstacles_on:
        for i in range(len(config.moving_obs)):
            model.set_variable('_tvp', f'x_moving_obs{i}')
            model.set_variable('_tvp', f'y_moving_obs{i}')

    model.setup()
    return model

def build_two_car_model(config):
    """
    集中式两车模型:
      状态: x = [x1, y1, th1, x2, y2, th2]^T ∈ R^6
      输入: u = [v1, w1, v2, w2]^T ∈ R^4
    """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # States
    _x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))

    # Controls
    _u = model.set_variable(var_type='_u', var_name='u', shape=(4, 1))

    # 拆成两个 3x1
    x1 = _x[0:3]
    x2 = _x[3:6]

    u1 = _u[0:2]
    u2 = _u[2:4]

    # 各自的 B(x)
    B1 = get_B_unicycle(x1)  # 3x2
    B2 = get_B_unicycle(x2)  # 3x2

    # block-diagonal 动力学
    x1_next = x1 + B1 @ u1 * config.Ts
    x2_next = x2 + B2 @ u2 * config.Ts

    x_next = SX.zeros(6, 1)
    x_next[0:3] = x1_next
    x_next[3:6] = x2_next

    model.set_rhs('x', x_next)

    # cost: 对两个车分别计算误差，然后用大 Q
    if config.control_type == "setpoint":
        X_err = _x - config.goal  # 6x1
    else:
        raise NotImplementedError("Only 'setpoint' control is implemented in this template.")

    cost_expr = (X_err.T @ config.Q @ X_err)[0, 0]
    model.set_expression('cost', cost_expr)

    # 暂时不考虑 moving obstacle 的 tvp（可类似单车加）

    model.setup()
    return model
