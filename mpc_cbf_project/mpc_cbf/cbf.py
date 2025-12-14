# mpc_cbf/cbf.py
from casadi import SX

def h_obstacle_single(x, obstacle, r_robot, safety_dist):
    """
    单车对圆形障碍物的 CBF:
    h(x) = dist^2 - (r_robot + r_obs + safety)^2
    obstacle: (x_obs, y_obs, r_obs)
    """
    x_obs, y_obs, r_obs = obstacle
    dx = x[0] - x_obs
    dy = x[1] - y_obs
    h = dx**2 + dy**2 - (r_robot + r_obs + safety_dist)**2
    return h

def h_inter_vehicle(x, config):
    """
    两车之间的 CBF:
      把对方视为障碍物:
        center = (x2, y2),
        radius = r1 + r2
      h(x) = ||p1 - p2||^2 - (r1 + r2 + safety)^2
    """
    x1 = x[0:3]
    x2 = x[3:6]

    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]

    r_total = config.r1 + config.r2 + config.safety_dist
    h = dx**2 + dy**2 - r_total**2
    return h

def discrete_cbf_constraint(h_k1, h_k, gamma):
    """
    离散 CBF 条件:
      h(x_{k+1}) >= (1 - gamma) * h(x_k)
    等价为:
      -h(x_{k+1}) + (1 - gamma) * h(x_k) <= 0
    """
    return -h_k1 + (1 - gamma) * h_k
