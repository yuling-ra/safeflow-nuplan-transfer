# mpc_cbf/simulate.py
import numpy as np

def run_single_car_closed_loop(bundle, config):
    """
    返回:
      xs: (sim_time+1, 3)  状态轨迹
      us: (sim_time, 2)    控制轨迹
    """
    mpc = bundle['mpc']
    sim = bundle['simulator']
    est = bundle['estimator']

    x = config.x0.copy()
    xs = [x.flatten()]
    us = []

    for k in range(config.sim_time):
        u = mpc.make_step(x)
        y_next = sim.make_step(u)
        x = est.make_step(y_next)

        xs.append(x.flatten())
        us.append(u.flatten())

    xs = np.array(xs)  # (T+1, 3)
    us = np.array(us)  # (T, 2)
    return xs, us

def run_two_car_closed_loop(bundle, config):
    """
    集中式两车 closed-loop:
      xs: (sim_time+1, 6)
      us: (sim_time, 4)
    """
    mpc = bundle['mpc']
    sim = bundle['simulator']
    est = bundle['estimator']

    x = config.x0.copy()
    xs = [x.flatten()]
    us = []

    for k in range(config.sim_time):
        u = mpc.make_step(x)
        y_next = sim.make_step(u)
        x = est.make_step(y_next)

        xs.append(x.flatten())
        us.append(u.flatten())

    xs = np.array(xs)  # (T+1, 6)
    us = np.array(us)  # (T, 4)
    return xs, us
