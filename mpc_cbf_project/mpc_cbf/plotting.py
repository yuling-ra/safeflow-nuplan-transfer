# mpc_cbf/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_single_car(xs, config):
    """
    xs: (T+1, 3)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs[:, 0], xs[:, 1], '-o', markersize=2, label='trajectory')

    # 初始点 & 目标点
    ax.plot(xs[0, 0], xs[0, 1], 'gs', label='start')
    ax.plot(config.goal[0, 0], config.goal[1, 0], 'r*', markersize=12, label='goal')

    # 画静态障碍物（圆）
    if config.static_obstacles_on:
        for (xo, yo, ro) in config.obs:
            circle = plt.Circle((xo, yo), ro + config.r + config.safety_dist,
                                fill=False, linestyle='--', label='obstacle')
            ax.add_patch(circle)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Single Car Trajectory with CBF')
    plt.show()

def plot_two_cars(xs, config):
    """
    xs: (T+1, 6)
    """
    x1 = xs[:, 0:3]
    x2 = xs[:, 3:6]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x1[:, 0], x1[:, 1], '-o', markersize=2, label='car 1')
    ax.plot(x2[:, 0], x2[:, 1], '-o', markersize=2, label='car 2')

    ax.plot(x1[0, 0], x1[0, 1], 'gs', label='car1 start')
    ax.plot(x2[0, 0], x2[0, 1], 'bs', label='car2 start')

    ax.plot(config.goal[0, 0], config.goal[1, 0], 'r*', markersize=10, label='car1 goal')
    ax.plot(config.goal[3, 0], config.goal[4, 0], 'm*', markersize=10, label='car2 goal')

    # 可选画障碍
    if config.static_obstacles_on:
        for (xo, yo, ro) in config.obs:
            circle = plt.Circle((xo, yo), ro + max(config.r1, config.r2) + config.safety_dist,
                                fill=False, linestyle='--', label='obstacle')
            ax.add_patch(circle)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Two Cars Head-on with CBF Inter-Vehicle Constraint')
    plt.show()
