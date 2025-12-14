# mpc_cbf/mpc_setup.py
import numpy as np
import do_mpc
from .cbf import h_obstacle_single, h_inter_vehicle, discrete_cbf_constraint
from .models import build_single_car_model, build_two_car_model

def build_single_car_mpc_bundle(config):
    """
    返回一个 dict:
      {
        'model': model,
        'mpc': mpc,
        'simulator': sim,
        'estimator': est
      }
    """
    model = build_single_car_model(config)
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': config.T_horizon,
        't_step': config.Ts,
        'state_discretization': 'discrete',
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # objective
    mterm = model.aux['cost']
    lterm = model.aux['cost']
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=config.R)

    # 输入约束
    max_u = np.array([config.v_limit, config.omega_limit])
    mpc.bounds['lower', '_u', 'u'] = -max_u
    mpc.bounds['upper', '_u', 'u'] =  max_u

    # CBF 约束（静态障碍）
    if config.static_obstacles_on and config.controller == "MPC-CBF":
        x = model.x['x']      # 3x1
        u = model.u['u']      # 2x1
        Ts = config.Ts

        from .models import get_B_unicycle
        B = get_B_unicycle(x)
        x_k1 = x + B @ u * Ts

        i = 0
        for obs in config.obs:
            h_k  = h_obstacle_single(x,    obs, config.r, config.safety_dist)
            h_k1 = h_obstacle_single(x_k1, obs, config.r, config.safety_dist)
            cbc = discrete_cbf_constraint(h_k1, h_k, config.gamma)
            mpc.set_nl_cons(f'cbf_obs_{i}', cbc, ub=0.0)
            i += 1

    # 如果启用 moving obstacles，可以在这里类似添加对应的 CBF 约束

    # 时间变化参数 tvp
    if config.moving_obstacles_on:
        tvp_struct_mpc = mpc.get_tvp_template()
        def tvp_fun_mpc(t_now):
            return tvp_struct_mpc
        mpc.set_tvp_fun(tvp_fun_mpc)

    mpc.setup()

    # Simulator & Estimator
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=config.Ts)

    if config.moving_obstacles_on:
        tvp_template = simulator.get_tvp_template()
        def tvp_fun(t_now):
            return tvp_template
        simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    # 设置初始状态
    mpc.x0 = config.x0
    simulator.x0 = config.x0
    estimator.x0 = config.x0
    mpc.set_initial_guess()

    return {
        'model': model,
        'mpc': mpc,
        'simulator': simulator,
        'estimator': estimator
    }

def build_two_car_mpc_bundle(config):
    """
    集中式两车 MPC + CBF (车-车避障)
    """
    model = build_two_car_model(config)
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': config.T_horizon,
        't_step': config.Ts,
        'state_discretization': 'discrete',
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # objective
    mterm = model.aux['cost']
    lterm = model.aux['cost']
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=config.R)

    # 输入约束 (v1,w1,v2,w2)
    max_u = np.array([config.v_limit, config.omega_limit,
                      config.v_limit, config.omega_limit])
    mpc.bounds['lower', '_u', 'u'] = -max_u
    mpc.bounds['upper', '_u', 'u'] =  max_u

    # CBF 约束：车-车之间 + (可选) 静态障碍
    if config.controller == "MPC-CBF":
        x = model.x['x']   # 6x1
        u = model.u['u']   # 4x1

        from casadi import SX
        from .models import get_B_unicycle

        Ts = config.Ts
        x1 = x[0:3]
        x2 = x[3:6]
        u1 = u[0:2]
        u2 = u[2:4]

        B1 = get_B_unicycle(x1)
        B2 = get_B_unicycle(x2)
        x1_k1 = x1 + B1 @ u1 * Ts
        x2_k1 = x2 + B2 @ u2 * Ts

        x_k1 = SX.zeros(6, 1)
        x_k1[0:3] = x1_k1
        x_k1[3:6] = x2_k1

        # 车-车 CBF
        h_k  = h_inter_vehicle(x,    config)
        h_k1 = h_inter_vehicle(x_k1, config)
        cbc_12 = discrete_cbf_constraint(h_k1, h_k, config.gamma)
        mpc.set_nl_cons('cbf_inter_vehicle', cbc_12, ub=0.0)

        # 如果还要静态障碍，对每个车分别加：
        if config.static_obstacles_on:
            i = 0
            for obs in config.obs:
                h1_k  = h_obstacle_single(x1,    obs, config.r1, config.safety_dist)
                h1_k1 = h_obstacle_single(x1_k1, obs, config.r1, config.safety_dist)
                cbc1 = discrete_cbf_constraint(h1_k1, h1_k, config.gamma)
                mpc.set_nl_cons(f'cbf_car1_obs_{i}', cbc1, ub=0.0)

                h2_k  = h_obstacle_single(x2,    obs, config.r2, config.safety_dist)
                h2_k1 = h_obstacle_single(x2_k1, obs, config.r2, config.safety_dist)
                cbc2 = discrete_cbf_constraint(h2_k1, h2_k, config.gamma)
                mpc.set_nl_cons(f'cbf_car2_obs_{i}', cbc2, ub=0.0)
                i += 1

    mpc.setup()

    # Simulator & Estimator
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=config.Ts)
    simulator.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    # 初始状态
    mpc.x0 = config.x0
    simulator.x0 = config.x0
    estimator.x0 = config.x0
    mpc.set_initial_guess()

    return {
        'model': model,
        'mpc': mpc,
        'simulator': simulator,
        'estimator': estimator
    }
