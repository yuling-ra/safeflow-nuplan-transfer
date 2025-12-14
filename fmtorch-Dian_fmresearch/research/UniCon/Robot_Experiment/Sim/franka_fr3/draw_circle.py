#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer


# ---------------------- SO(3) 与姿态工具函数 ----------------------
def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def _so3_log(R: np.ndarray) -> np.ndarray:
    """旋转矩阵对数映射：R -> 轴角向量（角速度误差的合理近似）"""
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        # 小角度近似：w ≈ vee(R - R^T)/2
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]], dtype=float)
    w = (1.0 / (2.0 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0],
                                                  R[1, 0] - R[0, 1]], dtype=float)
    return theta * w

def _yaw_from_R(R: np.ndarray) -> float:
    """ZYX 欧拉约定下的偏航（绕世界 z）"""
    # 对应 yaw = atan2(r10, r00)
    return float(np.arctan2(R[1, 0], R[0, 0]))


# ---------------------- 6D 解析速度步进（位置+姿态） ----------------------
def resolved_rate_step_6d(model, data, site_id,
                          x_des: np.ndarray, v_des: np.ndarray,
                          R_des: np.ndarray, q_ref: np.ndarray,
                          kp_cart: float = 8.0, kp_ori: float = 6.0,
                          damping: float = 0.02, dt: float = 0.002) -> np.ndarray:
    """
    同时跟踪末端位置 x_des/v_des 与姿态 R_des（水平约束）。
    通过 6×nv 的雅可比做阻尼最小二乘，积分得到新的 q_ref。
    """
    # --- 当前末端位姿 ---
    x = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3).copy()

    # --- 位置误差与线速度指令 ---
    x_err = x_des - x
    v_cmd = v_des + kp_cart * x_err  # (3,)

    # --- 姿态误差与角速度指令：R_err = R_des * R^T ---
    R_err = R_des @ R.T
    w_cmd = kp_ori * _so3_log(R_err)  # (3,)

    # --- 末端雅可比（位置 & 角速度，世界系） ---
    J_pos = np.zeros((3, model.nv))
    J_ori = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_pos, J_ori, site_id)
    J6 = np.vstack([J_pos, J_ori])  # (6, nv)

    # --- 阻尼最小二乘 ---
    y = np.hstack([v_cmd, w_cmd])  # (6,)
    JJt = J6 @ J6.T
    lam2I = (damping ** 2) * np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt + lam2I, y)  # (nv,)

    # --- 积分并夹到关节范围（针对 hinge/slide） ---
    q_ref_new = q_ref + qdot * dt
    idx_q = 0
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            r = model.jnt_range[j]
            if r[0] < r[1]:
                q_ref_new[idx_q] = np.clip(q_ref_new[idx_q], r[0], r[1])
            idx_q += 1
    return q_ref_new


# ---------------------- 主程序 ----------------------
def main():
    parser = argparse.ArgumentParser(description="FR3 EE draw circle on plane x = const (MuJoCo), with level tool.")
    parser.add_argument("--xml", type=str, default="scene.xml", help="Path to FR3 MuJoCo XML.")
    parser.add_argument("--x-plane", type=float, default=0.40, help="Target plane x (m).")
    parser.add_argument("--radius", type=float, default=0.1, help="Circle radius (m).")
    parser.add_argument("--omega", type=float, default=0.6, help="Angular speed (rad/s).")
    parser.add_argument("--kp-cart", type=float, default=8.0, help="Cartesian P gain.")
    parser.add_argument("--kp-ori", type=float, default=6.0, help="Orientation P gain.")
    parser.add_argument("--damping", type=float, default=0.02, help="DLS lambda.")
    parser.add_argument("--dt", type=float, default=0.0, help="Control dt (s). 0 => use model.opt.timestep")
    parser.add_argument("--warmup", type=float, default=0.5, help="Hold initial pose seconds.")
    parser.add_argument("--yaw-mode", type=str, default="fixed", choices=["fixed", "tangent"],
                        help="fixed: 保持初始偏航；tangent: 沿圆的切向偏航（仍保持水平）")
    parser.add_argument("--site", type=str, default="attachment_site", help="End-effector site name.")
    parser.add_argument("--hide-viewer", action="store_true", help="Run headless (no GUI).")
    args = parser.parse_args()

    # 载入模型与数据
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # reset 到 keyframe "home"（如存在）
    key_id = -1
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
            key_id = i
            break
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    
    # 设置初始关节位置
    initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105])
    data.qpos[:7] = initial_qpos

    # 控制与时间步
    nu = model.nu
    assert nu >= 7, "需要 position 型执行器（与你的 XML 一致），并至少 7 个。"
    sim_dt = model.opt.timestep
    ctrl_dt = args.dt if args.dt > 0 else sim_dt

    # 末端 site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    if site_id < 0:
        raise RuntimeError(f"Site '{args.site}' not found.")

    mujoco.mj_forward(model, data)
    x0 = data.site_xpos[site_id].copy()
    R0 = data.site_xmat[site_id].reshape(3, 3).copy()

    # 圆参数（以当前 y,z 为圆心）
    cy, cz = x0[1], x0[2]
    x_plane, radius, omega = args.x_plane, args.radius, args.omega

    # —— “水平”约束：末端 z 轴与世界 z 对齐；偏航按模式确定 ——
    yaw0 = _yaw_from_R(R0)

    def _normalize(v, eps=1e-12):
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    def desired_orientation(t: float, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        让末端坐标系的 z 轴（棒轴）对齐世界 +X。
        其余轴用 up_hint 构右手正交基（默认世界 +Z 为“上”）。
        """
        z_axis_world = np.array([1.0, 0.0, 0.0])   # 棒轴(局部 z) -> 世界 +X
        up_hint      = np.array([0.0, 0.0, 1.0])   # 偏好的“上”方向

        x_axis_world = np.cross(up_hint, z_axis_world)
        if np.linalg.norm(x_axis_world) < 1e-8:
            up_hint = np.array([0.0, 1.0, 0.0])
            x_axis_world = np.cross(up_hint, z_axis_world)

        x_axis_world = _normalize(x_axis_world)
        z_axis_world = _normalize(z_axis_world)
        y_axis_world = np.cross(z_axis_world, x_axis_world)

        # 列向量为 (x, y, z)
        R_des = np.column_stack([x_axis_world, y_axis_world, z_axis_world])
        return R_des



    # 关节参考（用作 position actuator 的 ctrl）
    q_ref = data.qpos[:7].copy()  # FR3 七关节

    # 获取工具尖端的 site ID
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    if tool_tip_site_id < 0:
        raise RuntimeError("Site 'tool_tip' not found.")
    
    # 用于控制打印频率的变量
    last_print_time = 0.0
    print_interval = 0.1  # 每0.1秒打印一次

    # warmup：保持当前
    warm_steps = max(0, int(args.warmup / sim_dt))
    for _ in range(warm_steps):
        data.ctrl[:7] = q_ref
        mujoco.mj_step(model, data)

    # 控制回调
    def control_step():
        nonlocal q_ref, last_print_time
        t = data.time

        # 期望圆轨迹（x 固定在平面 x = const，y-z 画圆）
        y_des = cy + radius * np.cos(omega * t)
        z_des = cz + radius * np.sin(omega * t)
        x_des = np.array([x_plane, y_des, z_des], dtype=float)

        # 期望线速度（只在 y-z 平面有速度）
        vy = -radius * omega * np.sin(omega * t)
        vz =  radius * omega * np.cos(omega * t)
        v_des = np.array([0.0, vy, vz], dtype=float)

        # 期望水平姿态（根据模式确定偏航）
        R_des = desired_orientation(t, x_des, v_des)

        # 6D 解析速度步进
        q_ref = resolved_rate_step_6d(
            model, data, site_id,
            x_des, v_des, R_des, q_ref,
            kp_cart=args.kp_cart, kp_ori=args.kp_ori,
            damping=args.damping, dt=ctrl_dt
        )
        data.ctrl[:7] = q_ref
        
        # 每0.1秒打印一次末端小棒末端的绝对位置
        if t - last_print_time >= print_interval:
            tool_tip_pos = data.site_xpos[tool_tip_site_id].copy()
            print(f"时间: {t:.3f}s, 工具尖端位置: [{tool_tip_pos[0]:.4f}, {tool_tip_pos[1]:.4f}, {tool_tip_pos[2]:.4f}]")
            last_print_time = t
        
        # 注释掉的qpos打印
        # qpos_str = ", ".join([f"{q:.4f}" for q in data.qpos[:7]])
        # print(f"时间: {t:.3f}s, qpos: [{qpos_str}]")

    # 仿真循环
    if args.hide_viewer:
        while True:
            control_step()
            mujoco.mj_step(model, data)
            time.sleep(0.0005)  # 稍微让出 CPU
    else:
        try:
            with viewer.launch_passive(model, data) as v:
                while v.is_running():
                    control_step()
                    mujoco.mj_step(model, data)
                    v.sync()
                    time.sleep(0.0005)
        except Exception:
            with viewer.launch(model, data) as v:
                while v.is_running():
                    control_step()
                    mujoco.mj_step(model, data)
                    v.sync()
                    time.sleep(0.0005)


if __name__ == "__main__":
    main()
