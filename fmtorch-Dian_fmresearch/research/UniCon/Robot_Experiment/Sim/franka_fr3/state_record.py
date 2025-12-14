#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FR3 Half-Circle Record (MuJoCo)

功能：
- 在固定 x 平面上让末端执行器（EE）画「半圈」（π 弧度）的轨迹。
- 控制：任务空间 resolved-rate（6D）求 q_ref + 关节 PD -> 力矩。
- 只记录：joint space q (7 DoF) 与 EE 的笛卡尔位置 ee (3D)。
- 结束后绘图，并将数据保存为 .npz （默认 state_record.npz）。

示例：
python state_record.py --xml scene.xml --hide-viewer --out state_record.npz 

依赖：mujoco, matplotlib, numpy
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# ---------------------- SO(3) Utils ----------------------
def _so3_log(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]], dtype=float)
    w = (1.0 / (2.0 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0],
                                                  R[1, 0] - R[0, 1]], dtype=float)
    return theta * w

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

# ---------------------- IK-style 6D step (for q_ref) ----------------------
def resolved_rate_step_6d(model, data, site_id,
                          x_des: np.ndarray, v_des: np.ndarray,
                          R_des: np.ndarray, q_ref: np.ndarray,
                          kp_cart: float = 8.0, kp_ori: float = 6.0,
                          damping: float = 0.02, dt: float = 0.002) -> np.ndarray:
    x = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3).copy()
    x_err = x_des - x
    v_cmd = v_des + kp_cart * x_err
    R_err = R_des @ R.T
    w_cmd = kp_ori * _so3_log(R_err)
    J_pos = np.zeros((3, model.nv))
    J_ori = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_pos, J_ori, site_id)
    J6 = np.vstack([J_pos, J_ori])
    y = np.hstack([v_cmd, w_cmd])
    JJt = J6 @ J6.T
    lam2I = (damping ** 2) * np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt + lam2I, y)
    q_ref_new = q_ref + qdot * dt
    # clip joint limits
    idx_q = 0
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            r = model.jnt_range[j]
            if r[0] < r[1]:
                q_ref_new[idx_q] = np.clip(q_ref_new[idx_q], r[0], r[1])
            idx_q += 1
    return q_ref_new

# ---------------------- Helpers ----------------------
def desired_orientation(pos: np.ndarray | None, vel: np.ndarray | None) -> np.ndarray:
    """让 EE 在 x 固定平面内绘制，世界坐标 x 轴作为 EE 的 z 轴朝向（示意）。"""
    z_axis_world = np.array([1.0, 0.0, 0.0])  # 指向世界 x
    up_hint = np.array([0.0, 0.0, 1.0])
    x_axis_world = np.cross(up_hint, z_axis_world)
    if np.linalg.norm(x_axis_world) < 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0])
        x_axis_world = np.cross(up_hint, z_axis_world)
    x_axis_world = _normalize(x_axis_world)
    z_axis_world = _normalize(z_axis_world)
    y_axis_world = np.cross(z_axis_world, x_axis_world)
    return np.column_stack([x_axis_world, y_axis_world, z_axis_world])

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("FR3 half-circle record")
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--dt", type=float, default=0.0, help="controller dt; 0=>use model.opt.timestep")
    parser.add_argument("--hide-viewer", action="store_true")
    parser.add_argument("--out", type=str, default="state_record.npz", help="npz 输出路径")
    # 轨迹参数（在 x 固定平面上，y-z 平面半圆）
    parser.add_argument("--x-plane", type=float, default=0.40)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--omega", type=float, default=0.6, help="角速度 [rad/s]，总时长=pi/omega")
    parser.add_argument("--hold", type=float, default=2.0, help="开始前原地保持 [s]")
    # IK/PD 参数
    parser.add_argument("--kp-cart", type=float, default=8.0)
    parser.add_argument("--kp-ori", type=float, default=6.0)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--kp", type=float, nargs=7, default=[4500,4500,3500,3500,2000,2000,2000])
    parser.add_argument("--kv", type=float, nargs=7, default=[450,450,350,350,200,200,200])
    args = parser.parse_args()

    # 加载模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 设定初始位姿（若有 keyframe=home 则用之）
    key_id = -1
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
            key_id = i; break
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    # 若需要，可自定义初始关节角
    # initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981,  2.7076, 0.7105])
    # data.qpos[:7] = initial_qpos
    mujoco.mj_forward(model, data)

    # 重要对象 ID
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    if tip_id < 0 and site_id >= 0:
        tip_id = site_id  # 兜底使用 attachment_site 作为 EE 位置
    assert tip_id >= 0, "需要存在名为 'tool_tip' 或 'attachment_site' 的 site 以读取 EE 位置"

    # 控制周期
    ctrl_dt = args.dt if args.dt > 0 else model.opt.timestep

    # 构造半圆轨迹的几何中心（基于当前 EE 的初始高度/侧向位置）
    x0 = data.site_xpos[tip_id].copy()
    cy, cz = x0[1], x0[2] + 0.15  # 可根据需要调整中心高度
    x_plane = args.x_plane
    R = args.radius
    omega = args.omega

    # 半圆参数化：s in [0, pi]
    def circle_pos(s: float) -> np.ndarray:
        y = cy + R * np.cos(s)
        z = cz + R * np.sin(s)
        return np.array([x_plane, y, z])

    def circle_vel(s: float) -> np.ndarray:
        # ds/dt = omega
        vy = -R * np.sin(s) * omega
        vz =  R * np.cos(s) * omega
        return np.array([0.0, vy, vz])

    # 将 q_ref 初始化在起点 s=0 处
    q_ref = data.qpos[:7].copy()
    s_target0 = 0.0
    x_target0 = circle_pos(s_target0)
    R_des0 = desired_orientation(None, None)
    for _ in range(200):
        q_ref = resolved_rate_step_6d(model, data, tip_id,
                                      x_des=x_target0, v_des=np.zeros(3),
                                      R_des=R_des0, q_ref=q_ref,
                                      kp_cart=10.0, kp_ori=8.0, damping=0.01, dt=ctrl_dt)
        data.qpos[:7] = q_ref
        mujoco.mj_kinematics(model, data)
        if np.linalg.norm(data.site_xpos[tip_id] - x_target0) < 1e-3:
            break
    mujoco.mj_forward(model, data)

    # PD 增益，将 q_ref -> torque
    Kp = np.array(args.kp, dtype=float)
    Kv = np.array(args.kv, dtype=float)
    ctrl_range = model.actuator_ctrlrange.copy()

    # 记录容器
    times: list[float] = []
    q_hist: list[np.ndarray] = []
    ee_hist: list[np.ndarray] = []

    # 控制+记录一步
    def step_control(t: float):
        nonlocal q_ref
        # 轨迹：t < hold 时保持在起点；t >= hold 后沿半圆前进
        if t < args.hold:
            s = 0.0
        else:
            s = (t - args.hold) * omega
            s = np.clip(s, 0.0, np.pi)
        x_des = circle_pos(s)
        v_des = circle_vel(s) if t >= args.hold else np.zeros(3)
        R_des = desired_orientation(x_des, v_des)

        # 任务空间求 q_ref
        q_ref = resolved_rate_step_6d(model, data, tip_id,
                                      x_des, v_des, R_des, q_ref,
                                      kp_cart=args.kp_cart, kp_ori=args.kp_ori,
                                      damping=args.damping, dt=ctrl_dt)
        # PD -> torque
        tau = Kp * (q_ref - data.qpos[:7]) - Kv * data.qvel[:7]
        tau = np.clip(tau, ctrl_range[:7, 0], ctrl_range[:7, 1])
        data.ctrl[:] = 0.0
        data.ctrl[:7] = tau

        # 记录（pre-step）
        times.append(data.time)
        q_hist.append(data.qpos[:7].copy())
        ee_hist.append(data.site_xpos[tip_id].copy())

        mujoco.mj_step(model, data)

    # 运行：直到画完半圈（s 达到 π）
    total_T = args.hold + np.pi / omega
    print(f"== Half-circle recording: T_total = {total_T:.3f}s (hold={args.hold:.2f}s, draw={np.pi/omega:.3f}s) ==")

    def run_loop(run_viewer: bool):
        if run_viewer:
            try:
                with viewer.launch_passive(model, data) as v:
                    while v.is_running() and data.time < total_T:
                        step_control(data.time)
                        v.sync()
                        time.sleep(0.0005)
            except Exception:
                with viewer.launch(model, data) as v:
                    while v.is_running() and data.time < total_T:
                        step_control(data.time)
                        v.sync()
                        time.sleep(0.0005)
        else:
            while data.time < total_T:
                step_control(data.time)
                time.sleep(0.0002)

    run_loop(run_viewer=not args.hide_viewer)

    # 转为数组并保存
    times_arr = np.array(times)
    q_arr = np.vstack(q_hist) if len(q_hist) else np.zeros((0, 7))
    ee_arr = np.vstack(ee_hist) if len(ee_hist) else np.zeros((0, 3))

    np.savez(args.out, times=times_arr, q=q_arr, ee=ee_arr)
    print(f"Saved npz -> {args.out} | steps={len(times_arr)}")

    # 绘图：关节角 vs 时间
    plt.figure(figsize=(12, 5))
    for j in range(min(7, q_arr.shape[1])):
        plt.plot(times_arr, q_arr[:, j], label=f"q{j+1}")
    plt.xlabel("t (s)"); plt.ylabel("joint angle (rad)")
    plt.title("Joint positions over time")
    plt.legend(ncol=4); plt.grid(True); plt.tight_layout(); plt.show()

    # 绘图：EE 在 y-z 平面的轨迹（半圆）
    plt.figure(figsize=(6, 6))
    if len(ee_arr):
        plt.plot(ee_arr[:, 1], ee_arr[:, 2], linewidth=2)
        plt.scatter([ee_arr[0,1]], [ee_arr[0,2]], marker='o', label='start')
        plt.scatter([ee_arr[-1,1]], [ee_arr[-1,2]], marker='x', label='end')
    plt.xlabel("y (m)"); plt.ylabel("z (m)")
    plt.title("End-effector path in YZ plane (half circle)")
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
