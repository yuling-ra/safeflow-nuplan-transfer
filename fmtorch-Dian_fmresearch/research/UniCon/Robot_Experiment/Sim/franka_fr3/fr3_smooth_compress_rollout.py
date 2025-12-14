#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FR3 — Smooth Control -> Keypoint Compression -> Decompress + Align -> Open-loop Rollout (MuJoCo)

目标：
1) 基于末端 8 字任务轨迹，用 computed-torque（逆动力学+PD）生成“高频平滑”的力矩并仿真；
2) 用关键点选择（1~5% 等）+ 样条/FOH 对力矩进行压缩与解压（非 ZOH）；
3) 只对齐第一帧，开环重放解压后的力矩；
4) 评估并可视化（力矩/轨迹/误差），便于后续被关键点编码与 Flow Matching 训练。

不读取/写入 .npz；控制器“whatever”取最平滑可用的 computed-torque。
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt


# ---------------------- SO(3) & 任务空间步进（用于生成参考 q_ref） ----------------------
def _so3_log(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=float)
    w = (1.0/(2.0*np.sin(theta))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=float)
    return theta * w

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def resolved_rate_step_6d(model, data, site_id,
                          x_des, v_des, R_des, q_ref,
                          kp_cart=10.0, kp_ori=8.0, damping=0.01, dt=0.002):
    """任务空间速度控制，返回 (q_ref_new, qdot_ref)"""
    x = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3,3).copy()
    x_err = x_des - x
    v_cmd = v_des + kp_cart * x_err
    R_err = R_des @ R.T
    w_cmd = kp_ori * _so3_log(R_err)

    Jp = np.zeros((3, model.nv)); Jo = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jo, site_id)
    J6 = np.vstack([Jp, Jo])
    y  = np.hstack([v_cmd, w_cmd])

    JJt = J6 @ J6.T
    qdot = J6.T @ np.linalg.solve(JJt + (damping**2)*np.eye(6), y)

    q_new = q_ref + qdot*dt
    # clip 到关节范围
    idx_q = 0
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            r = model.jnt_range[j]
            if r[0] < r[1]:
                q_new[idx_q] = np.clip(q_new[idx_q], r[0], r[1])
            idx_q += 1
    return q_new, qdot[:7]


# ---------------------- 自带的“自然三次样条”（无 SciPy 依赖） ----------------------
def _cubic_fit(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    N = len(t); h = np.diff(t)
    A = np.zeros((N, N)); b = np.zeros(N)
    A[0,0] = A[-1,-1] = 1.0
    for i in range(1, N-1):
        A[i,i-1] = h[i-1]; A[i,i] = 2*(h[i-1]+h[i]); A[i,i+1] = h[i]
        b[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    m = np.linalg.solve(A, b)
    return {"t": t, "y": y, "h": h, "m": m}

def _cubic_eval(spl, x):
    t = spl["t"]; y = spl["y"]; h = spl["h"]; m = spl["m"]
    idx = np.searchsorted(t, x, side="right") - 1
    idx = np.clip(idx, 0, len(t)-2)
    k = idx; hk = h[k]
    a = (t[k+1]-x)/hk; b = (x-t[k])/hk
    return a*y[k] + b*y[k+1] + ((a**3-a)*m[k] + (b**3-b)*m[k+1])*(hk**2)/6.0

def cubic_fit_eval(tk, yk, t_eval):
    return _cubic_eval(_cubic_fit(tk, yk), t_eval)


# ---------------------- 关键点选择（联合多关节，贪心最大误差） ----------------------
def greedy_keyframes(t, Y, target_pct=5, max_err=np.inf):
    """
    t: (N,), Y: (N, J)
    返回：保留的整型索引列表（升序）
    """
    N, J = Y.shape
    K = max(2, int(np.ceil(N*target_pct/100.0)))
    keep = set([0, N-1])

    def recon(idxset):
        idx = np.array(sorted(idxset))
        Yr = np.empty_like(Y)
        for j in range(J):
            Yr[:, j] = cubic_fit_eval(t[idx], Y[idx, j], t)
        return Yr

    while len(keep) < K:
        Yr = recon(keep)
        err = np.linalg.norm(Y - Yr, axis=1)  # 每时刻 L2 误差
        err[list(keep)] = -1.0
        i_star = int(np.argmax(err))
        # 只有当提供了有限阈值时才早停
        if np.isfinite(max_err) and err[i_star] <= max_err:
            break
        keep.add(i_star)
    return sorted(list(keep))

def reconstruct_series(t, Y, idx_keep, method="cubic"):
    idx = np.array(sorted(idx_keep))
    Yr = np.empty_like(Y)
    if method == "foh":
        for j in range(Y.shape[1]):
            Yr[:, j] = np.interp(t, t[idx], Y[idx, j])
    elif method == "cubic":
        for j in range(Y.shape[1]):
            Yr[:, j] = cubic_fit_eval(t[idx], Y[idx, j], t)
    else:
        raise ValueError("method must be 'cubic' or 'foh'")
    return Yr


# ---------------------- 相位对齐（整数步移位） ----------------------
def best_phase_shift(ref, cand, max_shift_steps, window_steps):
    N = ref.shape[0]; W = min(window_steps, N)
    best_s, best_e = 0, np.inf
    for s in range(-max_shift_steps, max_shift_steps+1):
        if s >= 0:
            r0 = ref[:W]; c0 = cand[s:s+W]
        else:
            r0 = ref[-s:-s+W]; c0 = cand[:W]
        if len(c0) != len(r0) or len(c0) == 0: continue
        e = np.linalg.norm(r0 - c0)
        if e < best_e: best_e, best_s = e, s
    return best_s


# ---------------------- 主流程 ----------------------
def main():
    ap = argparse.ArgumentParser("FR3 smooth control -> keypoints -> reconstruct -> align -> rollout")
    ap.add_argument("--xml", type=str, required=True, help="FR3 MuJoCo XML (torque motors)")
    ap.add_argument("--T", type=float, default=15.0, help="total duration (s)")
    ap.add_argument("--seed", type=int, default=0)

    # 控制器：computed-torque （逆动力学 + 关节 PD）
    ap.add_argument("--kp", type=float, nargs=7, default=[100,100,80,80,50,50,30])
    ap.add_argument("--kd", type=float, nargs=7, default=[10,10,8,8,5,5,3])
    ap.add_argument("--tau-smooth-ms", type=float, default=20.0, help="EMA smoothing of torque (ms)")

    # 压缩与重建
    ap.add_argument("--compress-pct", type=int, default=5, help="keyframe keep percentage (1..100)")
    ap.add_argument("--method", type=str, default="cubic", choices=["cubic","foh"], help="reconstruction method")

    # 相位对齐
    ap.add_argument("--align-search", type=float, default=1.0, help="±search seconds")
    ap.add_argument("--align-window", type=float, default=0.5, help="window seconds for alignment")

    # 8 字轨迹参数（末端 y-z 平面）
    ap.add_argument("--x-plane", type=float, default=0.40)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--omega", type=float, default=0.6)
    ap.add_argument("--phi",   type=float, default=0.0)

    ap.add_argument("--hide-viewer", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)

    # 模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    dt = model.opt.timestep
    N  = int(np.round(args.T / dt))
    t  = np.arange(N) * dt

    data = mujoco.MjData(model)
    # 尝试 keyframe 'home'
    if model.nkey > 0:
        for i in range(model.nkey):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
                mujoco.mj_resetDataKeyframe(model, data, i); break
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    assert site_id >= 0 and tip_id >= 0, "XML 需包含 site: attachment_site / tool_tip"

    # 任务几何
    x_plane = args.x_plane
    radius  = args.radius
    omega   = args.omega
    phi     = args.phi

    x0_home = data.site_xpos[site_id].copy()
    cy, cz = x0_home[1], x0_home[2] + 0.15
    Ay = radius
    Az = 0.5*radius

    def desired_orientation(pos, vel):
        # 工具 x 朝世界 x（水平），其余正交
        z_axis_world = np.array([1.0, 0.0, 0.0])
        up_hint = np.array([0.0, 0.0, 1.0])
        x_axis_world = _normalize(np.cross(up_hint, z_axis_world))
        z_axis_world = _normalize(z_axis_world)
        y_axis_world = np.cross(z_axis_world, x_axis_world)
        return np.column_stack([x_axis_world, y_axis_world, z_axis_world])

    R_h = desired_orientation(None, None)

    # ---------- 生成 q_ref, dq_ref, ddq_ref（用 RR 步进） ----------
    q_ref = np.zeros((N,7)); dq_ref = np.zeros_like(q_ref); ddq_ref = np.zeros_like(q_ref)
    q_guess = data.qpos[:7].copy()
    for k in range(N):
        tau = t[k]
        y_des = cy + Ay * np.sin(omega * tau)
        z_des = cz + Az * np.sin(2.0 * omega * tau + phi)
        x_des = np.array([x_plane, y_des, z_des], float)
        vy = Ay * omega * np.cos(omega * tau)
        vz = 2.0 * Az * omega * np.cos(2.0 * omega * tau + phi)
        v_des = np.array([0.0, vy, vz], float)

        data.qpos[:7] = q_guess
        mujoco.mj_forward(model, data)
        q_next, qdot_ref = resolved_rate_step_6d(model, data, site_id, x_des, v_des, R_h,
                                                 q_ref=q_guess, kp_cart=10.0, kp_ori=8.0, damping=0.01, dt=dt)
        q_ref[k]  = q_next
        dq_ref[k] = qdot_ref
        if k > 0:
            ddq_ref[k] = (dq_ref[k] - dq_ref[k-1]) / dt
        q_guess = q_next.copy()
    ddq_ref[0] = ddq_ref[1]

    # ---------- computed-torque 控制生成“平滑高频” τ，并仿真 ----------
    data_sim = mujoco.MjData(model)
    data_sim.qpos[:7] = q_ref[0]; data_sim.qvel[:7] = dq_ref[0]
    mujoco.mj_forward(model, data_sim)

    Kp = np.array(args.kp, float); Kd = np.array(args.kd, float)
    ctrl_range = model.actuator_ctrlrange.copy()

    tau_hist = np.zeros((N,7)); q_hist = np.zeros((N,7)); dq_hist = np.zeros((N,7))
    tau_filt = np.zeros(7)
    alpha = dt / (args.tau_smooth_ms*1e-3 + dt) if args.tau_smooth_ms > 0 else 1.0  # EMA 系数；=1 等于无滤波

    for k in range(N):
        # M, bias
        M = np.zeros((model.nv, model.nv)); mujoco.mj_fullM(model, M, data_sim.qM)
        M7 = M[:7,:7]
        bias = data_sim.qfrc_bias[:7].copy()  # C + g + passive

        # 期望加速度 + PD
        ddq_des = ddq_ref[k] + Kp*(q_ref[k]-data_sim.qpos[:7]) + Kd*(dq_ref[k]-data_sim.qvel[:7])
        tau_cmd = M7 @ ddq_des + bias

        # 一阶低通（让 τ 更顺滑、带宽受控）
        tau_filt = tau_filt + alpha*(tau_cmd - tau_filt)

        # 裁剪并施加
        tau_apply = np.clip(tau_filt, ctrl_range[:7,0], ctrl_range[:7,1])
        data_sim.ctrl[:] = 0.0
        data_sim.ctrl[:7] = tau_apply
        mujoco.mj_step(model, data_sim)

        tau_hist[k] = tau_apply
        q_hist[k]   = data_sim.qpos[:7]
        dq_hist[k]  = data_sim.qvel[:7]

    # ---------- 压缩（关键点）与重建（非 ZOH） ----------
    pct = int(np.clip(args.compress_pct, 1, 100))
    if pct >= 100:
        idx_keep = list(range(N))       # 不压缩
    else:
        idx_keep = greedy_keyframes(t, tau_hist, target_pct=pct)
    tau_rec = reconstruct_series(t, tau_hist, idx_keep, method=args.method)

    # ---------- 只对齐第一帧 + 开环重放 ----------
    max_shift = int(np.round(args.align_search / dt))
    win_steps = int(np.round(args.align_window / dt))
    s_best = best_phase_shift(tau_hist, tau_rec, max_shift, win_steps)

    if s_best >= 0:
        tau_play = tau_rec[s_best:]
        q0, dq0  = q_hist[s_best], dq_hist[s_best]
    else:
        tau_play = tau_rec[:s_best]
        q0, dq0  = q_hist[0], dq_hist[0]

    Mlen = min(len(tau_play), N)
    tau_play = tau_play[:Mlen]
    q_ref_play = q_hist[:Mlen]

    data_play = mujoco.MjData(model)
    data_play.qpos[:7] = q0; data_play.qvel[:7] = dq0
    mujoco.mj_forward(model, data_play)

    q_roll = np.zeros((Mlen,7))
    for k in range(Mlen):
        data_play.ctrl[:] = 0.0
        data_play.ctrl[:7] = tau_play[k]
        mujoco.mj_step(model, data_play)
        q_roll[k] = data_play.qpos[:7]

    # ---------- 指标 ----------
    comp_ratio = len(idx_keep)/N
    tau_mse = float(np.mean((tau_hist[:Mlen] - tau_rec[:Mlen])**2))
    q_mse   = float(np.mean((q_ref_play      - q_roll        )**2))

    print("\n=== Results ===")
    print(f"dt={dt:.6f}s, N={N}, T={N*dt:.3f}s")
    print(f"Computed-torque Kp={Kp.tolist()}, Kd={Kd.tolist()}, EMA={args.tau_smooth_ms:.1f} ms")
    print(f"Compression keep {len(idx_keep)}/{N} -> {comp_ratio*100:.2f}% (target {pct}%)  method={args.method}")
    print(f"Phase best shift: {s_best} steps (~{s_best*dt:.3f}s)")
    print(f"MSE(tau) recon vs smooth : {tau_mse:.4e}")
    print(f"MSE(q)   rollout diff    : {q_mse:.4e}")

    # ---------- 可视化播放（可选） ----------
    if not args.hide_viewer:
        try:
            data_vis = mujoco.MjData(model)
            data_vis.qpos[:7] = q0; data_vis.qvel[:7] = dq0
            mujoco.mj_forward(model, data_vis)
            with viewer.launch_passive(model, data_vis) as v:
                i = 0
                while v.is_running() and i < Mlen:
                    data_vis.ctrl[:] = 0.0
                    data_vis.ctrl[:7] = tau_play[i]
                    mujoco.mj_step(model, data_vis)
                    v.sync(); i += 1
        except Exception:
            pass

    # ---------- 画图 ----------
    if not args.no_plot:
        tt = np.arange(Mlen)*dt
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        ax = axes[0,0]
        for j in range(7):
            ax.plot(t, tau_hist[:,j], label=f"J{j+1}")
        ax.set_title("Smooth torque (generated)"); ax.set_xlabel("t(s)"); ax.set_ylabel("Nm"); ax.grid(True)

        ax = axes[0,1]
        for j in range(7):
            ax.plot(t, tau_rec[:,j], label=f"J{j+1}")
        ax.set_title(f"Reconstructed torque ({args.method}, keep~{comp_ratio*100:.1f}%)")
        ax.set_xlabel("t(s)"); ax.set_ylabel("Nm"); ax.grid(True)

        ax = axes[1,0]
        for j in range(7):
            ax.plot(t[:Mlen], q_ref_play[:,j])
        ax.set_title("Reference rollout q (generated torque)")
        ax.set_xlabel("t(s)"); ax.set_ylabel("rad"); ax.grid(True)

        ax = axes[1,1]
        for j in range(7):
            ax.plot(tt, q_roll[:,j])
        ax.set_title("Open-loop q with reconstructed torque")
        ax.set_xlabel("t(s)"); ax.set_ylabel("rad"); ax.grid(True)

        ax = axes[2,0]
        ax.plot(t[:Mlen], np.linalg.norm(tau_hist[:Mlen]-tau_rec[:Mlen], axis=1), 'r-')
        ax.set_title("||tau_err||"); ax.grid(True)

        ax = axes[2,1]
        ax.plot(tt, np.linalg.norm(q_ref_play - q_roll, axis=1), 'r-')
        ax.set_title("||q_err||"); ax.grid(True)

        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
