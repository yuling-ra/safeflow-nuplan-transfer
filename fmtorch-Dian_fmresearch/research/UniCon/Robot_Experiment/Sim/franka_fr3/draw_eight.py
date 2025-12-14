#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FR3 Perfect Torque Replay (MuJoCo) — Slim Storage

要点（与旧版相比的变化）：
1) 存储极简：只保存 q0,dq0, dt, times(T,), tau(T,7), qpos(T,7), qvel(T,7)，可选 xfrc_tool(T,6)（仅 --pert 时）。
   - 全部 float32 + np.savez_compressed，避免 object 和 allow_pickle。
   - 不再保存 pre_* / post_* 全状态，不再保存全场景 xfrc(T, nbodies, 6)。
2) 回放：用“初始状态 + τ 序列(+ 可选工具端扳手)”进行确定性重放，可与记录的 qpos 对齐做误差评估。
3) 其他控制、IK、可视化、usage 维持不变。

用法示例（与旧版一致）：
python fr3_torque_replay.py --xml scene.xml --record-then-replay --onecyc --pert  # 录制一轮并回放（含扰动）
python fr3_torque_replay.py --xml scene.xml --record-then-replay --onecyc        # 录制一轮并回放（无扰动）

体积估算（T≈5736 步）：
- 单条： (7轴)*(τ+q+dq=3) + times ≈ (7*3+1)*T*4B ≈ ~0.5 MB
- 5000 条 ≈ ~2.5 GB（显著低于旧版）
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import os

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

# ---------------------- Slim Recorder ----------------------
class RecorderSlim:
    """ 最小必要记录：tau(T,7), qpos(T,7), qvel(T,7), times(T,), 可选 xfrc_tool(T,6) """
    def __init__(self, action_dim=7, store_tool_wrench=False):
        self.qpos = []
        self.qvel = []
        self.tau  = []
        self.times = []
        self.store_tool_wrench = store_tool_wrench
        self.xfrc_tool = [] if store_tool_wrench else None
        self.action_dim = action_dim

    def record_post_step(self, data, tool_body_id=None):
        # 在 mj_step() 之后调用
        self.qpos.append(data.qpos[:self.action_dim].copy())
        self.qvel.append(data.qvel[:self.action_dim].copy())
        self.tau.append(data.ctrl[:self.action_dim].copy())
        self.times.append(data.time)
        if self.store_tool_wrench and tool_body_id is not None:
            self.xfrc_tool.append(data.xfrc_applied[tool_body_id].copy())

    def to_arrays(self):
        out = {
            "qpos":  np.asarray(self.qpos, dtype=np.float32),
            "qvel":  np.asarray(self.qvel, dtype=np.float32),
            "tau":   np.asarray(self.tau,  dtype=np.float32),
            "times": np.asarray(self.times,dtype=np.float32),
        }
        if self.store_tool_wrench and self.xfrc_tool is not None:
            out["xfrc_tool"] = np.asarray(self.xfrc_tool, dtype=np.float32)  # (T,6)
        return out

# ---------------------- helpers ----------------------
def restore_initial_state(model, data, q0, dq0, set_time=0.0):
    data.qpos[:len(q0)] = q0
    data.qvel[:len(dq0)] = dq0
    data.time = set_time
    mujoco.mj_forward(model, data)

def downsample_zoh(actions, xfrc_tool, keep_percentage: int):
    if keep_percentage >= 100:
        return actions, xfrc_tool
    n = actions.shape[0]
    k = max(2, int(np.ceil(n * keep_percentage / 100.0)))
    keep_idx = np.linspace(0, n-1, k, dtype=int)
    # ZOH
    actions_zoh = actions.copy()
    last = 0
    for j in range(1, len(keep_idx)):
        actions_zoh[keep_idx[j-1]:keep_idx[j]] = actions[keep_idx[j-1]]
        last = keep_idx[j]
    actions_zoh[last:] = actions[keep_idx[-1]]

    xfrc_zoh = None
    if xfrc_tool is not None:
        xfrc_zoh = xfrc_tool.copy()
        last = 0
        for j in range(1, len(keep_idx)):
            xfrc_zoh[keep_idx[j-1]:keep_idx[j]] = xfrc_tool[keep_idx[j-1]]
            last = keep_idx[j]
        xfrc_zoh[last:] = xfrc_tool[keep_idx[-1]]
    return actions_zoh, xfrc_zoh

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("FR3 perfect torque record & replay (slim)")
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--dt", type=float, default=0.0, help="controller dt; 0=>use model.opt.timestep")
    parser.add_argument("--onecyc", action="store_true", help="only run one cycle of drawing after warmup")
    parser.add_argument("--record-then-replay", action="store_true", help="record one run then replay exactly")
    parser.add_argument("--hide-viewer", action="store_true")
    parser.add_argument("--pert", action="store_true", help="enable and record external perturbations (tool body only)")
    parser.add_argument("--percentage", type=int, default=100, choices=range(1,101),
                        help="downsample percentage for replay (ZOH)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store-path", type=str, default=None,
                        help="Path to save recorded trajectory data (.npz)")
    parser.add_argument("--replay-path", type=str, default=None,
                        help="Path to load trajectory data for replay (.npz), skips recording.")
    # trajectory params
    parser.add_argument("--x-plane", type=float, default=0.40)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--ay", type=float, default=None)
    parser.add_argument("--az", type=float, default=None)
    parser.add_argument("--phi", type=float, default=0.0)
    # IK/PD
    parser.add_argument("--kp-cart", type=float, default=8.0)
    parser.add_argument("--kp-ori", type=float, default=6.0)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--kp", type=float, nargs=7, default=[4500,4500,3500,3500,2000,2000,2000])
    parser.add_argument("--kv", type=float, nargs=7, default=[450,450,350,350,200,200,200])
    args = parser.parse_args()

    np.random.seed(args.seed)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Use first 7 actuated joints as FR3
    action_dim = min(7, model.nu)

    # If replay only
    arr = None
    if args.replay_path:
        print(f"== Loading from {args.replay_path} for replay ==")
        arr = np.load(args.replay_path)  # no pickle
    else:
        # set keyframe "home" if exists
        key_id = -1
        for i in range(model.nkey):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
                key_id = i; break
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)

        # Initial joint pose（示例）
        initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981,  2.7076, 0.7105], dtype=np.float64)
        data.qpos[:action_dim] = initial_qpos[:action_dim]
        mujoco.mj_forward(model, data)

        # Controller/trajectory setup
        ctrl_dt = args.dt if args.dt > 0 else model.opt.timestep
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_dyn")
        assert site_id >= 0 and tip_id >= 0 and tool_body_id >= 0

        x0_home = data.site_xpos[site_id].copy()
        cy, cz = x0_home[1], x0_home[2] + 0.15
        x_plane, radius, omega = args.x_plane, args.radius, args.omega
        Ay = args.ay if args.ay is not None else radius
        Az = args.az if args.az is not None else radius * 0.5
        phi = args.phi

        def desired_orientation(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
            z_axis_world = np.array([1.0, 0.0, 0.0])
            up_hint = np.array([0.0, 0.0, 1.0])
            x_axis_world = np.cross(up_hint, z_axis_world)
            if np.linalg.norm(x_axis_world) < 1e-8:
                up_hint = np.array([0.0, 1.0, 0.0])
                x_axis_world = np.cross(up_hint, z_axis_world)
            x_axis_world = _normalize(x_axis_world)
            z_axis_world = _normalize(z_axis_world)
            y_axis_world = np.cross(z_axis_world, x_axis_world)
            return np.column_stack([x_axis_world, y_axis_world, z_axis_world])

        # Random initial EE pose outside bbox
        rand_margin = 0.05
        while True:
            y_offset = np.random.uniform(-Ay - rand_margin, Ay + rand_margin)
            z_offset = np.random.uniform(-Az - rand_margin, Az + rand_margin)
            if abs(y_offset) > Ay or abs(z_offset) > Az:
                break
        x_rand = np.array([x_plane, cy + y_offset, cz + z_offset], dtype=float)

        # Solve IK to land there
        q_ref = data.qpos[:action_dim].copy()
        R_h = desired_orientation(None, None)
        for _ in range(200):
            q_ref = resolved_rate_step_6d(model, data, site_id,
                                          x_des=x_rand, v_des=np.zeros(3),
                                          R_des=R_h, q_ref=q_ref,
                                          kp_cart=10.0, kp_ori=8.0, damping=0.01, dt=ctrl_dt)
            data.qpos[:action_dim] = q_ref
            mujoco.mj_kinematics(model, data)
            if np.linalg.norm(data.site_xpos[site_id] - x_rand) < 1e-3:
                break
        mujoco.mj_forward(model, data)

        # Find closest phase on figure-8
        num_samples = 200
        thetas = np.linspace(0, 2*np.pi/omega, num_samples, endpoint=False)
        traj_y = cy + Ay * np.sin(omega * thetas)
        traj_z = cz + Az * np.sin(2.0 * omega * thetas + phi)
        rand2d = x_rand[1:]
        d2 = (traj_y - rand2d[0])**2 + (traj_z - rand2d[1])**2
        t_traj_start = thetas[np.argmin(d2)]

        # PD gains（把 q_ref 转成 torque）
        Kp = np.array(args.kp, dtype=float)[:action_dim]
        Kv = np.array(args.kv, dtype=float)[:action_dim]

        # Phases
        INITIAL_HOLD = 1.0
        cycle_T = 2*np.pi/omega

        # perturbation settings（只作用于 tool_body_id，存 xfrc_tool(T,6)）
        pert_active = False
        pert_end_time = 0.0
        next_pert_check = INITIAL_HOLD + 5.0
        pert_prob = 0.6

        # clip with actuator ctrlrange
        ctrl_range = model.actuator_ctrlrange.copy()

        # ----------- RECORD RUN -----------
        # 缓存录制开始前的初始状态（关键修复：避免保存末态）
        q0_init  = data.qpos[:action_dim].copy()
        dq0_init = data.qvel[:action_dim].copy()
        
        rec = RecorderSlim(action_dim=action_dim, store_tool_wrench=args.pert)
        time_print_prev = 0.0

        def control_and_record():
            nonlocal q_ref, pert_active, pert_end_time, next_pert_check, time_print_prev
            t = data.time
            # Desired EE target
            if t < INITIAL_HOLD:
                x_des = x_rand
                v_des = np.zeros(3)
                R_des = R_h
            else:
                tau = (t - INITIAL_HOLD) + t_traj_start
                y_des = cy + Ay * np.sin(omega * tau)
                z_des = cz + Az * np.sin(2.0 * omega * tau + phi)
                x_des = np.array([x_plane, y_des, z_des])
                vy = Ay * omega * np.cos(omega * tau)
                vz = 2.0 * Az * omega * np.cos(2.0 * omega * tau + phi)
                v_des = np.array([0.0, vy, vz])
                R_des = desired_orientation(x_des, v_des)

            # Update q_ref by task-space resolved rate
            q_ref = resolved_rate_step_6d(model, data, site_id,
                                          x_des, v_des, R_des, q_ref,
                                          kp_cart=args.kp_cart, kp_ori=args.kp_ori,
                                          damping=args.damping, dt=ctrl_dt)

            # 1) 录制期：确保不留任何外力痕迹（防分叉）
            data.xfrc_applied[:] = 0

            # PD -> torque
            torque_cmd = Kp * (q_ref - data.qpos[:action_dim]) - Kv * data.qvel[:action_dim]
            torque_cmd = np.clip(torque_cmd, ctrl_range[:action_dim, 0], ctrl_range[:action_dim, 1])
            data.ctrl[:] = 0.0
            data.ctrl[:action_dim] = torque_cmd

            # perturbation (仅工具 body；记录为 xfrc_tool(T,6))
            if args.pert:
                if pert_active and t >= pert_end_time:
                    pert_active = False
                    data.xfrc_applied[tool_body_id] = 0
                if (not pert_active) and t >= next_pert_check:
                    next_pert_check = t + 5.0
                    if np.random.rand() < pert_prob and t > INITIAL_HOLD + 0.2:
                        pert_active = True
                        force_mag = np.random.uniform(2.0, 8.0)
                        ang = np.random.uniform(0, 2*np.pi)
                        f = force_mag * np.array([0, np.cos(ang), np.sin(ang)])
                        torque_mag = np.random.uniform(0.1, 0.4)
                        dir3 = np.random.randn(3); dir3 /= np.linalg.norm(dir3)
                        tau_ext = torque_mag * dir3
                        dur = np.random.uniform(0.06, 0.15)
                        pert_end_time = t + dur
                        data.xfrc_applied[tool_body_id, :3] = tau_ext
                        data.xfrc_applied[tool_body_id, 3:6] = f
            else:
                data.xfrc_applied[:] = 0

            # ---- step & record post ----
            mujoco.mj_step(model, data)
            rec.record_post_step(data, tool_body_id=tool_body_id)

            if data.time - time_print_prev >= 0.1:
                tip = data.site_xpos[tip_id].copy()
                print(f"t={data.time:.3f}s, tip=[{tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}]")
                time_print_prev = data.time

        print("== Recording run ==")
        try:
            if args.hide_viewer:
                while True:
                    if args.onecyc and data.time >= (INITIAL_HOLD + cycle_T):
                        break
                    control_and_record()
                    time.sleep(0.0005)
            else:
                try:
                    with viewer.launch_passive(model, data) as v:
                        while v.is_running():
                            if args.onecyc and data.time >= (INITIAL_HOLD + cycle_T):
                                break
                            control_and_record()
                            v.sync()
                            time.sleep(0.0005)
                except Exception:
                    with viewer.launch(model, data) as v:
                        while v.is_running():
                            if args.onecyc and data.time >= (INITIAL_HOLD + cycle_T):
                                break
                            control_and_record()
                            v.sync()
                            time.sleep(0.0005)
        except KeyboardInterrupt:
            print("Recording interrupted.")

        arr_out = rec.to_arrays()
        print(f"[record] steps={len(arr_out['times'])}, t_end={arr_out['times'][-1]:.3f}s")
        
        # 2) 存储：升级 τ 和 times 为 float64（稳住数值，防分叉）
        arr_out["tau"]   = arr_out["tau"].astype(np.float64)
        arr_out["times"] = arr_out["times"].astype(np.float64)

        # 简单绘图（record torque）
        plt.figure(figsize=(12,6))
        for j in range(min(action_dim, arr_out["tau"].shape[1])):
            plt.plot(arr_out["times"], arr_out["tau"][:, j], label=f"J{j+1}")
        plt.legend(); plt.title("Recorded Torques"); plt.xlabel("t (s)"); plt.ylabel("Nm"); plt.grid(True)
        plt.tight_layout(); plt.show()

        if args.store_path:
            print(f"== Storing slim trajectory to {args.store_path} ==")
            # 确保目录存在
            store_dir = os.path.dirname(args.store_path)
            if store_dir and not os.path.exists(store_dir):
                os.makedirs(store_dir, exist_ok=True)
                print(f"Created directory: {store_dir}")
            meta = {
                "q0": q0_init.astype(np.float64),      # 使用录制开始时的初始状态（升级 float64）
                "dq0": dq0_init.astype(np.float64),    # 而不是录制结束时的末态（升级 float64）
                "dt": np.float64(model.opt.timestep),  # 升级 float64
                "action_dim": np.int32(action_dim),
                "has_pert": np.bool_(args.pert),
                "tool_body_id": np.int32(tool_body_id),
            }
            np.savez_compressed(args.store_path, **arr_out, **meta)

    # 如果没有要求回放，直接返回
    if not (args.record_then_replay or args.replay_path):
        return

    # ----------- REPLAY RUN -----------
    if args.replay_path:
        print(f"== Replaying from {args.replay_path} with initial-state restore ==")
    else:
        print("== Replaying with initial-state restore ==")

    # 若是刚录完的 run，使用刚才内存中的数据；否则从 npz 读取
    if arr is None:
        assert args.store_path is not None, "No data to replay; provide --replay-path or --record-then-replay with --store-path"
        arr = np.load(args.store_path)

    # 读取必要字段（5) 回放使用 float64 动作）
    q0   = arr["q0"].astype(np.float64)
    dq0  = arr["dq0"].astype(np.float64)
    dt_s = float(arr["dt"])  # float64 -> Python float
    actions = arr["tau"].astype(np.float64)  # 不降为 float32，保持 float64
    qpos_rec = arr["qpos"].astype(np.float32) if "qpos" in arr.files else None  # qpos 可保持 float32
    times_rec = arr["times"].astype(np.float64) if "times" in arr.files else None  # times 也用 float64
    action_dim = int(arr["action_dim"]) if "action_dim" in arr.files else min(7, model.nu)

    xfrc_tool = arr["xfrc_tool"].astype(np.float32) if "xfrc_tool" in arr.files else None
    has_pert = bool(arr["has_pert"]) if "has_pert" in arr.files else False
    tool_body_id = int(arr["tool_body_id"]) if "tool_body_id" in arr.files else -1

    # 可选降采样（ZOH）
    if 1 <= args.percentage < 100:
        print(f"[replay] ZOH downsample to {args.percentage}% frames")
        actions, xfrc_tool = downsample_zoh(actions, xfrc_tool, args.percentage)

    # 调整仿真 dt 以匹配存储（低保真模拟可叠加 percentage）
    original_timestep = model.opt.timestep
    new_timestep = dt_s
    if 1 <= args.percentage < 100:
        factor = 100.0 / args.percentage
        new_timestep = dt_s * factor
    model.opt.timestep = new_timestep
    print(f"==> Simulation dt: {original_timestep:.6f}s -> {model.opt.timestep:.6f}s")

    data2 = mujoco.MjData(model)
    restore_initial_state(model, data2, q0, dq0, set_time=0.0)

    # 3) 回放前：读取 ctrlrange 以备夹紧
    ctrl_range = model.actuator_ctrlrange.copy()
    
    errors = []
    times2 = []
    replayed_qpos = []
    
    # 4) 误差绊线阈值
    ERR_TRIP = 5e-4  # 可调整，1e-3 更宽松

    def replay_step(i: int):
        # 3) 强制"纯净环境"：零外力（世界系 & 广义力）
        data2.xfrc_applied[:] = 0
        if hasattr(data2, "qfrc_applied"):
            data2.qfrc_applied[:] = 0
        
        # 施加动作（并再次夹紧到 ctrlrange）
        a = actions[i]
        a = np.clip(a, ctrl_range[:action_dim, 0], ctrl_range[:action_dim, 1])
        data2.ctrl[:] = 0.0
        data2.ctrl[:action_dim] = a
        
        # 如果有扰动记录则应用（仅在 --pert 模式）
        if has_pert and xfrc_tool is not None and tool_body_id >= 0:
            data2.xfrc_applied[tool_body_id, :] = xfrc_tool[i]
        
        mujoco.mj_step(model, data2)
        
        # 3) 监控：若任何"外力/接触异常"出现，立刻报警
        xnorm = float(np.max(np.abs(data2.xfrc_applied))) if data2.xfrc_applied.size else 0.0
        qappl = float(np.max(np.abs(getattr(data2, "qfrc_applied", 0.0))))
        ncon  = int(data2.ncon)
        if xnorm > 1e-12 or qappl > 1e-12:
            if not (has_pert and xfrc_tool is not None):  # 仅在非预期扰动时报警
                print(f"[alert] unexpected external force at step {i} t={data2.time:.5f}: "
                      f"max|xfrc_applied|={xnorm:.2e}, max|qfrc_applied|={qappl:.2e}")

        # 误差累计（若有记录的 qpos）
        if qpos_rec is not None and i < len(qpos_rec):
            e = data2.qpos[:action_dim].copy() - qpos_rec[i, :action_dim]
            errors.append(e)
            
            # 4) 运行中误差绊线：一旦越界就打印接触明细
            if np.max(np.abs(e)) > ERR_TRIP:
                print(f"[trip] step={i} t={data2.time:.6f} max|dq|={np.max(np.abs(e)):.3e}")
                print(f"       ncon={ncon}, max|xfrc_applied|={xnorm:.2e}")
                # 打印前几个接触体（geom 名字）
                for k in range(min(ncon, 5)):
                    c = data2.contact[k]
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                    print(f"       contact[{k}]: {g1} <-> {g2}, dist={c.dist:.3e}")
        
        # 稳妥性检查：第一步就校验对齐
        if qpos_rec is not None and i == 0:
            diff0 = np.linalg.norm(data2.qpos[:action_dim] - qpos_rec[0, :action_dim], ord=np.inf)
            if diff0 > 1e-6:
                print(f"[debug] step0 mismatch max|dq|={diff0:.3e}  "
                      f"(check q0/dq0, dt, ctrl alignment, perturbation)")
        
        times2.append(data2.time)
        replayed_qpos.append(data2.qpos[:action_dim].copy())

    if args.hide_viewer:
        for i in range(len(actions)):
            replay_step(i)
    else:
        try:
            with viewer.launch_passive(model, data2) as v:
                i = 0
                while v.is_running() and i < len(actions):
                    replay_step(i)
                    v.sync()
                    time.sleep(0.0004)
                    i += 1
        except Exception:
            with viewer.launch(model, data2) as v:
                i = 0
                while v.is_running() and i < len(actions):
                    replay_step(i)
                    v.sync()
                    time.sleep(0.0004)
                    i += 1

    # 恢复原始 dt
    model.opt.timestep = original_timestep

    if errors:
        errors = np.stack(errors, axis=0)
        mse = float(np.mean(errors**2))
        print(f"[replay] MSE (qpos) = {mse:.6e}")
    else:
        print("[replay] No recorded qpos to compare against (OK if you disabled saving qpos).")

    replayed_qpos = np.stack(replayed_qpos, axis=0)
    times2 = np.array(times2, dtype=np.float32)

    # 绘制回放的力矩
    plt.figure(figsize=(12, 6))
    num_replayed_steps = len(times2)
    for j in range(min(action_dim, actions.shape[1])):
        plt.plot(times2, actions[:num_replayed_steps, j], label=f"J{j+1}")
    plt.legend()
    plt.title("Replayed Torques")
    plt.xlabel("t (s)")
    plt.ylabel("Nm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制回放的关节轨迹（若有 recorded 轨迹则对齐显示）
    plt.figure(figsize=(12, 6))
    for j in range(min(action_dim, replayed_qpos.shape[1])):
        plt.plot(times2, replayed_qpos[:, j], label="Replayed" if j == 0 else "", linestyle='-')
    if qpos_rec is not None:
        t_rec = times_rec[:num_replayed_steps] if times_rec is not None else np.arange(num_replayed_steps) * dt_s
        for j in range(min(action_dim, replayed_qpos.shape[1])):
            plt.plot(t_rec, qpos_rec[:num_replayed_steps, j],
                     label="Recorded" if j == 0 else "", linestyle='--', alpha=0.8)
    plt.legend()
    plt.title("Replayed vs Recorded Joint Trajectory")
    plt.xlabel("t (s)")
    plt.ylabel("rad")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------- helpers -----------
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

if __name__ == "__main__":
    main()
