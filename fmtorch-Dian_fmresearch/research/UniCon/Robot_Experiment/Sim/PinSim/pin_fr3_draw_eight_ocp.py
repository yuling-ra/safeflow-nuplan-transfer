#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panda 7-DoF — CasADi + IPOPT OCP for 1.2-lap figure-8 (YZ plane) with Quintic Hermite Spline

- Path-following OCP (phase theta), minimize normal error to figure-8 curve
- Full rigid-body dynamics: M(q) ddq + h(q,dq) = tau (enforced at collocation points)
- Quintic Hermite spline (C² continuous) parameterization with K=128 knots
- Variables: q, dq, ddq at each knot (7×128×3 ≈ 2688 decision variables)
- Collocation points enforce dynamics & path tracking
- tau eliminated (computed via M(q)a + h(q,v))
- Trajectory: 1.2 laps in 13.612s (T_total = 6806 × 0.002s)
- Boundary: q0,dq0 fixed from IK landing; theta_N - theta_0 = 1.2*2*pi; dtheta >= 0
- Phase uniformity: soft constraint on dtheta to prevent phase accumulation
- Exports dense dataset (6806 samples @ 500Hz) and plots; MeshCat visualization

This script should run under casadi conda environment.
"""

import os, time, numpy as np
import argparse
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

# ---------- MeshCat (same as your code) ----------
PinMeshcatVis = None
try:
    from pinocchio.visualize import MeshcatVisualizer as PinMeshcatVis
except Exception:
    pass

try:
    import meshcat
    import meshcat.geometry as g
    HAS_MESHCAT = True
except Exception:
    HAS_MESHCAT = False
    print("[warn] meshcat not found; running headless.")

# ---------- helpers (reused) ----------
def so3_log(R):
    tr = np.trace(R); c = np.clip((tr-1.0)*0.5, -1.0, 1.0); th = np.arccos(c)
    if th < 1e-8:
        return 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    w = (1.0/(2.0*np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return th*w

def resolved_rate_step6(model, data, frame_id, q,
                        x_des, v_des, R_des,
                        kp_pos=16.0, kp_ori=4.0,
                        base_damp=0.14,
                        qdot_limit=2.2,
                        rate_alpha=0.6,
                        dt=0.002):
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[frame_id]; p = oMf.translation; R = oMf.rotation

    v_cmd = v_des + kp_pos*(x_des - p)
    w_cmd = kp_ori * so3_log(R_des @ R.T)

    J6 = pin.computeFrameJacobian(model, data, q, frame_id,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    s = np.linalg.svd(J6, compute_uv=False)
    smin = float(s[-1]) if s.size else 0.0
    speed_scale = np.clip(smin / 0.02, 0.25, 1.0)
    v_cmd *= speed_scale; w_cmd *= speed_scale

    y = np.hstack([v_cmd, w_cmd])
    lam = max(base_damp, 0.05)
    JJt_reg = J6 @ J6.T + (lam**2)*np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt_reg, y)

    lim = np.full(model.nv, qdot_limit, dtype=float)
    scale = np.max(np.abs(qdot)/lim) if lim.size else 0.0
    if scale > 1.0: qdot /= scale

    q_next = pin.integrate(model, q, rate_alpha * qdot * dt)
    return q_next, qdot, smin, np.linalg.norm(x_des - p)

def nearest_phase(y0, z0, cy, cz, Ay, Az, phi, K=1200):
    """
    搜索相位 θ ∈ [0, 2π) 使 (y(θ), z(θ)) 离 (y0, z0) 最近
    返回: 相位 θ (弧度)
    """
    thetas = np.linspace(0.0, 2.0*np.pi, K, endpoint=False)
    y  = cy + Ay*np.sin(thetas)
    z  = cz + Az*np.sin(2.0*thetas + phi)
    i  = np.argmin((y - y0)**2 + (z - z0)**2)
    return float(thetas[i])

# ---------- build casadi FK/dynamics functions ----------
def build_casadi_functions(model, ee_fid):
    """Return CasADi Functions: fk(q)->p(3), M(q)->(nv,nv), h(q,v)->(nv,) for Pinocchio 3.x casadi binding."""
    import casadi as ca
    import pinocchio.casadi as cpin

    # 将 double 型 Model 转成 CasADi 型 Model，并创建对应 Data（关键！）
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", model.nq)
    v = ca.SX.sym("v", model.nv)

    # 质矩阵 M(q)
    # 你这版的签名是 crba(model, data, q, convention=...)
    M = cpin.crba(cmodel, cdata, q)

    # 非线性项 h(q,v)
    if hasattr(cpin, "nonLinearEffects"):
        h = cpin.nonLinearEffects(cmodel, cdata, q, v)
    elif hasattr(cpin, "nle"):
        h = cpin.nle(cmodel, cdata, q, v)
    else:
        raise RuntimeError("pinocchio.casadi has neither nonLinearEffects nor nle")

    # 末端位置：用 FK + updateFramePlacements，然后读 cdata.oMf
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    p = cdata.oMf[ee_fid].translation  # SX(3,)

    fk_fun = ca.Function("fk", [q], [p])
    M_fun  = ca.Function("M" , [q], [M])
    h_fun  = ca.Function("h" , [q, v], [h])
    return fk_fun, M_fun, h_fun



# ---------- Quintic Hermite basis (position/velocity/acceleration) ----------
def hermite5(q0, v0, a0, q1, v1, a1, h, s):
    """
    Quintic Hermite on s in [0,1], segment duration = h.
    Returns: (q, v, a) all as CasADi expressions
    """
    # q(s)
    H00 = 1 - 10*s**3 + 15*s**4 - 6*s**5
    H10 =      s     -  6*s**3 +  8*s**4 - 3*s**5
    H20 = 0.5*s**2   - 1.5*s**3 + 1.5*s**4 - 0.5*s**5
    H01 = 10*s**3 - 15*s**4 + 6*s**5
    H11 = -4*s**3 + 7*s**4 - 3*s**5
    H21 = 0.5*s**3 -  s**4 + 0.5*s**5
    q = H00*q0 + H10*(h*v0) + H20*(h*h*a0) + H01*q1 + H11*(h*v1) + H21*(h*h*a1)

    # dq/dt
    dH00 = -30*s**2 + 60*s**3 - 30*s**4
    dH10 = 1 - 18*s**2 + 32*s**3 - 15*s**4
    dH20 = s - 4.5*s**2 + 6*s**3 - 2.5*s**4
    dH01 = 30*s**2 - 60*s**3 + 30*s**4
    dH11 = -12*s**2 + 28*s**3 - 15*s**4
    dH21 = 1.5*s**2 - 4*s**3 + 2.5*s**4
    dqds = dH00*q0 + dH10*(h*v0) + dH20*(h*h*a0) + dH01*q1 + dH11*(h*v1) + dH21*(h*h*a1)
    v = dqds / h

    # ddq/dt^2
    ddH00 = -60*s + 180*s**2 - 120*s**3
    ddH10 = -36*s + 96*s**2 - 60*s**3
    ddH20 = 1 - 9*s + 18*s**2 - 10*s**3
    ddH01 = 60*s - 180*s**2 + 120*s**3
    ddH11 = -24*s + 84*s**2 - 60*s**3
    ddH21 = 3*s - 12*s**2 + 10*s**3
    d2qds2 = ddH00*q0 + ddH10*(h*v0) + ddH20*(h*h*a0) + ddH01*q1 + ddH11*(h*v1) + ddH21*(h*h*a1)
    a = d2qds2 / (h*h)
    return q, v, a

# ---------- figure-8 geometry ----------
def fig8_yz(theta, cy, cz, Ay, Az, phi):
    """returns (y,z) CasADi expressions from theta (can be numpy if theta float)"""
    y = cy + Ay * ca.sin(theta)
    z = cz + Az * ca.sin(2*theta + phi)
    return y, z

def fig8_tangent(theta, Ay, Az, phi, eps=1e-8):
    """returns tangent T in YZ (2x1), normalized"""
    dy = Ay * ca.cos(theta)
    dz = 2*Az * ca.cos(2*theta + phi)
    T  = ca.vertcat(dy, dz)
    n  = ca.sqrt(dy*dy + dz*dz) + eps
    return T / n

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Panda FR3 OCP (CasADi+Spline) for 1.2-lap figure-8")
    parser.add_argument("--speed", type=float, default=2.5, help="MeshCat slowdown factor")
    parser.add_argument("--N", type=int, default=300, help="IK initial guess discretization nodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save", type=str, default="eight_ocp_dataset.npz", help="Output npz filename")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # === Load model / frames ===
    URDF_PATH = os.path.join(os.path.dirname(__file__),
                             "panda_description", "urdf", "panda_stick.urdf")
    assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"
    PACKAGE_DIRS = [os.path.join(os.path.dirname(__file__), "panda_description")]

    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, PACKAGE_DIRS)
    data  = model.createData()
    print(f"[pin] nq={model.nq}, nv={model.nv}")
    assert model.nv == 7

    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    assert ee_fid < len(model.frames), f"EE frame not found: {EE_NAME}"

    # Initial state (same as your script)
    q0_num = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105], dtype=float)
    dq0_num= np.zeros(model.nv)

    # --- workspace & figure-8 params ---
    pin.forwardKinematics(model, data, q0_num); pin.updateFramePlacements(model, data)
    p0_ref = data.oMf[ee_fid].translation.copy()
    x_plane = p0_ref[0] + 0.25
    cy, cz = p0_ref[1], p0_ref[2] + 0.12
    Ay, Az = 0.10, 0.05
    omega, phi = 0.6, 0.0

    # --- Trajectory duration and laps ---
    # 固定时长和圈数 (可通过命令行参数覆盖)
    laps = 1.2  # 绕 1.2 圈
    T_total = 13.612  # 固定总时长 (s)
    T_cycle = T_total / laps  # 反推单圈时长
    N = args.N
    dt = T_total/(N-1)
    print(f"[ocp] laps={laps:.1f}, T_total={T_total:.3f}s, T_cycle={T_cycle:.3f}s")

    # --- IK landing to nearest phase for good q0 ---
    pin.forwardKinematics(model, data, q0_num); pin.updateFramePlacements(model, data)
    p0 = data.oMf[ee_fid].translation.copy()
    # 找到最近的相位 θ（弧度）
    theta0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, phi)
    # 构建着陆目标（纯相位）
    y0 = float(cy + Ay*np.sin(theta0_guess))
    z0 = float(cz + Az*np.sin(2*theta0_guess + phi))
    
    # 期望速度：用平均相位速度
    theta_dot_avg = (laps * 2.0*np.pi) / T_total
    vy0 = Ay * np.cos(theta0_guess) * theta_dot_avg
    vz0 = 2 * Az * np.cos(2*theta0_guess + phi) * theta_dot_avg
    x_des0 = np.array([x_plane, y0, z0])
    v_des0 = np.array([0.0, vy0, vz0])
    # horizontal ori with tool-z along world x
    ez = np.array([1.,0.,0.]); up=np.array([0.,0.,1.])
    ey = np.cross(up, ez); ey/=np.linalg.norm(ey); ex=np.cross(ey, ez)
    R_des0 = np.column_stack([ex,ey,ez])

    q_num = q0_num.copy(); dq_num = dq0_num.copy()
    for _ in range(300):
        q_ref, qdot_ref, _, pos_err = resolved_rate_step6(
            model, data, ee_fid, q_num, x_des0, v_des0, R_des0)
        e = pin.difference(model, q_num, q_ref)
        ddq_cmd = 60.0*e + 2.0*np.sqrt(60.0)* (qdot_ref - dq_num)  # mild
        tau = pin.rnea(model, data, q_num, dq_num, ddq_cmd)
        M = pin.crba(model, data, q_num); h = pin.nle(model, data, q_num, dq_num)
        ddq = np.linalg.solve(M, tau - h)
        dq_num += ddq*0.002
        q_num  = pin.integrate(model, q_num, dq_num*0.002)
        if pos_err < 1e-4: break
    pin.forwardKinematics(model, data, q_num); pin.updateFramePlacements(model, data)
    p0_land = data.oMf[ee_fid].translation.copy()
    theta0 = nearest_phase(p0_land[1], p0_land[2], cy, cz, Ay, Az, phi)
    print(f"[init] landed. |p-x*|~{np.linalg.norm(p0_land-x_des0):.3e}, theta0={theta0:.3f} rad")

    # --- CasADi functions ---
    fk_fun, M_fun, h_fun = build_casadi_functions(model, ee_fid)

    # --- Spline NLP variables (K=128 knots) ---
    nq, nv = model.nq, model.nv
    K = 128  # 使用 128 个结点以获得更高精度
    h_seg = T_total/(K-1)
    print(f"[ocp] Spline mode: K={K} knots, h_seg={h_seg:.4f}s")

    # knots: q_k, dq_k, ddq_k 以及相位 TH_k
    Qk = ca.SX.sym("Qk", nq, K)
    Vk = ca.SX.sym("Vk", nv, K)
    Ak = ca.SX.sym("Ak", nv, K)
    TH = ca.SX.sym("TH", 1,  K)

    w  = ca.vertcat(ca.reshape(Qk,-1,1), ca.reshape(Vk,-1,1), ca.reshape(Ak,-1,1), ca.reshape(TH,-1,1))
    g, gl, gu = [], [], []

    # --- bounds/limits ---
    qmin = np.array([-2.9, -1.76, -2.9, -3.07, -2.9,  0.9, -2.9])
    qmax = np.array([ 2.9,  1.76,  2.9, -0.05,  2.9,  3.75,  2.9])
    vmax = 2.5*np.ones(nv)
    amax = 10.0*np.ones(nv)
    taumax = np.array([85,85,85,85,20,20,20], dtype=float)

    # bounds
    lbw, ubw = [], []
    for _ in range(K): lbw += list(qmin);  ubw += list(qmax)      # Qk
    for _ in range(K): lbw += list(-vmax); ubw += list(vmax)      # Vk
    for _ in range(K): lbw += list(-amax); ubw += list(amax)      # Ak
    for _ in range(K): lbw += [-1e3];      ubw += [1e9]           # TH

    # initial/terminal & monotonic theta
    g.append(Qk[:,0]-q_num);           gl += [0.0]*nq; gu += [0.0]*nq
    g.append(Vk[:,0]-dq0_num);         gl += [0.0]*nv; gu += [0.0]*nv
    g.append(TH[0]-theta0);            gl += [0.0];    gu += [0.0]
    # 终点相位：theta_N - theta_0 = laps * 2*pi (1.2 圈)
    g.append(TH[-1]-TH[0]-laps*(2*np.pi)); gl += [0.0]; gu += [0.0]
    for k in range(K-1):
        dth = TH[k+1]-TH[k]
        g.append(dth); gl += [0.0]; gu += [1e9]

    # cost
    J = 0.0
    w_e, w_x, w_tau = 1e5, 5e3, 5.0
    w_dv, w_da = 1.0, 5.0  # 结点平滑
    w_dth = 1.0            # 相位均匀推进 (防止堆积)
    avg_dth = laps*(2.0*np.pi)/(K-1)  # 期望的平均相位增量 (1.2 圈)

    # 2 collocation points per segment (Gauss-Legendre)
    colloc_s = [0.211324865405187, 0.788675134594813]
    print(f"[ocp] Using {len(colloc_s)} collocation points per segment")

    for k in range(K-1):
        q0, v0, a0 = Qk[:,k],   Vk[:,k],   Ak[:,k]
        q1, v1, a1 = Qk[:,k+1], Vk[:,k+1], Ak[:,k+1]
        
        # 结点平滑正则
        if k>0:
            J += w_dv*ca.dot(Vk[:,k]-Vk[:,k-1], Vk[:,k]-Vk[:,k-1])
            J += w_da*ca.dot(Ak[:,k]-Ak[:,k-1], Ak[:,k]-Ak[:,k-1])
        
        # 相位均匀推进软约束 (防止在某些段堆积)
        dth = TH[k+1] - TH[k]
        J += w_dth * (dth - avg_dth)**2

        for s in colloc_s:
            qs, vs, as_ = hermite5(q0,v0,a0, q1,v1,a1, h_seg, ca.DM(s))
            ths = (1.0-s)*TH[k] + s*TH[k+1]

            pk = fk_fun(qs)                      # EE pos
            x_err = pk[0]-x_plane
            yref, zref = fig8_yz(ths, cy, cz, Ay, Az, phi)
            e_yz = ca.vertcat(pk[1]-yref, pk[2]-zref)

            Tdir = fig8_tangent(ths, Ay, Az, phi)
            Pn = ca.DM.eye(2) - Tdir@Tdir.T      # 法向投影
            e_n = Pn @ e_yz

            Mk = M_fun(qs)
            hk = h_fun(qs, vs)
            tau_expr = Mk @ as_ + hk             # 扭矩表达式

            J += w_e*ca.dot(e_n,e_n) + w_x*(x_err*x_err) + w_tau*ca.dot(tau_expr,tau_expr)
            g.append(tau_expr); gl += list(-taumax); gu += list(taumax)  # 扭矩盒约束

    # terminal regularization
    J += 1e-3 * (ca.dot(Vk[:,-1], Vk[:,-1]) + ca.dot(Ak[:,-1], Ak[:,-1]))

    # build & solve
    print(f"[ocp] Building NLP with {w.shape[0]} variables and {len(g)} constraint blocks...")
    nlp = {"x": w, "f": J, "g": ca.vertcat(*g)}
    opts = {"ipopt.print_level":5,"ipopt.max_iter":2000,"ipopt.tol":1e-4,"ipopt.linear_solver":"mumps","print_time":False}
    solver = ca.nlpsol("solver","ipopt", nlp, opts)

    # initial guess（把 euler 的 q_guess 下采样成 K 结点）
    print("[init] Generating initial guess via IK...")
    th_grid = np.linspace(theta0, theta0 + laps*(2.0*np.pi), N)
    q_guess = np.zeros((nq, N)); v_guess = np.zeros((nv,N))
    q_guess[:,0] = q_num.copy()
    last_q = q_num.copy(); last_dq = np.zeros(nv)
    for k in range(1, N):
        thk = th_grid[k]
        yk = cy + Ay*np.sin(thk); zk = cz + Az*np.sin(2*thk + phi)
        dyk= Ay*np.cos(thk); dzk = 2*Az*np.cos(2*thk + phi)
        Tvec = np.array([0., dyk, dzk])
        ey_vec = Tvec/np.linalg.norm(Tvec) if np.linalg.norm(Tvec)>1e-8 else np.array([0.,1.,0.])
        ex_vec = np.cross(ey_vec, np.array([1.,0.,0.]))
        if np.linalg.norm(ex_vec) > 1e-8:
            ex_vec /= np.linalg.norm(ex_vec)
        Rdes = np.column_stack([ex_vec,ey_vec,np.array([1.,0.,0.])])
        xdes = np.array([x_plane, yk, zk]); vdes = np.zeros(3)
        qk = last_q.copy()
        for _ in range(8):
            qk, _, *_ = resolved_rate_step6(model, data, ee_fid, qk, xdes, vdes, Rdes, dt=dt)
        q_guess[:,k] = qk; last_q = qk
    v_guess[:, :-1] = np.diff(q_guess, axis=1)/dt

    # 下采样到 K knots
    idxK = np.round(np.linspace(0, N-1, K)).astype(int)
    Q0 = q_guess[:, idxK]
    V0 = np.zeros_like(Q0); V0[:, :-1] = np.diff(Q0, axis=1)/h_seg
    A0 = np.zeros_like(Q0); A0[:, :-1] = np.diff(V0, axis=1)/h_seg
    TH0 = np.linspace(theta0, theta0 + laps*(2*np.pi), K)

    x0 = np.concatenate([Q0.reshape(-1), V0.reshape(-1), A0.reshape(-1), TH0.reshape(-1)])
    print(f"[init] Initial guess prepared: {len(x0)} variables")

    # --- solve ---
    print("[solve] IPOPT start...")
    t_start = time.perf_counter()
    sol = solver(x0=x0, lbg=np.array(gl), ubg=np.array(gu), lbx=np.array(lbw), ubx=np.array(ubw))
    t_end = time.perf_counter()
    wopt = np.array(sol["x"]).squeeze()
    print(f"[solve] done in {t_end-t_start:.2f}s")

    # 解包
    ofs = 0
    def take(n):
        nonlocal ofs
        out = wopt[ofs:ofs+n]; ofs += n; return out
    Qk_opt = take(nq*K).reshape(nq, K)
    Vk_opt = take(nv*K).reshape(nv, K)
    Ak_opt = take(nv*K).reshape(nv, K)
    TH_opt = take(1*K).reshape(1,  K).flatten()

    # 生成致密轨迹用于可视化/保存（固定 6806 样本，dt=0.002s，对应 500Hz）
    T_dense = 6806  # 固定样本数
    t_dense = np.linspace(0.0, T_total, T_dense)
    dt_sample = T_total / (T_dense - 1)
    fs = 1.0 / dt_sample  # 实际采样率
    q_opt = np.zeros((nq, T_dense))
    v_opt = np.zeros((nv, T_dense))
    a_opt = np.zeros((nv, T_dense))
    th_opt_dense = np.zeros(T_dense)
    print(f"[dense] Generating {T_dense} samples, dt={dt_sample:.4f}s ({fs:.1f} Hz, T={T_total:.3f}s) via spline interpolation...")
    for i, t in enumerate(t_dense):
        u = min(max(t / h_seg, 0.0), K-1-1e-12)
        k = int(np.floor(u)); s = u - k
        if k >= K-1:
            k = K-2; s = 1.0
        q0,v0,a0 = Qk_opt[:,k],   Vk_opt[:,k],   Ak_opt[:,k]
        q1,v1,a1 = Qk_opt[:,k+1], Vk_opt[:,k+1], Ak_opt[:,k+1]
        qs,vs,as_ = hermite5(q0,v0,a0, q1,v1,a1, h_seg, ca.DM(s))
        q_opt[:,i] = np.array(qs).squeeze()
        v_opt[:,i] = np.array(vs).squeeze()
        a_opt[:,i] = np.array(as_).squeeze()
        th_opt_dense[i] = (1.0-s)*TH_opt[k] + s*TH_opt[k+1]

    # 计算扭矩和末端位置
    t_grid = t_dense
    th_opt = th_opt_dense
    tau_opt = np.zeros_like(a_opt)
    ee_opt  = np.zeros((T_dense,3))
    ee_des = np.zeros((T_dense,3))
    print("[dense] Computing torques and end-effector positions...")
    for i in range(T_dense):
        # 扭矩
        M = pin.crba(model, data, q_opt[:,i])
        h = pin.nle(model, data, q_opt[:,i], v_opt[:,i])
        tau_opt[:,i] = M @ a_opt[:,i] + h
        # 末端位置
        pin.forwardKinematics(model, data, q_opt[:,i])
        pin.updateFramePlacements(model, data)
        ee_opt[i,:] = data.oMf[ee_fid].translation.copy()
        yk = cy + Ay*np.sin(th_opt[i])
        zk = cz + Az*np.sin(2*th_opt[i]+phi)
        ee_des[i,:] = np.array([x_plane, yk, zk])

    # --- save dataset ---
    np.savez(args.save,
             t=t_grid, q=q_opt.T, dq=v_opt.T, ddq=a_opt.T, tau=tau_opt.T,
             ee=ee_opt, ee_des=ee_des, theta=th_opt, x_plane=x_plane,
             fig8_params=np.array([cy,cz,Ay,Az,phi]),
             knots_q=Qk_opt.T, knots_dq=Vk_opt.T, knots_ddq=Ak_opt.T, knots_theta=TH_opt,
             sample_rate=fs)
    print(f"[save] npz saved -> {args.save}")
    print(f"       Dense trajectory: {T_dense} samples @ {fs} Hz ({T_total:.2f}s)")
    print(f"       Knot data: {K} knots @ {h_seg:.4f}s intervals")

    # ---------- MeshCat visualization ----------
    full_viz = None
    if HAS_MESHCAT and PinMeshcatVis is not None:
        full_viz = PinMeshcatVis(model, collision_model, visual_model)
        try:
            full_viz.initViewer(open=True)
        except Exception:
            full_viz.initViewer(open=False)
        full_viz.loadViewerModel()
        full_viz.display(q_opt[:,0])
        print("[viz] MeshCat ON")
        print("URL:", full_viz.viewer.url())

        slowdown = args.speed
        last = time.perf_counter()
        skip = max(1, int(0.01 * T_dense / T_total))  # ~100Hz viz update
        for i in range(0, T_dense, skip):
            # keep roughly realtime*slowdown
            now = time.perf_counter()
            elapsed = now - last
            dt_viz = T_total / T_dense * skip
            to_sleep = dt_viz*slowdown - elapsed
            if to_sleep > 0: time.sleep(to_sleep)
            full_viz.display(q_opt[:,i])
            last = time.perf_counter()

    # ---------- plots ----------
    try:
        import matplotlib.pyplot as plt
        
        # YZ path
        plt.figure(figsize=(6,6))
        plt.plot(ee_des[:,1], ee_des[:,2], '--', label='desired YZ', alpha=0.7, linewidth=2)
        plt.plot(ee_opt[:,1], ee_opt[:,2], '-', label='optimized YZ', linewidth=1.5)
        plt.axis('equal'); plt.grid(True); plt.xlabel('Y (m)'); plt.ylabel('Z (m)')
        plt.title('End-Effector path on YZ plane (OCP Spline)')
        plt.legend(); plt.tight_layout()

        # Joint torques
        plt.figure(figsize=(12,5))
        for j in range(nv):
            plt.plot(t_grid, tau_opt[j,:], label=f'J{j+1}')
        plt.grid(True); plt.legend(ncol=4)
        plt.title('Optimized Torques (via Hermite spline)'); plt.xlabel('t (s)'); plt.ylabel('Nm')
        plt.tight_layout()

        # Tracking error
        tracking_error = np.linalg.norm(ee_opt - ee_des, axis=1)
        plt.figure(figsize=(12,4))
        plt.plot(t_grid, tracking_error*1000)  # mm
        plt.grid(True); plt.xlabel('t (s)'); plt.ylabel('Tracking error (mm)')
        plt.title(f'End-Effector Tracking Error (mean: {np.mean(tracking_error)*1000:.2f} mm, max: {np.max(tracking_error)*1000:.2f} mm)')
        plt.tight_layout()

        plt.show()
    except Exception as e:
        print(f"[plot] Failed: {e}")

if __name__ == "__main__":
    main()
