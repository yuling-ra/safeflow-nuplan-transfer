#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panda 7-DoF — CasADi + IPOPT OCP for 3-lap figure-8 (YZ plane) with Quintic Hermite Spline

- Path-following OCP (phase theta), minimize normal error to figure-8 curve
- Full rigid-body dynamics: M(q) ddq + h(q,dq) = tau (enforced at collocation points)
- Quintic Hermite spline (C² continuous) parameterization with K knots
- Variables: q, dq, ddq at each knot (7×K×3 = ~1.3k for K=64)
- Collocation points enforce dynamics & path tracking
- tau eliminated (computed via M(q)a + h(q,v))
- Exports dense 500Hz dataset and plots; MeshCat visualization of optimized q(t)

This script should run under casadi conda environment.
"""

import os, time, numpy as np
import argparse
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

# ---------- MeshCat ----------
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

def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
    thetas = np.linspace(0.0, 2.0*np.pi/omega, K, endpoint=False)
    y  = cy + Ay*np.sin(omega*thetas)
    z  = cz + Az*np.sin(2.0*omega*thetas + phi)
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
def hermite5_casadi(q0, v0, a0, q1, v1, a1, h, s):
    """
    Quintic Hermite on s in [0,1], segment duration = h.
    q(s)  = H00*q0 + H10*h*v0 + H20*h^2*a0 + H01*q1 + H11*h*v1 + H21*h^2*a1
    dq/dt = (1/h) * d/ds(q(s))
    ddq/dt2 = (1/h^2) * d2/ds2(q(s))
    Returns: (q, v, a) all as CasADi expressions
    """
    # basis for q
    H00 = 1 - 10*s**3 + 15*s**4 - 6*s**5
    H10 =      s     -  6*s**3 +  8*s**4 - 3*s**5
    H20 = 0.5*s**2   - 1.5*s**3 + 1.5*s**4 - 0.5*s**5
    H01 = 10*s**3 - 15*s**4 + 6*s**5
    H11 = -4*s**3 + 7*s**4 - 3*s**5
    H21 = 0.5*s**3 -  s**4 + 0.5*s**5

    q = H00*q0 + H10*(h*v0) + H20*(h*h*a0) + H01*q1 + H11*(h*v1) + H21*(h*h*a1)

    # derivatives wrt s
    dH00 = -30*s**2 + 60*s**3 - 30*s**4
    dH10 = 1 - 18*s**2 + 32*s**3 - 15*s**4
    dH20 = s - 4.5*s**2 + 6*s**3 - 2.5*s**4
    dH01 = 30*s**2 - 60*s**3 + 30*s**4
    dH11 = -12*s**2 + 28*s**3 - 15*s**4
    dH21 = 1.5*s**2 - 4*s**3 + 2.5*s**4

    dqds = dH00*q0 + dH10*(h*v0) + dH20*(h*h*a0) + dH01*q1 + dH11*(h*v1) + dH21*(h*h*a1)
    v = dqds / h

    # second derivatives wrt s
    ddH00 = -60*s + 180*s**2 - 120*s**3
    ddH10 = -36*s + 96*s**2 - 60*s**3
    ddH20 = 1 - 9*s + 18*s**2 - 10*s**3
    ddH01 = 60*s - 180*s**2 + 120*s**3
    ddH11 = -24*s + 84*s**2 - 60*s**3
    ddH21 = 3*s - 12*s**2 + 10*s**3

    d2qds2 = ddH00*q0 + ddH10*(h*v0) + ddH20*(h*h*a0) + ddH01*q1 + ddH11*(h*v1) + ddH21*(h*h*a1)
    a = d2qds2 / (h*h)

    return q, v, a

def hermite5_numpy(q0, v0, a0, q1, v1, a1, h, s):
    """
    Numpy version of quintic Hermite for dense trajectory generation
    """
    # basis for q
    H00 = 1 - 10*s**3 + 15*s**4 - 6*s**5
    H10 =      s     -  6*s**3 +  8*s**4 - 3*s**5
    H20 = 0.5*s**2   - 1.5*s**3 + 1.5*s**4 - 0.5*s**5
    H01 = 10*s**3 - 15*s**4 + 6*s**5
    H11 = -4*s**3 + 7*s**4 - 3*s**5
    H21 = 0.5*s**3 -  s**4 + 0.5*s**5

    q = H00*q0 + H10*(h*v0) + H20*(h*h*a0) + H01*q1 + H11*(h*v1) + H21*(h*h*a1)

    # derivatives wrt s
    dH00 = -30*s**2 + 60*s**3 - 30*s**4
    dH10 = 1 - 18*s**2 + 32*s**3 - 15*s**4
    dH20 = s - 4.5*s**2 + 6*s**3 - 2.5*s**4
    dH01 = 30*s**2 - 60*s**3 + 30*s**4
    dH11 = -12*s**2 + 28*s**3 - 15*s**4
    dH21 = 1.5*s**2 - 4*s**3 + 2.5*s**4

    dqds = dH00*q0 + dH10*(h*v0) + dH20*(h*h*a0) + dH01*q1 + dH11*(h*v1) + dH21*(h*h*a1)
    v = dqds / h

    # second derivatives wrt s
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
    parser = argparse.ArgumentParser(description="Panda FR3 OCP (CasADi+Spline) for 3-lap figure-8")
    parser.add_argument("--speed", type=float, default=2.5, help="MeshCat slowdown factor")
    parser.add_argument("--K", type=int, default=64, help="Number of spline knots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="eight_ocp_spline_dataset.npz")
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

    T_cycle = 2.0*np.pi/omega
    laps = 3.0
    T_total = laps * T_cycle
    print(f"[ocp] T_total={T_total:.2f}s")

    # --- IK landing to nearest phase for good q0 ---
    pin.forwardKinematics(model, data, q0_num); pin.updateFramePlacements(model, data)
    p0 = data.oMf[ee_fid].translation.copy()
    theta0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, omega, phi)
    # build initial ee target for landing
    y0 = cy + Ay*np.sin(omega*theta0_guess)
    z0 = cz + Az*np.sin(2*omega*theta0_guess + phi)
    vy0= Ay*omega*np.cos(omega*theta0_guess)
    vz0= 2*Az*omega*np.cos(2*omega*theta0_guess + phi)
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
        ddq_cmd = 60.0*e + 2.0*np.sqrt(60.0)* (qdot_ref - dq_num)
        tau = pin.rnea(model, data, q_num, dq_num, ddq_cmd)
        M = pin.crba(model, data, q_num); h = pin.nle(model, data, q_num, dq_num)
        ddq = np.linalg.solve(M, tau - h)
        dq_num += ddq*0.002
        q_num  = pin.integrate(model, q_num, dq_num*0.002)
        if pos_err < 1e-4: break
    pin.forwardKinematics(model, data, q_num); pin.updateFramePlacements(model, data)
    p0_land = data.oMf[ee_fid].translation.copy()
    theta0 = nearest_phase(p0_land[1], p0_land[2], cy, cz, Ay, Az, omega, phi)
    print(f"[init] landed. |p-x*|~{np.linalg.norm(p0_land-x_des0):.3e}, theta0={theta0:.3f}")

    # --- CasADi functions ---
    fk_fun, M_fun, h_fun = build_casadi_functions(model, ee_fid)

    # --- NLP variables (spline knots) ---
    K = args.K                      # knots
    t_knots = np.linspace(0, T_total, K)
    h_seg   = float(T_total/(K-1))  # 等间隔
    print(f"[ocp] K={K} knots, h_seg={h_seg:.4f}s")

    nq, nv = model.nq, model.nv
    
    # 结点变量：q_k, dq_k, ddq_k
    Qk = ca.SX.sym("Qk", nq, K)
    Vk = ca.SX.sym("Vk", nv, K)
    Ak = ca.SX.sym("Ak", nv, K)

    # theta 也在结点上定义（保持单调）
    TH = ca.SX.sym("TH", 1, K)

    # 拼装决策向量
    w = ca.vertcat(
        ca.reshape(Qk, -1, 1),
        ca.reshape(Vk, -1, 1),
        ca.reshape(Ak, -1, 1),
        ca.reshape(TH, -1, 1),
    )

    g  = []
    gl = []
    gu = []

    # --- bounds/limits ---
    qmin = np.array([-2.9, -1.76, -2.9, -3.07, -2.9,  0.9, -2.9])
    qmax = np.array([ 2.9,  1.76,  2.9, -0.05,  2.9,  3.75,  2.9])
    vmax = 2.5*np.ones(nv)
    amax = 10.0*np.ones(nv)
    taumax = np.array([85,85,85,85,20,20,20], dtype=float)

    lbw = []
    ubw = []
    # Q bounds
    for k in range(K): lbw += list(qmin); ubw += list(qmax)
    # V bounds
    for k in range(K): lbw += list(-vmax); ubw += list(vmax)
    # A bounds
    for k in range(K): lbw += list(-amax); ubw += list(amax)
    # TH bounds（只给一个很宽的下上界，单调由约束控制）
    for k in range(K): lbw += [-1e3];      ubw += [1e9]

    # --- initial conditions ---
    # q0, v0, th0 硬约束
    g.append(Qk[:,0] - q_num);        gl += [0.0]*nq; gu += [0.0]*nq
    g.append(Vk[:,0] - dq_num);       gl += [0.0]*nv; gu += [0.0]*nv
    g.append(TH[0]   - theta0);       gl += [0.0];    gu += [0.0]

    # --- terminal theta: +3*(2*pi) ---
    g.append(TH[-1] - TH[0] - 3.0*(2.0*np.pi)); gl += [0.0]; gu += [0.0]

    # --- monotonic theta ---
    for k in range(K-1):
        dth = TH[k+1] - TH[k]
        g.append(dth); gl += [0.0]; gu += [1e9]

    # --- cost & collocation constraints on each segment ---
    J = 0.0
    w_e     = 1e5     # normal tracking
    w_x     = 5e3     # x-plane soft
    w_tau   = 5.0     # torque L2
    w_dv    = 1.0     # 结点速度平滑（可调）
    w_da    = 5.0     # 结点加速度平滑（可调）

    # 2-pt Gauss collocation in [0,1]
    colloc_s = [0.211324865405187, 0.788675134594813]
    
    print(f"[ocp] Using {len(colloc_s)} collocation points per segment")

    for k in range(K-1):
        q0 = Qk[:,k];  v0 = Vk[:,k];  a0 = Ak[:,k]
        q1 = Qk[:,k+1];v1 = Vk[:,k+1];a1 = Ak[:,k+1]

        # 结点处平滑正则（可选）
        if k > 0:
            J += w_dv * ca.dot(Vk[:,k] - Vk[:,k-1], Vk[:,k] - Vk[:,k-1])
            J += w_da * ca.dot(Ak[:,k] - Ak[:,k-1], Ak[:,k] - Ak[:,k-1])

        for s in colloc_s:
            # 样条评估
            q_s, v_s, a_s = hermite5_casadi(q0, v0, a0, q1, v1, a1, h_seg, ca.DM(s))
            # theta 用线性插值就够（也可仿照上面换成 quintic）
            th_s = (1.0 - s)*TH[k] + s*TH[k+1]

            # 末端位置
            pk = fk_fun(q_s)  # (3,)

            # 目标曲线误差（YZ 平面法向投影）
            x_err = pk[0] - x_plane
            yk, zk = fig8_yz(th_s, cy, cz, Ay, Az, phi)
            e_yz = ca.vertcat(pk[1]-yk, pk[2]-zk)

            # 切向 & 法向投影
            T = fig8_tangent(th_s, Ay, Az, phi)
            Pn = ca.DM.eye(2) - T@T.T
            e_n = Pn @ e_yz

            # 动力学扭矩表达式（不是变量）
            Mk = M_fun(q_s)
            hk = h_fun(q_s, v_s)
            tau_expr = Mk @ a_s + hk

            # 代价
            J += w_e * ca.dot(e_n, e_n)
            J += w_x * (x_err*x_err)
            J += w_tau * ca.dot(tau_expr, tau_expr)

            # 扭矩盒约束：-taumax <= tau_expr <= taumax
            g.append(tau_expr);      gl += list(-taumax); gu += list(taumax)

    # Terminal regularization
    J += 1e-3 * (ca.dot(Vk[:,-1], Vk[:,-1]) + ca.dot(Ak[:,-1], Ak[:,-1]))

    # --- NLP build & solve ---
    print(f"[ocp] Building NLP with {w.shape[0]} decision variables and {len(g)} constraint blocks...")
    nlp = {"x": w, "f": J, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-4,
        "ipopt.linear_solver": "mumps",
        "print_time": False,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # --- 初始猜测 ---
    # 用 IK 生成粗略轨迹，然后下采样到 K 结点
    print("[init] Generating initial guess via IK...")
    N_guess = 300
    th_grid_dense = np.linspace(theta0, theta0 + 3.0*(2.0*np.pi), N_guess)
    q_guess_dense = np.zeros((nq, N_guess))
    q_guess_dense[:,0] = q_num.copy()
    last_q = q_num.copy()
    dt_guess = T_total/(N_guess-1)
    
    for k in range(1, N_guess):
        thk = th_grid_dense[k]
        yk = cy + Ay*np.sin(thk); zk = cz + Az*np.sin(2*thk + phi)
        dyk= Ay*np.cos(thk); dzk = 2*Az*np.cos(2*thk + phi)
        Tvec = np.array([0., dyk, dzk])
        if np.linalg.norm(Tvec) < 1e-8:
            ey_vec = np.array([0.,1.,0.])
        else:
            ey_vec = Tvec/np.linalg.norm(Tvec)
        ex_vec = np.cross(ey_vec, np.array([1.,0.,0.]))
        if np.linalg.norm(ex_vec) > 1e-8:
            ex_vec /= np.linalg.norm(ex_vec)
        Rdes = np.column_stack([ex_vec, ey_vec, np.array([1.,0.,0.])])
        xdes = np.array([x_plane, yk, zk]); vdes = np.zeros(3)

        qk = last_q.copy()
        # few IK smoothing steps
        for _ in range(8):
            qk, qdot_ref, *_ = resolved_rate_step6(model, data, ee_fid, qk, xdes, vdes, Rdes, dt=dt_guess)
        q_guess_dense[:,k] = qk
        last_q = qk

    # 下采样到 K 结点
    sample_idx = np.round(np.linspace(0, N_guess-1, K)).astype(int)
    Q0 = q_guess_dense[:, sample_idx]
    V0 = np.zeros((nv, K))
    V0[:, :-1] = np.diff(Q0, axis=1)/h_seg
    A0 = np.zeros((nv, K))
    A0[:, :-1] = np.diff(V0, axis=1)/h_seg
    TH0 = np.linspace(theta0, theta0+3.0*(2*np.pi), K)

    x0 = np.concatenate([
        Q0.reshape(-1), V0.reshape(-1), A0.reshape(-1), TH0.reshape(-1)
    ])
    
    print(f"[init] Initial guess prepared: {len(x0)} variables")

    # --- solve ---
    print("[solve] IPOPT start...")
    t_solve_start = time.perf_counter()
    sol = solver(x0=x0, lbg=np.array(gl), ubg=np.array(gu), lbx=np.array(lbw), ubx=np.array(ubw))
    t_solve_end = time.perf_counter()
    wopt = np.array(sol["x"]).squeeze()
    print(f"[solve] done in {t_solve_end - t_solve_start:.2f}s")

    # --- 解包 & 生成致密轨迹用于可视化/导出 ---
    idx = 0
    def take(n):
        nonlocal idx
        out = wopt[idx:idx+n]; idx += n; return out

    Qk_opt = take(nq*K).reshape(nq, K)
    Vk_opt = take(nv*K).reshape(nv, K)
    Ak_opt = take(nv*K).reshape(nv, K)
    TH_opt = take(1*K).reshape(1,  K).flatten()

    # 细采样到 500Hz（对齐 PD 版本）
    dt_dense = 0.002  # 500Hz
    T_dense = int(T_total / dt_dense) + 1
    t_dense = np.linspace(0.0, T_total, T_dense)
    q_dense = np.zeros((T_dense, nq))
    dq_dense= np.zeros((T_dense, nv))
    ddq_dense=np.zeros((T_dense, nv))
    th_dense = np.zeros(T_dense)

    print(f"[dense] Generating {T_dense} samples at 500Hz...")
    for i, t in enumerate(t_dense):
        # 定位到段 k, s
        u = min(max(t / h_seg, 0.0), K-1-1e-9)   # 防越界
        k = int(np.floor(u))
        if k >= K-1:
            k = K-2
        s = u - k
        
        q0 = Qk_opt[:,k]; v0 = Vk_opt[:,k]; a0 = Ak_opt[:,k]
        q1 = Qk_opt[:,k+1]; v1 = Vk_opt[:,k+1]; a1 = Ak_opt[:,k+1]
        q_s, v_s, a_s = hermite5_numpy(q0, v0, a0, q1, v1, a1, h_seg, s)
        q_dense[i,:]   = q_s
        dq_dense[i,:]  = v_s
        ddq_dense[i,:] = a_s
        th_dense[i] = (1.0 - s)*TH_opt[k] + s*TH_opt[k+1]

    # Compute end-effector positions and torques
    print("[dense] Computing end-effector positions and torques...")
    ee_opt  = np.zeros((T_dense, 3))
    ee_des = np.zeros((T_dense, 3))
    tau_dense = np.zeros((T_dense, nv))
    
    for i in range(T_dense):
        # FK for end-effector
        pin.forwardKinematics(model, data, q_dense[i,:])
        pin.updateFramePlacements(model, data)
        ee_opt[i,:] = data.oMf[ee_fid].translation.copy()
        
        # Desired end-effector
        yk = cy + Ay*np.sin(th_dense[i])
        zk = cz + Az*np.sin(2*th_dense[i] + phi)
        ee_des[i,:] = np.array([x_plane, yk, zk])
        
        # Compute torque via inverse dynamics
        M = pin.crba(model, data, q_dense[i,:])
        h = pin.nle(model, data, q_dense[i,:], dq_dense[i,:])
        tau_dense[i,:] = M @ ddq_dense[i,:] + h

    # --- save dataset ---
    np.savez(args.save,
             t=t_dense, q=q_dense, dq=dq_dense, ddq=ddq_dense, tau=tau_dense,
             ee=ee_opt, ee_des=ee_des, theta=th_dense, x_plane=x_plane,
             fig8_params=np.array([cy,cz,Ay,Az,phi]),
             knots_t=t_knots, knots_q=Qk_opt.T, knots_dq=Vk_opt.T, 
             knots_ddq=Ak_opt.T, knots_theta=TH_opt)
    print(f"[save] npz saved -> {args.save}")
    print(f"       Dense trajectory: {T_dense} samples at 500Hz")
    print(f"       Knot data: {K} knots")

    # ---------- MeshCat visualization ----------
    full_viz = None
    if HAS_MESHCAT and PinMeshcatVis is not None:
        full_viz = PinMeshcatVis(model, collision_model, visual_model)
        try:
            full_viz.initViewer(open=True)
        except Exception:
            full_viz.initViewer(open=False)
        full_viz.loadViewerModel()
        full_viz.display(q_dense[0,:])
        print("[viz] MeshCat ON")
        print("URL:", full_viz.viewer.url())

        slowdown = args.speed
        last = time.perf_counter()
        skip = max(1, int(0.01 / dt_dense))  # update at ~100Hz for viz
        for i in range(0, T_dense, skip):
            # keep roughly realtime*slowdown
            now = time.perf_counter()
            elapsed = now - last
            to_sleep = dt_dense*skip*slowdown - elapsed
            if to_sleep > 0: time.sleep(to_sleep)
            full_viz.display(q_dense[i,:])
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
            plt.plot(t_dense, tau_dense[:,j], label=f'J{j+1}')
        plt.grid(True); plt.legend(ncol=4)
        plt.title('Optimized Torques (via spline)'); plt.xlabel('t (s)'); plt.ylabel('Nm')
        plt.tight_layout()

        # Tracking error
        tracking_error = np.linalg.norm(ee_opt - ee_des, axis=1)
        plt.figure(figsize=(12,4))
        plt.plot(t_dense, tracking_error*1000)  # mm
        plt.grid(True); plt.xlabel('t (s)'); plt.ylabel('Tracking error (mm)')
        plt.title(f'End-Effector Tracking Error (mean: {np.mean(tracking_error)*1000:.2f} mm, max: {np.max(tracking_error)*1000:.2f} mm)')
        plt.tight_layout()

        plt.show()
    except Exception as e:
        print(f"[plot] Failed to generate plots: {e}")

if __name__ == "__main__":
    main()

