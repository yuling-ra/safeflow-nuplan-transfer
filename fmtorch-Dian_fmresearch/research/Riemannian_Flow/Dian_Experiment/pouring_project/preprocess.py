import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

# ==============================================================================
# SE(3) / SO(3) Math Utilities
# ==============================================================================

def project_to_SO3(R_mat):
    """通过SVD将3x3矩阵投影到SO(3)"""
    U, _, Vt = np.linalg.svd(R_mat)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt

def project_to_SO3_batch(R_batch):
    """批量投影到SO(3)"""
    return np.array([project_to_SO3(R) for R in R_batch])

def _skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def se3_log(T):
    """SE(3)的对数映射: T -> ξ ∈ R^6 [ω(3), v(3)]"""
    R_mat = T[:3, :3]
    t = T[:3, 3]

    omega = R.from_matrix(R_mat).as_rotvec()
    theta = np.linalg.norm(omega)
    I = np.eye(3)

    if theta < 1e-6:
        omega_sk = _skew(omega)
        V_inv = I - 0.5 * omega_sk + (1.0 / 12.0) * (omega_sk @ omega_sk)
    else:
        omega_sk = _skew(omega)
        half_theta = 0.5 * theta
        cot_half = np.cos(half_theta) / np.sin(half_theta)
        V_inv = I - 0.5 * omega_sk + ((1.0 / (theta * theta)) * (1 - (theta * cot_half) / 2.0)) * (omega_sk @ omega_sk)

    v = V_inv @ t
    return np.concatenate([omega, v])

def se3_exp(xi):
    """SE(3)的指数映射: ξ ∈ R^6 -> T ∈ SE(3)"""
    omega = xi[:3]
    v = xi[3:]

    theta = np.linalg.norm(omega)
    I = np.eye(3)

    if theta < 1e-6:
        R_mat = R.from_rotvec(omega).as_matrix()
        omega_sk = _skew(omega)
        V = I + 0.5 * omega_sk + (1.0 / 6.0) * (omega_sk @ omega_sk)
    else:
        R_mat = R.from_rotvec(omega).as_matrix()
        omega_sk = _skew(omega)
        A = (1 - np.cos(theta)) / (theta * theta)
        B = (theta - np.sin(theta)) / (theta * theta * theta)
        V = I + A * omega_sk + B * (omega_sk @ omega_sk)

    t = V @ v

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================

def load_pouring_data(pkl_path):
    """加载原始倒水轨迹数据"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"数据加载自: {pkl_path}")
    print(f"  - 轨迹形状: {data['traj'].shape}")
    return data['traj'], data

def align_to_first_frame(T_seq_raw):
    """将轨迹对齐到第一帧"""
    R_seq_clean = project_to_SO3_batch(T_seq_raw[:, :3, :3])
    t_seq = T_seq_raw[:, :3, 3]

    T_0 = np.eye(4)
    T_0[:3, :3] = R_seq_clean[0]
    T_0[:3, 3] = t_seq[0]
    T_0_inv = np.linalg.inv(T_0)

    T_seq_aligned = np.zeros_like(T_seq_raw)
    for k in range(T_seq_raw.shape[0]):
        T_k = np.eye(4)
        T_k[:3, :3] = R_seq_clean[k]
        T_k[:3, 3] = t_seq[k]
        T_seq_aligned[k] = T_0_inv @ T_k
    
    return T_seq_aligned, T_0

def preprocess_absolute_pose(T_seq_aligned, artifacts_dir: Path):
    """方法1: R9+t3 绝对位姿表示"""
    print("方法1: R9+t3 预处理...")
    num_frames = T_seq_aligned.shape[0]
    
    R_seq_aligned = T_seq_aligned[:, :3, :3]
    t_seq_aligned = T_seq_aligned[:, :3, 3]

    r9_seq = R_seq_aligned.reshape(num_frames, 9)
    X_abs = np.concatenate([r9_seq, t_seq_aligned], axis=1)

    mu_R9 = r9_seq.mean(axis=0)
    sigma_R9 = r9_seq.std(axis=0) + 1e-8
    mu_t3 = t_seq_aligned.mean(axis=0)
    sigma_t3 = t_seq_aligned.std(axis=0) + 1e-8

    Z_abs = np.zeros_like(X_abs)
    Z_abs[:, :9] = (r9_seq - mu_R9) / sigma_R9
    Z_abs[:, 9:12] = (t_seq_aligned - mu_t3) / sigma_t3

    whitening_stats_abs = {
        'mu_R9': mu_R9.tolist(), 'sigma_R9': sigma_R9.tolist(),
        'mu_t3': mu_t3.tolist(), 'sigma_t3': sigma_t3.tolist()
    }
    with open(artifacts_dir / '04_whiten_abs.json', 'w') as f:
        json.dump(whitening_stats_abs, f, indent=2)

    print(f"  - Z_abs shape: {Z_abs.shape}")
    return Z_abs, whitening_stats_abs

def reconstruct_absolute_pose(Z_abs, stats, T_0):
    """从R9+t3表示重构SE(3)轨迹"""
    num_frames = Z_abs.shape[0]
    
    sigma_R9 = np.array(stats['sigma_R9'])
    mu_R9 = np.array(stats['mu_R9'])
    sigma_t3 = np.array(stats['sigma_t3'])
    mu_t3 = np.array(stats['mu_t3'])

    r9_recon = Z_abs[:, :9] * sigma_R9 + mu_R9
    t_recon = Z_abs[:, 9:12] * sigma_t3 + mu_t3

    R_recon_aligned = project_to_SO3_batch(r9_recon.reshape(num_frames, 3, 3))

    T_recon_abs = np.zeros((num_frames, 4, 4))
    for k in range(num_frames):
        T_aligned = np.eye(4)
        T_aligned[:3, :3] = R_recon_aligned[k]
        T_aligned[:3, 3] = t_recon[k]
        T_recon_abs[k] = T_0 @ T_aligned
        
    return T_recon_abs

def compute_delta_xi(T_seq_aligned):
    """仅计算 ΔSE(3) 的对数参数序列（不做白化），返回 (N-1, 6)"""
    num_frames = T_seq_aligned.shape[0]
    delta_xi_seq = [se3_log(np.linalg.inv(T_seq_aligned[k-1]) @ T_seq_aligned[k]) for k in range(1, num_frames)]
    return np.array(delta_xi_seq)

def reconstruct_incremental_pose(Z_delta, stats, T_0, T_seq_aligned_start):
    """从ΔSE(3)表示重构SE(3)轨迹"""
    num_frames = Z_delta.shape[0] + 1
    
    sigma_omega = np.array(stats['sigma_omega'])
    mu_omega = np.array(stats['mu_omega'])
    sigma_v = np.array(stats['sigma_v'])
    mu_v = np.array(stats['mu_v'])

    omega_recon = Z_delta[:, :3] * sigma_omega + mu_omega
    v_recon = Z_delta[:, 3:] * sigma_v + mu_v
    xi_recon = np.concatenate([omega_recon, v_recon], axis=1)

    T_recon_delta = np.zeros((num_frames, 4, 4))
    T_recon_delta[0] = T_seq_aligned_start

    for k in range(1, num_frames):
        delta_T_hat = se3_exp(xi_recon[k-1])
        T_recon_delta[k] = T_recon_delta[k-1] @ delta_T_hat
        T_recon_delta[k, :3, :3] = project_to_SO3(T_recon_delta[k, :3, :3])

    T_recon_delta_abs = np.array([T_0 @ T for T in T_recon_delta])
    return T_recon_delta_abs
