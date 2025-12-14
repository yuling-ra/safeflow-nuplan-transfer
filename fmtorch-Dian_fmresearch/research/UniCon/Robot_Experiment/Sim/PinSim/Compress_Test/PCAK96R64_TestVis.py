#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 PCA K=96 r=64 压缩-重建效果
随机选取一条轨迹，对比原始 vs 压缩重建后的 q, dq, tau 和末端执行器轨迹
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv,
    apply_group_weights, remove_group_weights,
    dct_truncate, idct_reconstruct, flatten_coeffs, unflatten_coeffs,
    group_rmse, weighted_rmse
)

# Pinocchio for forward kinematics
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[warn] pinocchio not found, will skip end-effector trajectory visualization")


def compute_ee_trajectory(q_traj, urdf_path, package_dirs):
    """
    计算末端执行器轨迹
    q_traj: (T, 7) 关节角度序列
    返回: (T, 3) 末端位置序列
    """
    if not HAS_PINOCCHIO:
        return None
    
    try:
        model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs)
        data = model.createData()
        EE_NAME = "panda_tool_tip"
        ee_fid = model.getFrameId(EE_NAME)
        
        T = q_traj.shape[0]
        ee_pos = np.zeros((T, 3))
        
        for i in range(T):
            pin.forwardKinematics(model, data, q_traj[i])
            pin.updateFramePlacements(model, data)
            ee_pos[i] = data.oMf[ee_fid].translation.copy()
        
        return ee_pos
    except Exception as e:
        print(f"[warn] Failed to compute EE trajectory: {e}")
        return None


def main():
    # ============================================================
    # 配置参数
    # ============================================================
    DATA_PATH = "test_complete.npz"
    K = 96
    R = 64
    
    # 权重设置（与训练时一致）
    WQ = 1.0
    WDQ = 0.25
    WTAU = 0.1
    
    SEED = 42
    np.random.seed(SEED)
    
    print("="*70)
    print(f"PCA 压缩-重建可视化测试")
    print(f"配置: K={K}, r={R}")
    print(f"权重: wq={WQ}, wdq={WDQ}, wtau={WTAU}")
    print("="*70)
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("\n[1/6] 加载数据...")
    dataset = load_dataset(DATA_PATH)
    X = dataset['X']  # (N, T, 21)
    N, T, F = X.shape
    print(f"  数据形状: N={N}, T={T}, F={F}")
    print(f"  原始数据大小: {X.nbytes / 1e6:.2f} MB")
    
    # ============================================================
    # 2. 预处理：Z-score + 加权
    # ============================================================
    print("\n[2/6] 预处理（Z-score + 加权）...")
    mean, std = zscore_fit_stats(X)
    Xz = zscore_apply(X, mean, std)
    Xzw = apply_group_weights(Xz, WQ, WDQ, WTAU)
    print(f"  均值范围: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  标准差范围: [{std.min():.3f}, {std.max():.3f}]")
    
    # ============================================================
    # 3. DCT 截断
    # ============================================================
    print(f"\n[3/6] DCT 截断到 K={K}...")
    C = dct_truncate(Xzw, K)  # (N, K, 21)
    print(f"  DCT 系数形状: {C.shape}")
    print(f"  DCT 后数据大小: {C.nbytes / 1e6:.2f} MB")
    
    # ============================================================
    # 4. PCA 压缩与重建
    # ============================================================
    print(f"\n[4/6] PCA 降维到 r={R}...")
    X_flat = flatten_coeffs(C)  # (N, K*21)
    print(f"  展平形状: {X_flat.shape}")
    
    # 训练 PCA
    pca = PCA(n_components=R, svd_solver='randomized', random_state=SEED)
    Z = pca.fit_transform(X_flat)  # (N, R)
    print(f"  压缩后形状: {Z.shape}")
    print(f"  压缩后数据大小: {Z.nbytes / 1e6:.2f} MB")
    print(f"  解释方差比: {pca.explained_variance_ratio_.sum():.6f}")
    print(f"  压缩率: {X.nbytes / Z.nbytes:.1f}x")
    
    # PCA 重建
    print("\n[5/6] PCA 重建...")
    X_flat_recon = pca.inverse_transform(Z)
    C_recon = unflatten_coeffs(X_flat_recon, K, F)
    
    # IDCT 重建
    Xzw_recon = idct_reconstruct(C_recon, T)
    
    # 反加权 & 反 Z-score
    Xz_recon = remove_group_weights(Xzw_recon, WQ, WDQ, WTAU)
    X_recon = zscore_inv(Xz_recon, mean, std)
    
    print(f"  重建形状: {X_recon.shape}")
    
    # 计算总体误差
    metrics = group_rmse(X, X_recon)
    wrmse = weighted_rmse(X, X_recon, WQ, WDQ, WTAU)
    print(f"\n  重建误差:")
    print(f"    WRMSE: {wrmse:.6f}")
    print(f"    RMSE q:   {metrics['rmse_q']:.6f} rad")
    print(f"    RMSE dq:  {metrics['rmse_dq']:.6f} rad/s")
    print(f"    RMSE tau: {metrics['rmse_tau']:.6f} Nm")
    
    # ============================================================
    # 6. 随机选择一条轨迹可视化
    # ============================================================
    print("\n[6/6] 可视化随机选择的轨迹...")
    traj_idx = np.random.randint(0, N)
    print(f"  选择轨迹索引: {traj_idx}")
    
    # 提取该轨迹的原始和重建数据
    q_orig = X[traj_idx, :, 0:7]      # (T, 7)
    dq_orig = X[traj_idx, :, 7:14]    # (T, 7)
    tau_orig = X[traj_idx, :, 14:21]  # (T, 7)
    
    q_recon = X_recon[traj_idx, :, 0:7]
    dq_recon = X_recon[traj_idx, :, 7:14]
    tau_recon = X_recon[traj_idx, :, 14:21]
    
    # 计算该轨迹的误差
    err_q = np.sqrt(np.mean((q_orig - q_recon)**2))
    err_dq = np.sqrt(np.mean((dq_orig - dq_recon)**2))
    err_tau = np.sqrt(np.mean((tau_orig - tau_recon)**2))
    
    print(f"  该轨迹的重建误差:")
    print(f"    q:   {err_q:.6f} rad")
    print(f"    dq:  {err_dq:.6f} rad/s")
    print(f"    tau: {err_tau:.6f} Nm")
    
    # 时间轴（假设 dt=0.002）
    dt = 0.002
    t = np.arange(T) * dt
    
    # ============================================================
    # 绘图
    # ============================================================
    print("\n绘制对比图...")
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(18, 12))
    
    # 颜色方案
    colors_orig = plt.cm.tab10(np.arange(7))
    colors_recon = colors_orig * 0.6  # 稍微暗一些
    
    # -------------------- 子图 1: 关节位置 q --------------------
    ax1 = plt.subplot(2, 2, 1)
    for j in range(7):
        ax1.plot(t, q_orig[:, j], '-', color=colors_orig[j], 
                 linewidth=1.5, label=f'J{j+1} orig', alpha=0.8)
        ax1.plot(t, q_recon[:, j], '--', color=colors_recon[j], 
                 linewidth=1.2, label=f'J{j+1} recon', alpha=0.7)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Joint Position (rad)', fontsize=12)
    ax1.set_title(f'Joint Positions (q) | RMSE={err_q:.4f} rad', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=8, loc='upper right')
    
    # -------------------- 子图 2: 关节速度 dq --------------------
    ax2 = plt.subplot(2, 2, 2)
    for j in range(7):
        ax2.plot(t, dq_orig[:, j], '-', color=colors_orig[j], 
                 linewidth=1.5, label=f'J{j+1} orig', alpha=0.8)
        ax2.plot(t, dq_recon[:, j], '--', color=colors_recon[j], 
                 linewidth=1.2, label=f'J{j+1} recon', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Joint Velocity (rad/s)', fontsize=12)
    ax2.set_title(f'Joint Velocities (dq) | RMSE={err_dq:.4f} rad/s', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8, loc='upper right')
    
    # -------------------- 子图 3: 关节力矩 tau --------------------
    ax3 = plt.subplot(2, 2, 3)
    for j in range(7):
        ax3.plot(t, tau_orig[:, j], '-', color=colors_orig[j], 
                 linewidth=1.5, label=f'J{j+1} orig', alpha=0.8)
        ax3.plot(t, tau_recon[:, j], '--', color=colors_recon[j], 
                 linewidth=1.2, label=f'J{j+1} recon', alpha=0.7)
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Joint Torque (Nm)', fontsize=12)
    ax3.set_title(f'Joint Torques (tau) | RMSE={err_tau:.4f} Nm', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=2, fontsize=8, loc='upper right')
    
    # -------------------- 子图 4: 末端执行器轨迹 --------------------
    ax4 = plt.subplot(2, 2, 4)
    
    if HAS_PINOCCHIO:
        # 尝试加载 URDF
        urdf_path = os.path.join(os.path.dirname(__file__), 
                                 "../panda_description/urdf/panda_stick.urdf")
        package_dirs = [os.path.join(os.path.dirname(__file__), "../panda_description")]
        
        if os.path.exists(urdf_path):
            print("  计算末端执行器轨迹...")
            ee_orig = compute_ee_trajectory(q_orig, urdf_path, package_dirs)
            ee_recon = compute_ee_trajectory(q_recon, urdf_path, package_dirs)
            
            if ee_orig is not None and ee_recon is not None:
                # 绘制 YZ 平面的 8 字轨迹
                ax4.plot(ee_orig[:, 1], ee_orig[:, 2], '-', 
                         color='blue', linewidth=2, label='Original', alpha=0.8)
                ax4.plot(ee_recon[:, 1], ee_recon[:, 2], '--', 
                         color='red', linewidth=1.5, label='Reconstructed', alpha=0.7)
                
                # 标记起点和终点
                ax4.plot(ee_orig[0, 1], ee_orig[0, 2], 'go', 
                         markersize=10, label='Start', zorder=5)
                ax4.plot(ee_orig[-1, 1], ee_orig[-1, 2], 'rs', 
                         markersize=10, label='End', zorder=5)
                
                # 计算 EE 误差
                ee_err = np.sqrt(np.mean(np.sum((ee_orig - ee_recon)**2, axis=1)))
                
                ax4.set_xlabel('Y Position (m)', fontsize=12)
                ax4.set_ylabel('Z Position (m)', fontsize=12)
                ax4.set_title(f'End-Effector Trajectory (YZ plane) | RMSE={ee_err*1000:.2f} mm', 
                              fontsize=13, fontweight='bold')
                ax4.axis('equal')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=10, loc='best')
                
                print(f"  末端执行器误差: {ee_err*1000:.3f} mm")
            else:
                ax4.text(0.5, 0.5, 'EE trajectory\ncomputation failed', 
                         ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        else:
            ax4.text(0.5, 0.5, f'URDF not found:\n{urdf_path}', 
                     ha='center', va='center', transform=ax4.transAxes, fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'Pinocchio not available\nInstall with:\npip install pin', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # ============================================================
    # 总标题
    # ============================================================
    fig.suptitle(f'PCA K={K} r={R} Compression-Reconstruction | Trajectory #{traj_idx} | '
                 f'WRMSE={wrmse:.4f} | Compression Ratio={X.nbytes/Z.nbytes:.1f}x',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    output_path = f'PCA_K{K}_R{R}_trajectory_{traj_idx}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("可视化完成！")
    print("="*70)


if __name__ == '__main__':
    main()

