#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界帧诊断工具 - 检查第一帧和最后一帧的重建质量
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_utils import (
    normalized_times, bspline_design_matrix, ridge_pinv, 
    encode_theta, decode_signal, open_uniform_knots
)
from utils import load_dataset, zscore_fit_stats, zscore_apply, zscore_inv, rmse
from scipy.interpolate import BSpline

# 使用与 quick_tune.py 相同的参数
M_Q_BASE = 64
M_DQ_BASE = 128
M_TAU_BASE = 128

DEGREE_Q = 3
DEGREE_DQ = 5
DEGREE_TAU = 5

LAMBDA_Q = 1e-6
LAMBDA_DQ = 1e-7
LAMBDA_TAU = 1e-7

DATA_PATH = "../../Compress_Test/test_complete.npz"
SAMPLE_INDEX = 0


def diagnose_boundary_reconstruction(q, dq, tau, sample_idx=0, M=64, degree=3, lam=1e-6, signal_name="q"):
    """
    诊断边界帧重建问题
    """
    N, T, J = q.shape
    
    print(f"\n{'='*80}")
    print(f"诊断 {signal_name} - Sample {sample_idx}")
    print(f"{'='*80}")
    print(f"T={T}, M={M}, degree={degree}, lambda={lam:.0e}")
    
    # 归一化
    m, s = zscore_fit_stats(q)
    qz = zscore_apply(q, m, s)
    
    # 归一化时间
    t = normalized_times(T)
    print(f"\n归一化时间范围: [{t[0]:.6f}, {t[-1]:.6f}]")
    print(f"时间步长: {t[1] - t[0]:.6f}")
    
    # 构造样条基
    B = bspline_design_matrix(t, M, degree)
    print(f"\n设计矩阵 B shape: {B.shape}")
    print(f"  B[0, :] 前10个值: {B[0, :10]}")
    print(f"  B[-1, :] 前10个值: {B[-1, :10]}")
    print(f"  B[0, :] sum: {np.sum(B[0, :]):.6f}")
    print(f"  B[-1, :] sum: {np.sum(B[-1, :]):.6f}")
    
    # 检查结向量
    knots = open_uniform_knots(M, degree)
    print(f"\n结向量 (knots) shape: {knots.shape}")
    print(f"  前 {degree+2} 个: {knots[:degree+2]}")
    print(f"  后 {degree+2} 个: {knots[-(degree+2):]}")
    
    # 计算伪逆
    P = ridge_pinv(B, lam)
    print(f"\n伪逆矩阵 P shape: {P.shape}")
    
    # 编码-解码
    i = sample_idx
    theta = encode_theta(P, qz[i])
    qz_hat = decode_signal(B, theta)
    q_rec = zscore_inv(qz_hat[None, :, :], m, s)[0]
    
    # 计算各帧误差
    errors = np.abs(q[i] - q_rec)
    
    # 前10帧和后10帧
    n_show = 10
    print(f"\n前 {n_show} 帧误差统计:")
    for j in range(J):
        mean_err = np.mean(errors[:n_show, j])
        max_err = np.max(errors[:n_show, j])
        print(f"  Joint {j+1}: mean={mean_err:.6f}, max={max_err:.6f}")
    
    print(f"\n后 {n_show} 帧误差统计:")
    for j in range(J):
        mean_err = np.mean(errors[-n_show:, j])
        max_err = np.max(errors[-n_show:, j])
        print(f"  Joint {j+1}: mean={mean_err:.6f}, max={max_err:.6f}")
    
    print(f"\n中间 {n_show} 帧误差统计 (t={T//2} 附近):")
    mid = T // 2
    for j in range(J):
        mean_err = np.mean(errors[mid-n_show//2:mid+n_show//2, j])
        max_err = np.max(errors[mid-n_show//2:mid+n_show//2, j])
        print(f"  Joint {j+1}: mean={mean_err:.6f}, max={max_err:.6f}")
    
    # 第一帧详细检查
    print(f"\n第一帧 (t=0) 详细分析:")
    print(f"  原始值: {q[i, 0, :]}")
    print(f"  重建值: {q_rec[0, :]}")
    print(f"  误差:   {errors[0, :]}")
    print(f"  相对误差: {errors[0, :] / (np.abs(q[i, 0, :]) + 1e-8)}")
    
    print(f"\n最后一帧 (t={T-1}) 详细分析:")
    print(f"  原始值: {q[i, -1, :]}")
    print(f"  重建值: {q_rec[-1, :]}")
    print(f"  误差:   {errors[-1, :]}")
    print(f"  相对误差: {errors[-1, :] / (np.abs(q[i, -1, :]) + 1e-8)}")
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 子图1: 沿时间的最大误差
    ax1 = axes[0]
    max_errors = np.max(errors, axis=1)
    ax1.plot(max_errors, linewidth=1.5)
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='First frame')
    ax1.axvline(x=T-1, color='b', linestyle='--', linewidth=2, alpha=0.7, label='Last frame')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Max Error across joints', fontsize=12)
    ax1.set_title(f'{signal_name}: Maximum Reconstruction Error over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 标注前10帧和后10帧区域
    ax1.axvspan(0, n_show, alpha=0.2, color='red', label=f'First {n_show} frames')
    ax1.axvspan(T-n_show, T-1, alpha=0.2, color='blue', label=f'Last {n_show} frames')
    ax1.legend(fontsize=11)
    
    # 子图2: 每个关节的前后边界误差对比
    ax2 = axes[1]
    x = np.arange(J)
    width = 0.35
    
    first_frame_errors = errors[0, :]
    last_frame_errors = errors[-1, :]
    middle_frame_errors = errors[T//2, :]
    
    ax2.bar(x - width, first_frame_errors, width, label='First frame (t=0)', alpha=0.8, color='red')
    ax2.bar(x, last_frame_errors, width, label=f'Last frame (t={T-1})', alpha=0.8, color='blue')
    ax2.bar(x + width, middle_frame_errors, width, label=f'Middle frame (t={T//2})', alpha=0.8, color='green')
    
    ax2.set_xlabel('Joint', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'{signal_name}: Boundary vs Middle Frame Errors', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'J{i+1}' for i in range(J)])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'boundary_diagnosis_{signal_name}.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 已保存诊断图: boundary_diagnosis_{signal_name}.png")
    plt.close()
    
    return errors


def main():
    print("="*80)
    print("B样条边界帧诊断工具")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    dataset = load_dataset(DATA_PATH)
    q = dataset['q']
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    print(f"数据形状: N={N}, T={T}, J={J}")
    
    if SAMPLE_INDEX >= N:
        print(f"错误: Sample index {SAMPLE_INDEX} 超出范围 {N}")
        return
    
    # 诊断 Position (q)
    errors_q = diagnose_boundary_reconstruction(
        q, dq, tau, SAMPLE_INDEX, M_Q_BASE, DEGREE_Q, LAMBDA_Q, "Position"
    )
    
    # 诊断 Velocity (dq)
    errors_dq = diagnose_boundary_reconstruction(
        dq, dq, tau, SAMPLE_INDEX, M_DQ_BASE, DEGREE_DQ, LAMBDA_DQ, "Velocity"
    )
    
    # 诊断 Torque (tau)
    errors_tau = diagnose_boundary_reconstruction(
        tau, dq, tau, SAMPLE_INDEX, M_TAU_BASE, DEGREE_TAU, LAMBDA_TAU, "Torque"
    )
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)
    print("\n生成的诊断图:")
    print("  - boundary_diagnosis_Position.png")
    print("  - boundary_diagnosis_Velocity.png")
    print("  - boundary_diagnosis_Torque.png")
    print("\n请查看图表分析边界帧重建质量。")


if __name__ == '__main__':
    main()

