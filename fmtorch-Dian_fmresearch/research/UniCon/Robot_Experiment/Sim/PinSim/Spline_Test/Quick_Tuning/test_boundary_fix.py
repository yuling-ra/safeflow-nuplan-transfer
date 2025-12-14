#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试边界权重修复方案
对比标准方法 vs 加权边界方法
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_utils import (
    normalized_times, bspline_design_matrix, ridge_pinv as ridge_pinv_standard,
    encode_theta, decode_signal
)
from spline_utils_fixed import (
    ridge_pinv_with_boundary_weights, compute_boundary_weights
)
from utils import load_dataset, zscore_fit_stats, zscore_apply, zscore_inv, rmse

# 配置
DATA_PATH = "../../Compress_Test/test_complete.npz"
SAMPLE_INDEX = 0
M = 64
DEGREE = 3
LAMBDA = 1e-6


def test_boundary_fix(q, sample_idx=0, boundary_weight=10.0, boundary_frames=10):
    """
    对比标准方法和边界加权方法
    """
    N, T, J = q.shape
    
    print(f"\n{'='*80}")
    print(f"边界修复测试")
    print(f"{'='*80}")
    print(f"Sample: {sample_idx}, T={T}, M={M}, degree={DEGREE}")
    print(f"边界权重: {boundary_weight}x, 边界帧数: {boundary_frames}")
    
    # 归一化
    m, s = zscore_fit_stats(q)
    qz = zscore_apply(q, m, s)
    
    # 归一化时间
    t = normalized_times(T)
    
    # 构造样条基
    B = bspline_design_matrix(t, M, DEGREE)
    
    # 方法1: 标准Ridge
    print("\n[方法1] 标准Ridge回归")
    P_standard = ridge_pinv_standard(B, LAMBDA)
    theta_standard = encode_theta(P_standard, qz[sample_idx])
    qz_hat_standard = decode_signal(B, theta_standard)
    q_rec_standard = zscore_inv(qz_hat_standard[None, :, :], m, s)[0]
    
    errors_standard = np.abs(q[sample_idx] - q_rec_standard)
    rmse_standard = rmse(q[sample_idx], q_rec_standard)
    
    print(f"  整体RMSE: {rmse_standard:.6f}")
    print(f"  第1帧最大误差: {np.max(errors_standard[0, :]):.6f}")
    print(f"  前10帧平均误差: {np.mean(errors_standard[:10, :]):.6f}")
    print(f"  中间帧平均误差: {np.mean(errors_standard[T//2-5:T//2+5, :]):.6f}")
    
    # 方法2: 边界加权Ridge
    print("\n[方法2] 边界加权Ridge回归")
    P_weighted = ridge_pinv_with_boundary_weights(B, LAMBDA, boundary_weight, boundary_frames)
    theta_weighted = encode_theta(P_weighted, qz[sample_idx])
    qz_hat_weighted = decode_signal(B, theta_weighted)
    q_rec_weighted = zscore_inv(qz_hat_weighted[None, :, :], m, s)[0]
    
    errors_weighted = np.abs(q[sample_idx] - q_rec_weighted)
    rmse_weighted = rmse(q[sample_idx], q_rec_weighted)
    
    print(f"  整体RMSE: {rmse_weighted:.6f}")
    print(f"  第1帧最大误差: {np.max(errors_weighted[0, :]):.6f}")
    print(f"  前10帧平均误差: {np.mean(errors_weighted[:10, :]):.6f}")
    print(f"  中间帧平均误差: {np.mean(errors_weighted[T//2-5:T//2+5, :]):.6f}")
    
    # 改进统计
    improvement_first = (np.max(errors_standard[0, :]) - np.max(errors_weighted[0, :])) / np.max(errors_standard[0, :]) * 100
    improvement_front = (np.mean(errors_standard[:10, :]) - np.mean(errors_weighted[:10, :])) / np.mean(errors_standard[:10, :]) * 100
    
    print(f"\n[改进效果]")
    print(f"  第1帧误差降低: {improvement_first:.1f}%")
    print(f"  前10帧误差降低: {improvement_front:.1f}%")
    print(f"  整体RMSE变化: {(rmse_weighted - rmse_standard) / rmse_standard * 100:.1f}%")
    
    # 可视化
    visualize_comparison(
        q[sample_idx], q_rec_standard, q_rec_weighted,
        errors_standard, errors_weighted,
        boundary_weight, boundary_frames, T
    )
    
    return {
        'standard': {'rmse': rmse_standard, 'errors': errors_standard},
        'weighted': {'rmse': rmse_weighted, 'errors': errors_weighted},
        'improvement': {'first_frame': improvement_first, 'front_10': improvement_front}
    }


def visualize_comparison(q_orig, q_rec_std, q_rec_wgt, err_std, err_wgt, 
                        boundary_weight, boundary_frames, T):
    """
    可视化对比标准方法和加权方法
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. 权重分布
    ax1 = fig.add_subplot(gs[0, :])
    weights = compute_boundary_weights(T, boundary_weight, boundary_frames)
    ax1.plot(weights, linewidth=2, color='purple')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Standard weight')
    ax1.axvspan(0, boundary_frames, alpha=0.2, color='red', label=f'Boundary zone ({boundary_frames} frames)')
    ax1.axvspan(T-boundary_frames, T-1, alpha=0.2, color='red')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Weight', fontsize=12)
    ax1.set_title(f'Boundary Weight Distribution (max={boundary_weight}x)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差随时间变化（全局）
    ax2 = fig.add_subplot(gs[1, :])
    max_err_std = np.max(err_std, axis=1)
    max_err_wgt = np.max(err_wgt, axis=1)
    ax2.plot(max_err_std, linewidth=1.5, label='Standard Ridge', alpha=0.8, color='blue')
    ax2.plot(max_err_wgt, linewidth=1.5, label='Weighted Ridge', alpha=0.8, color='red')
    ax2.axvspan(0, boundary_frames, alpha=0.1, color='orange')
    ax2.axvspan(T-boundary_frames, T-1, alpha=0.1, color='orange')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Max Error across Joints', fontsize=12)
    ax2.set_title('Maximum Reconstruction Error over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 前100帧误差细节
    ax3 = fig.add_subplot(gs[2, 0])
    n_show = 100
    ax3.plot(max_err_std[:n_show], linewidth=2, label='Standard', alpha=0.8, color='blue')
    ax3.plot(max_err_wgt[:n_show], linewidth=2, label='Weighted', alpha=0.8, color='red')
    ax3.fill_between(range(n_show), max_err_std[:n_show], max_err_wgt[:n_show], 
                     alpha=0.3, color='green', label='Improvement')
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Max Error', fontsize=12)
    ax3.set_title(f'First {n_show} Frames - Zoomed In', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 后100帧误差细节
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(max_err_std[-n_show:], linewidth=2, label='Standard', alpha=0.8, color='blue')
    ax4.plot(max_err_wgt[-n_show:], linewidth=2, label='Weighted', alpha=0.8, color='red')
    ax4.fill_between(range(n_show), max_err_std[-n_show:], max_err_wgt[-n_show:], 
                     alpha=0.3, color='green', label='Improvement')
    ax4.set_xlabel(f'Time Step (offset={T-n_show})', fontsize=12)
    ax4.set_ylabel('Max Error', fontsize=12)
    ax4.set_title(f'Last {n_show} Frames - Zoomed In', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 每个关节的第一帧误差对比
    ax5 = fig.add_subplot(gs[3, 0])
    J = q_orig.shape[1]
    x = np.arange(J)
    width = 0.35
    ax5.bar(x - width/2, err_std[0, :], width, label='Standard', alpha=0.8, color='blue')
    ax5.bar(x + width/2, err_wgt[0, :], width, label='Weighted', alpha=0.8, color='red')
    ax5.set_xlabel('Joint', fontsize=12)
    ax5.set_ylabel('Absolute Error', fontsize=12)
    ax5.set_title('First Frame (t=0) Error by Joint', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'J{i+1}' for i in range(J)])
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 统计对比
    ax6 = fig.add_subplot(gs[3, 1])
    categories = ['First\nFrame', 'Front 10\nFrames', 'Middle\nFrames', 'Back 10\nFrames', 'Last\nFrame']
    std_values = [
        np.max(err_std[0, :]),
        np.mean(err_std[:10, :]),
        np.mean(err_std[T//2-5:T//2+5, :]),
        np.mean(err_std[-10:, :]),
        np.max(err_std[-1, :])
    ]
    wgt_values = [
        np.max(err_wgt[0, :]),
        np.mean(err_wgt[:10, :]),
        np.mean(err_wgt[T//2-5:T//2+5, :]),
        np.mean(err_wgt[-10:, :]),
        np.max(err_wgt[-1, :])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, std_values, width, label='Standard', alpha=0.8, color='blue')
    ax6.bar(x + width/2, wgt_values, width, label='Weighted', alpha=0.8, color='red')
    ax6.set_ylabel('Error', fontsize=12)
    ax6.set_title('Error Statistics Comparison', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, fontsize=10)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Boundary Weight Fix: Standard vs Weighted Ridge\n'
                 f'Boundary Weight={boundary_weight}x, Boundary Frames={boundary_frames}',
                 fontsize=15, fontweight='bold')
    
    plt.savefig('boundary_fix_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 已保存对比图: boundary_fix_comparison.png")
    plt.close()


def main():
    print("="*80)
    print("边界权重修复方案测试")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    dataset = load_dataset(DATA_PATH)
    q = dataset['q']
    N, T, J = q.shape
    print(f"数据形状: N={N}, T={T}, J={J}")
    
    # 测试不同的边界权重
    print("\n" + "="*80)
    print("测试1: boundary_weight=10.0, boundary_frames=10")
    print("="*80)
    results1 = test_boundary_fix(q, SAMPLE_INDEX, boundary_weight=10.0, boundary_frames=10)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"\n第一帧误差改善: {results1['improvement']['first_frame']:.1f}%")
    print(f"前10帧误差改善: {results1['improvement']['front_10']:.1f}%")


if __name__ == '__main__':
    main()











