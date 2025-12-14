#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化样条压缩基准测试结果
生成参数量 vs RMSE 曲线图
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def load_results(results_dir: str) -> List[Dict]:
    """加载结果汇总文件"""
    summary_path = os.path.join(results_dir, 'results_summary.json')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到结果文件: {summary_path}")
    
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_params_vs_rmse(results: List[Dict], out_dir: str):
    """
    Plot latent dimensions vs RMSE curves
    Grouped by spline degree
    """
    # Extract data
    data_by_deg = {}
    
    for res in results:
        deg_q = res['degree']['q']
        latent = res['latent_total']
        wrmse = res['metrics']['wrmse']
        rmse_q = res['metrics']['rmse_q']
        rmse_dq = res['metrics']['rmse_dq']
        rmse_tau = res['metrics']['rmse_tau']
        
        if deg_q not in data_by_deg:
            data_by_deg[deg_q] = {
                'latent': [],
                'wrmse': [],
                'rmse_q': [],
                'rmse_dq': [],
                'rmse_tau': []
            }
        
        data_by_deg[deg_q]['latent'].append(latent)
        data_by_deg[deg_q]['wrmse'].append(wrmse)
        data_by_deg[deg_q]['rmse_q'].append(rmse_q)
        data_by_deg[deg_q]['rmse_dq'].append(rmse_dq)
        data_by_deg[deg_q]['rmse_tau'].append(rmse_tau)
    
    # Convert to numpy arrays and sort
    for deg in data_by_deg:
        for key in data_by_deg[deg]:
            data_by_deg[deg][key] = np.array(data_by_deg[deg][key])
        
        # Sort by latent
        idx = np.argsort(data_by_deg[deg]['latent'])
        for key in data_by_deg[deg]:
            data_by_deg[deg][key] = data_by_deg[deg][key][idx]
    
    # Plot WRMSE
    plt.figure(figsize=(10, 6))
    for deg in sorted(data_by_deg.keys()):
        plt.plot(data_by_deg[deg]['latent'], data_by_deg[deg]['wrmse'],
                marker='o', label=f'Degree {deg}', alpha=0.7, linewidth=2)
    
    plt.xlabel('Latent Dimensions', fontsize=12)
    plt.ylabel('Weighted RMSE', fontsize=12)
    plt.title('B-Spline Compression: Latent Dims vs Weighted RMSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_latent_vs_wrmse.png'), dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_dir}/curve_latent_vs_wrmse.png")
    
    # Plot grouped RMSE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for deg in sorted(data_by_deg.keys()):
        axes[0].plot(data_by_deg[deg]['latent'], data_by_deg[deg]['rmse_q'],
                    marker='o', label=f'Degree {deg}', alpha=0.7, linewidth=2)
        axes[1].plot(data_by_deg[deg]['latent'], data_by_deg[deg]['rmse_dq'],
                    marker='o', label=f'Degree {deg}', alpha=0.7, linewidth=2)
        axes[2].plot(data_by_deg[deg]['latent'], data_by_deg[deg]['rmse_tau'],
                    marker='o', label=f'Degree {deg}', alpha=0.7, linewidth=2)
    
    axes[0].set_xlabel('Latent Dimensions', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title('q (Position)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    
    axes[1].set_xlabel('Latent Dimensions', fontsize=11)
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_title('dq (Velocity)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    
    axes[2].set_xlabel('Latent Dimensions', fontsize=11)
    axes[2].set_ylabel('RMSE', fontsize=11)
    axes[2].set_title('tau (Torque)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_latent_vs_rmse_groups.png'), dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_dir}/curve_latent_vs_rmse_groups.png")
    
    plt.close('all')


def plot_compression_ratio(results: List[Dict], out_dir: str):
    """Plot compression ratio vs RMSE"""
    compression_ratios = [r['compression_ratio'] for r in results]
    wrmse_values = [r['metrics']['wrmse'] for r in results]
    degrees = [r['degree']['q'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    for deg in sorted(set(degrees)):
        idx = [i for i, d in enumerate(degrees) if d == deg]
        cr = [compression_ratios[i] for i in idx]
        wr = [wrmse_values[i] for i in idx]
        plt.scatter(cr, wr, label=f'Degree {deg}', alpha=0.6, s=60)
    
    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('Weighted RMSE', fontsize=12)
    plt.title('B-Spline Compression: Compression Ratio vs Weighted RMSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_compression_vs_wrmse.png'), dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_dir}/curve_compression_vs_wrmse.png")
    
    plt.close('all')


def print_summary(results: List[Dict]):
    """Print results summary"""
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    
    print(f"\nTotal Configurations: {len(results)}")
    
    # Statistics
    wrmse_values = [r['metrics']['wrmse'] for r in results]
    latent_values = [r['latent_total'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    
    print(f"\nWeighted RMSE:")
    print(f"  Min:    {np.min(wrmse_values):.6f}")
    print(f"  Max:    {np.max(wrmse_values):.6f}")
    print(f"  Mean:   {np.mean(wrmse_values):.6f}")
    print(f"  Median: {np.median(wrmse_values):.6f}")
    
    print(f"\nLatent Dimensions:")
    print(f"  Min:  {np.min(latent_values)}")
    print(f"  Max:  {np.max(latent_values)}")
    print(f"  Mean: {np.mean(latent_values):.1f}")
    
    print(f"\nCompression Ratio:")
    print(f"  Min:  {np.min(compression_ratios):.2f}x")
    print(f"  Max:  {np.max(compression_ratios):.2f}x")
    print(f"  Mean: {np.mean(compression_ratios):.2f}x")
    
    # Best configurations
    sorted_results = sorted(results, key=lambda x: x['metrics']['wrmse'])
    
    print(f"\nBest Configurations (Top 10):")
    print(f"{'Rank':<5} {'M(q,dq,tau)':<15} {'Deg(q,dq,tau)':<18} {'WRMSE':<12} {'Latent':<8} {'Ratio':<8}")
    print("-" * 70)
    
    for i, res in enumerate(sorted_results[:10], 1):
        M = res['M']
        deg = res['degree']
        wrmse = res['metrics']['wrmse']
        latent = res['latent_total']
        ratio = res['compression_ratio']
        
        M_str = f"({M['q']},{M['dq']},{M['tau']})"
        deg_str = f"({deg['q']},{deg['dq']},{deg['tau']})"
        
        print(f"{i:<4} {M_str:<15} {deg_str:<18} {wrmse:<12.6f} {latent:<8} {ratio:<8.2f}x")


def main():
    parser = argparse.ArgumentParser(description='可视化样条压缩基准测试结果')
    parser.add_argument('--input', type=str, required=True, help='结果目录路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录（默认与输入相同）')
    
    args = parser.parse_args()
    
    # 设置输出目录
    out_dir = args.output if args.output else args.input
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载结果
    print(f"[加载] 从 {args.input} 加载结果...")
    results = load_results(args.input)
    print(f"[加载] 找到 {len(results)} 个配置")
    
    # 打印摘要
    print_summary(results)
    
    # 生成图表
    print(f"\n[绘图] 生成可视化图表...")
    plot_params_vs_rmse(results, out_dir)
    plot_compression_ratio(results, out_dir)
    
    print(f"\n[完成] 图表保存到 {out_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
