#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比不同策略的结果
生成对比表格和图表
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def load_results(results_dir: str) -> List[Dict]:
    """加载结果"""
    summary_path = os.path.join(results_dir, 'results_summary.json')
    
    if not os.path.exists(summary_path):
        return []
    
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_pareto_frontier(all_results: Dict[str, List[Dict]], out_dir: str):
    """Plot Pareto frontier: Latent dimensions vs WRMSE"""
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (strategy, results) in enumerate(all_results.items()):
        if not results:
            continue
        
        latent = [r['latent_total'] for r in results]
        wrmse = [r['metrics']['wrmse'] for r in results]
        
        plt.scatter(latent, wrmse, 
                   label=strategy, 
                   alpha=0.6, 
                   s=80,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)])
    
    # Mark 1000 dim line
    plt.axvline(x=1000, color='red', linestyle='--', alpha=0.5, label='1000 Dim Budget')
    
    plt.xlabel('Latent Dimensions', fontsize=12)
    plt.ylabel('Weighted RMSE', fontsize=12)
    plt.title('Pareto Frontier: Latent Dimensions vs Weighted RMSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_frontier.png'), dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_dir}/pareto_frontier.png")


def plot_rmse_breakdown(all_results: Dict[str, List[Dict]], out_dir: str):
    """Plot RMSE breakdown comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (strategy, results) in enumerate(all_results.items()):
        if not results:
            continue
        
        # Only take configs with latent <= 1000
        under_1k = [r for r in results if r['latent_total'] <= 1000]
        if not under_1k:
            continue
        
        latent = [r['latent_total'] for r in under_1k]
        rmse_q = [r['metrics']['rmse_q'] for r in under_1k]
        rmse_dq = [r['metrics']['rmse_dq'] for r in under_1k]
        rmse_tau = [r['metrics']['rmse_tau'] for r in under_1k]
        
        axes[0].scatter(latent, rmse_q, label=strategy, alpha=0.6, s=60, color=colors[i % len(colors)])
        axes[1].scatter(latent, rmse_dq, label=strategy, alpha=0.6, s=60, color=colors[i % len(colors)])
        axes[2].scatter(latent, rmse_tau, label=strategy, alpha=0.6, s=60, color=colors[i % len(colors)])
    
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
    plt.savefig(os.path.join(out_dir, 'rmse_breakdown_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_dir}/rmse_breakdown_comparison.png")


def print_comparison_table(all_results: Dict[str, List[Dict]]):
    """打印对比表格"""
    print("\n" + "="*100)
    print("策略对比表")
    print("="*100)
    
    for strategy, results in all_results.items():
        if not results:
            print(f"\n{strategy}: 无结果")
            continue
        
        print(f"\n{strategy}:")
        print(f"  总配置数: {len(results)}")
        
        # 统计
        wrmse_values = [r['metrics']['wrmse'] for r in results]
        latent_values = [r['latent_total'] for r in results]
        
        print(f"  WRMSE: 最小={np.min(wrmse_values):.6f}, 平均={np.mean(wrmse_values):.6f}")
        print(f"  潜变量: 最小={np.min(latent_values)}, 最大={np.max(latent_values)}, 平均={np.mean(latent_values):.1f}")
        
        # ≤1000维 最佳
        under_1k = [r for r in results if r['latent_total'] <= 1000]
        if under_1k:
            best = min(under_1k, key=lambda x: x['metrics']['wrmse'])
            print(f"  ≤1000维 最佳:")
            print(f"    M: q={best['M']['q']}, dq={best['M']['dq']}, tau={best['M']['tau']}")
            print(f"    degree: q={best['degree']['q']}, dq={best['degree']['dq']}, tau={best['degree']['tau']}")
            print(f"    潜变量: {best['latent_total']}")
            print(f"    WRMSE: {best['metrics']['wrmse']:.6f}")
            print(f"    RMSE: q={best['metrics']['rmse_q']:.6f}, dq={best['metrics']['rmse_dq']:.6f}, tau={best['metrics']['rmse_tau']:.6f}")
        
        # 全局最佳
        best_overall = min(results, key=lambda x: x['metrics']['wrmse'])
        print(f"  全局最佳:")
        print(f"    M: q={best_overall['M']['q']}, dq={best_overall['M']['dq']}, tau={best_overall['M']['tau']}")
        print(f"    潜变量: {best_overall['latent_total']}")
        print(f"    WRMSE: {best_overall['metrics']['wrmse']:.6f}")


def main():
    parser = argparse.ArgumentParser(description='对比不同策略的结果')
    parser.add_argument('--dirs', nargs='+', required=True, help='结果目录列表')
    parser.add_argument('--output', type=str, default='./comparison_results', help='输出目录')
    parser.add_argument('--names', nargs='+', default=None, help='策略名称（可选）')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 加载所有结果
    all_results = {}
    for i, results_dir in enumerate(args.dirs):
        if args.names and i < len(args.names):
            name = args.names[i]
        else:
            name = os.path.basename(results_dir)
        
        print(f"[加载] {name} 从 {results_dir}...")
        results = load_results(results_dir)
        all_results[name] = results
        print(f"  找到 {len(results)} 个配置")
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 生成图表
    print(f"\n[绘图] 生成对比图表...")
    plot_pareto_frontier(all_results, args.output)
    plot_rmse_breakdown(all_results, args.output)
    
    # 保存汇总
    summary = {}
    for strategy, results in all_results.items():
        if not results:
            continue
        
        under_1k = [r for r in results if r['latent_total'] <= 1000]
        best_under_1k = min(under_1k, key=lambda x: x['metrics']['wrmse']) if under_1k else None
        best_overall = min(results, key=lambda x: x['metrics']['wrmse'])
        
        summary[strategy] = {
            'total_configs': len(results),
            'best_under_1k': {
                'M': best_under_1k['M'],
                'degree': best_under_1k['degree'],
                'latent_total': best_under_1k['latent_total'],
                'wrmse': best_under_1k['metrics']['wrmse'],
                'rmse_q': best_under_1k['metrics']['rmse_q'],
                'rmse_dq': best_under_1k['metrics']['rmse_dq'],
                'rmse_tau': best_under_1k['metrics']['rmse_tau'],
            } if best_under_1k else None,
            'best_overall': {
                'M': best_overall['M'],
                'degree': best_overall['degree'],
                'latent_total': best_overall['latent_total'],
                'wrmse': best_overall['metrics']['wrmse'],
            }
        }
    
    with open(os.path.join(args.output, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[完成] 对比结果保存到 {args.output}")
    print("="*100)


if __name__ == '__main__':
    main()
