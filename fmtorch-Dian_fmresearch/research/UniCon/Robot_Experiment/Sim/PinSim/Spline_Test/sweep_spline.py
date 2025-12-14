#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样条压缩网格扫描脚本（带资源监控与并行支持）
支持分批处理，避免系统过载
"""
import os
import sys
import argparse
import time
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
import numpy as np

from spline_benchmark import run_spline_compression
from utils import load_dataset, save_json
from resource_monitor import ResourceMonitor, print_system_info


def worker_task(config: Tuple, q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                wq: float, wdq: float, wtau: float,
                eval_dq_from_q: bool) -> dict:
    """
    单个配置的工作任务
    """
    Mq, Mdq, Mtau, deg_q, deg_dq, deg_tau, lam_q, lam_dq, lam_tau = config
    
    try:
        result = run_spline_compression(
            q, dq, tau,
            Mq, Mdq, Mtau,
            deg_q, deg_dq, deg_tau,
            lam_q, lam_dq, lam_tau,
            wq, wdq, wtau,
            eval_dq_from_q
        )
        return result
    except Exception as e:
        print(f"[错误] 配置 M=({Mq},{Mdq},{Mtau}), deg=({deg_q},{deg_dq},{deg_tau}), λ=({lam_q},{lam_dq},{lam_tau}): {e}")
        return None


def run_sweep(data_path: str, out_dir: str,
              Mq_list: List[int], Mdq_list: List[int], Mtau_list: List[int],
              deg_q_list: List[int], deg_dq_list: List[int], deg_tau_list: List[int],
              lam_q_list: List[float], lam_dq_list: List[float], lam_tau_list: List[float],
              wq: float, wdq: float, wtau: float,
              eval_dq_from_q: bool,
              max_samples: int = None,
              n_workers: int = 4,
              batch_size: int = 10):
    """
    运行网格扫描
    
    参数:
        data_path: 数据文件路径
        out_dir: 输出目录
        Mq_list, Mdq_list, Mtau_list: 控制点数量列表
        deg_q_list, deg_dq_list, deg_tau_list: 样条阶数列表
        lam_q, lam_dq, lam_tau: 正则化参数
        wq, wdq, wtau: 权重
        eval_dq_from_q: 是否从 q 导数评估 dq
        max_samples: 最大样本数
        n_workers: 并行 worker 数量
        batch_size: 批处理大小
    """
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    
    # 打印系统信息
    print_system_info()
    
    # 初始化资源监控
    monitor = ResourceMonitor(cpu_threshold=85.0, memory_threshold=85.0)
    
    # 动态调整 worker 数量
    safe_workers = monitor.get_safe_worker_count(n_workers)
    print(f"\n[并行] 使用 {safe_workers} 个 workers (请求: {n_workers})")
    
    # 加载数据
    print(f"\n[数据] 加载 {data_path}...")
    dataset = load_dataset(data_path, max_samples=max_samples)
    q = dataset['q']
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    print(f"[数据] 形状: N={N}, T={T}, J={J}")
    
    # 生成配置网格（包含 lambda）
    configs = []
    for Mq in Mq_list:
        for Mdq in Mdq_list:
            for Mtau in Mtau_list:
                for deg_q in deg_q_list:
                    for deg_dq in deg_dq_list:
                        for deg_tau in deg_tau_list:
                            for lam_q in lam_q_list:
                                for lam_dq in lam_dq_list:
                                    for lam_tau in lam_tau_list:
                                        configs.append((Mq, Mdq, Mtau, deg_q, deg_dq, deg_tau,
                                                       lam_q, lam_dq, lam_tau))
    
    total_configs = len(configs)
    print(f"\n[扫描] 总配置数: {total_configs}")
    print(f"[扫描] 批处理大小: {batch_size}")
    print("="*70)
    
    # 保存元数据
    meta = {
        'data_path': data_path,
        'data_shape': {'N': N, 'T': T, 'J': J},
        'total_configs': total_configs,
        'n_workers': safe_workers,
        'batch_size': batch_size,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'Mq': Mq_list, 'Mdq': Mdq_list, 'Mtau': Mtau_list,
            'deg_q': deg_q_list, 'deg_dq': deg_dq_list, 'deg_tau': deg_tau_list,
            'lam_q': lam_q_list, 'lam_dq': lam_dq_list, 'lam_tau': lam_tau_list,
            'wq': wq, 'wdq': wdq, 'wtau': wtau,
        }
    }
    save_json(meta, os.path.join(out_dir, 'sweep_meta.json'))
    
    # 分批处理
    results = []
    start_time = time.time()
    
    for batch_start in range(0, total_configs, batch_size):
        batch_end = min(batch_start + batch_size, total_configs)
        batch_configs = configs[batch_start:batch_end]
        
        print(f"\n[批次 {batch_start//batch_size + 1}/{(total_configs + batch_size - 1)//batch_size}] "
              f"处理配置 {batch_start+1}-{batch_end}/{total_configs}")
        
        # 检查资源
        safe, msg = monitor.check_resources()
        if not safe:
            print(f"[资源警告] {msg}")
            print("[资源警告] 等待资源恢复...")
            if not monitor.wait_for_resources(timeout=60.0):
                print("[错误] 资源长时间不可用，终止扫描")
                break
        
        # 并行处理批次
        if safe_workers > 1:
            with mp.Pool(processes=safe_workers) as pool:
                worker_fn = partial(
                    worker_task,
                    q=q, dq=dq, tau=tau,
                    wq=wq, wdq=wdq, wtau=wtau,
                    eval_dq_from_q=eval_dq_from_q
                )
                batch_results = pool.map(worker_fn, batch_configs)
        else:
            # 串行处理
            batch_results = []
            for config in batch_configs:
                result = worker_task(
                    config, q, dq, tau,
                    wq, wdq, wtau,
                    eval_dq_from_q
                )
                batch_results.append(result)
        
        # 收集结果
        for result in batch_results:
            if result is not None:
                results.append(result)
        
        # 保存中间结果
        save_json(results, os.path.join(out_dir, 'results_summary_partial.json'))
        
        print(f"  完成: {len([r for r in batch_results if r is not None])}/{len(batch_configs)} 配置")
        print(f"  累计: {len(results)}/{total_configs} 配置")
    
    elapsed_time = time.time() - start_time
    
    # 保存最终结果
    save_json(results, os.path.join(out_dir, 'results_summary.json'))
    
    # 保存统计信息
    stats = {
        'total_configs': total_configs,
        'successful_configs': len(results),
        'failed_configs': total_configs - len(results),
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_formatted': time.strftime('%H:%M:%S', time.gmtime(elapsed_time)),
    }
    
    if results:
        wrmse_values = [r['metrics']['wrmse'] for r in results]
        stats['wrmse_min'] = float(np.min(wrmse_values))
        stats['wrmse_max'] = float(np.max(wrmse_values))
        stats['wrmse_mean'] = float(np.mean(wrmse_values))
        stats['wrmse_median'] = float(np.median(wrmse_values))
    
    save_json(stats, os.path.join(out_dir, 'sweep_stats.json'))
    
    # 打印总结
    print("\n" + "="*70)
    print("扫描完成")
    print("="*70)
    print(f"总配置数: {total_configs}")
    print(f"成功: {len(results)}")
    print(f"失败: {total_configs - len(results)}")
    print(f"耗时: {stats['elapsed_time_formatted']}")
    
    if results:
        print(f"\nWRMSE 统计:")
        print(f"  最小: {stats['wrmse_min']:.6f}")
        print(f"  最大: {stats['wrmse_max']:.6f}")
        print(f"  平均: {stats['wrmse_mean']:.6f}")
        print(f"  中位数: {stats['wrmse_median']:.6f}")
        
        # 最佳配置
        sorted_results = sorted(results, key=lambda x: x['metrics']['wrmse'])
        print(f"\n最佳配置 (Top 5):")
        for i, res in enumerate(sorted_results[:5], 1):
            M = res['M']
            deg = res['degree']
            wrmse = res['metrics']['wrmse']
            latent = res['latent_total']
            ratio = res['compression_ratio']
            print(f"{i}. M=({M['q']},{M['dq']},{M['tau']}) deg=({deg['q']},{deg['dq']},{deg['tau']}) | "
                  f"WRMSE={wrmse:.6f} | Latent={latent} | 压缩比={ratio:.2f}x")
    
    print(f"\n结果保存到: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='样条压缩网格扫描')
    
    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='NPZ 数据文件路径')
    parser.add_argument('--out', type=str, required=True, help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数')
    
    # 网格参数
    parser.add_argument('--Mq', nargs='+', type=int, default=[16, 32, 48, 64, 96, 128],
                        help='q 控制点数量列表')
    parser.add_argument('--Mdq', nargs='+', type=int, default=[16, 32, 48, 64],
                        help='dq 控制点数量列表')
    parser.add_argument('--Mtau', nargs='+', type=int, default=[16, 32, 48, 64],
                        help='tau 控制点数量列表')
    parser.add_argument('--deg-q', nargs='+', type=int, default=[3, 4, 5],
                        help='q 样条阶数列表')
    parser.add_argument('--deg-dq', nargs='+', type=int, default=[3],
                        help='dq 样条阶数列表')
    parser.add_argument('--deg-tau', nargs='+', type=int, default=[3],
                        help='tau 样条阶数列表')
    
    # 其他参数（支持多档位）
    parser.add_argument('--lam-q', nargs='+', type=float, default=[1e-6])
    parser.add_argument('--lam-dq', nargs='+', type=float, default=[1e-6])
    parser.add_argument('--lam-tau', nargs='+', type=float, default=[1e-6])
    parser.add_argument('--wq', type=float, default=1.0)
    parser.add_argument('--wdq', type=float, default=0.25)
    parser.add_argument('--wtau', type=float, default=0.1)
    parser.add_argument('--eval-dq-from-q', action='store_true')
    
    # 并行参数
    parser.add_argument('--workers', type=int, default=4, help='并行 worker 数量')
    parser.add_argument('--batch-size', type=int, default=10, help='批处理大小')
    
    # 预设模式
    parser.add_argument('--mode', type=str, choices=['quick', 'small', 'medium', 'large', 'full'],
                        default=None, help='预设模式')
    
    args = parser.parse_args()
    
    # 预设模式
    if args.mode == 'quick':
        print("[模式] 快速测试")
        args.max_samples = 10
        args.Mq = [16, 32]
        args.Mdq = [16]
        args.Mtau = [16]
        args.deg_q = [3]
        args.deg_dq = [3]
        args.deg_tau = [3]
        args.batch_size = 5
    elif args.mode == 'small':
        print("[模式] 小规模扫描")
        args.Mq = [16, 32, 48, 64]
        args.Mdq = [16, 32]
        args.Mtau = [16, 32]
        args.deg_q = [3, 5]
        args.batch_size = 10
    elif args.mode == 'medium':
        print("[模式] 中等规模扫描")
        args.Mq = [16, 32, 48, 64, 96]
        args.Mdq = [16, 32, 48]
        args.Mtau = [16, 32, 48]
        args.deg_q = [3, 4, 5]
        args.batch_size = 15
    elif args.mode == 'large':
        print("[模式] 大规模扫描")
        args.Mq = [16, 32, 48, 64, 96, 128]
        args.Mdq = [16, 32, 48, 64]
        args.Mtau = [16, 32, 48, 64]
        args.deg_q = [3, 4, 5]
        args.batch_size = 20
    elif args.mode == 'full':
        print("[模式] 完整扫描")
        args.Mq = [16, 32, 48, 64, 96, 128, 160]
        args.Mdq = [16, 32, 48, 64, 96]
        args.Mtau = [16, 32, 48, 64, 96]
        args.deg_q = [3, 4, 5]
        args.deg_dq = [3, 4]
        args.deg_tau = [3, 4]
        args.batch_size = 25
    
    # 运行扫描
    run_sweep(
        data_path=args.data,
        out_dir=args.out,
        Mq_list=args.Mq,
        Mdq_list=args.Mdq,
        Mtau_list=args.Mtau,
        deg_q_list=args.deg_q,
        deg_dq_list=args.deg_dq,
        deg_tau_list=args.deg_tau,
        lam_q_list=args.lam_q,
        lam_dq_list=args.lam_dq,
        lam_tau_list=args.lam_tau,
        wq=args.wq,
        wdq=args.wdq,
        wtau=args.wtau,
        eval_dq_from_q=args.eval_dq_from_q,
        max_samples=args.max_samples,
        n_workers=args.workers,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
