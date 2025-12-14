#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级样条压缩网格扫描
支持：
1. 从 q 导数评估 dq (Mdq=0 时启用)
2. 多档位 lambda 扫描
3. 更灵活的配置组合
"""
import os
import sys
import argparse
import time
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
import numpy as np

from spline_utils import (
    normalized_times, bspline_design_matrix, bspline_deriv_design_matrix,
    ridge_pinv, encode_theta, decode_signal
)
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv,
    group_rmse, weighted_rmse, save_json
)
from resource_monitor import ResourceMonitor, print_system_info


def run_spline_compression_advanced(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                                    Mq: int, Mdq: int, Mtau: int,
                                    deg_q: int, deg_dq: int, deg_tau: int,
                                    lam_q: float, lam_dq: float, lam_tau: float,
                                    wq: float, wdq: float, wtau: float,
                                    eval_dq_from_q: bool = False) -> dict:
    """
    高级样条压缩-重建
    支持 Mdq=0 时从 q 导数评估 dq
    """
    N, T, J = q.shape
    
    # 1. Z-score 标准化（分组）
    mq, sq = zscore_fit_stats(q)
    mdq, sdq = zscore_fit_stats(dq)
    mtau, stau = zscore_fit_stats(tau)
    
    qz = zscore_apply(q, mq, sq)
    dqz = zscore_apply(dq, mdq, sdq)
    tauz = zscore_apply(tau, mtau, stau)
    
    # 2. 归一化时间
    t = normalized_times(T)
    
    # 3. 构造样条基矩阵
    Bq = bspline_design_matrix(t, Mq, deg_q)
    Pq = ridge_pinv(Bq, lam_q)
    
    # dq: 如果 Mdq=0，从 q 导数计算；否则独立拟合
    use_dq_from_q = (Mdq == 0) or eval_dq_from_q
    
    if Mdq > 0 and not eval_dq_from_q:
        Bdq = bspline_design_matrix(t, Mdq, deg_dq)
        Pdq = ridge_pinv(Bdq, lam_dq)
    else:
        Bdq = None
        Pdq = None
    
    # q 的导数基矩阵（用于从 q 评估 dq）
    if use_dq_from_q:
        Bq_dot = bspline_deriv_design_matrix(t, Mq, deg_q, order=1)
    else:
        Bq_dot = None
    
    # tau
    Btau = bspline_design_matrix(t, Mtau, deg_tau)
    Ptau = ridge_pinv(Btau, lam_tau)
    
    # 4. 编码-解码
    rmse_q_list = []
    rmse_dq_list = []
    rmse_tau_list = []
    
    for i in range(N):
        # q
        theta_q = encode_theta(Pq, qz[i])
        qz_hat = decode_signal(Bq, theta_q)
        q_rec = zscore_inv(qz_hat[None, :, :], mq, sq)[0]
        rmse_q_list.append(np.sqrt(np.mean((q[i] - q_rec) ** 2)))
        
        # dq
        if use_dq_from_q:
            # 从 q 的导数评估 dq
            dqz_hat = decode_signal(Bq_dot, theta_q)
            dq_rec = zscore_inv(dqz_hat[None, :, :], mdq, sdq)[0]
        else:
            # 独立拟合 dq
            theta_dq = encode_theta(Pdq, dqz[i])
            dqz_hat = decode_signal(Bdq, theta_dq)
            dq_rec = zscore_inv(dqz_hat[None, :, :], mdq, sdq)[0]
        
        rmse_dq_list.append(np.sqrt(np.mean((dq[i] - dq_rec) ** 2)))
        
        # tau
        theta_tau = encode_theta(Ptau, tauz[i])
        tauz_hat = decode_signal(Btau, theta_tau)
        tau_rec = zscore_inv(tauz_hat[None, :, :], mtau, stau)[0]
        rmse_tau_list.append(np.sqrt(np.mean((tau[i] - tau_rec) ** 2)))
    
    # 5. 计算平均指标
    rmse_q_avg = float(np.mean(rmse_q_list))
    rmse_dq_avg = float(np.mean(rmse_dq_list))
    rmse_tau_avg = float(np.mean(rmse_tau_list))
    
    wrmse = np.sqrt(wq * rmse_q_avg**2 + wdq * rmse_dq_avg**2 + wtau * rmse_tau_avg**2)
    
    # 6. 计算潜变量维度
    if use_dq_from_q:
        latent_total = Mq * J + Mtau * J  # dq 不占用额外维度
        Mdq_effective = 0
    else:
        latent_total = Mq * J + Mdq * J + Mtau * J
        Mdq_effective = Mdq
    
    orig_size = N * T * J * 3
    comp_size = N * latent_total
    compression_ratio = orig_size / comp_size
    
    result = {
        'method': 'Spline_Advanced',
        'M': {'q': int(Mq), 'dq': int(Mdq_effective), 'tau': int(Mtau), 'total': int(Mq + Mdq_effective + Mtau)},
        'degree': {'q': int(deg_q), 'dq': int(deg_dq), 'tau': int(deg_tau)},
        'lambda': {'q': float(lam_q), 'dq': float(lam_dq), 'tau': float(lam_tau)},
        'latent_total': int(latent_total),
        'dq_from_q_derivative': use_dq_from_q,
        'metrics': {
            'wrmse': float(wrmse),
            'rmse_q': rmse_q_avg,
            'rmse_dq': rmse_dq_avg,
            'rmse_tau': rmse_tau_avg,
        },
        'shapes': {
            'N': int(N),
            'T': int(T),
            'J': int(J),
        },
        'weights': {'wq': float(wq), 'wdq': float(wdq), 'wtau': float(wtau)},
        'compression_ratio': float(compression_ratio),
    }
    
    return result


def worker_task(config: Tuple, q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                wq: float, wdq: float, wtau: float,
                eval_dq_from_q: bool) -> dict:
    """单个配置的工作任务"""
    Mq, Mdq, Mtau, deg_q, deg_dq, deg_tau, lam_q, lam_dq, lam_tau = config
    
    try:
        result = run_spline_compression_advanced(
            q, dq, tau,
            Mq, Mdq, Mtau,
            deg_q, deg_dq, deg_tau,
            lam_q, lam_dq, lam_tau,
            wq, wdq, wtau,
            eval_dq_from_q
        )
        return result
    except Exception as e:
        print(f"[错误] 配置失败: M=({Mq},{Mdq},{Mtau}), deg=({deg_q},{deg_dq},{deg_tau}), λ=({lam_q},{lam_dq},{lam_tau}): {e}")
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
    """运行高级网格扫描"""
    
    os.makedirs(out_dir, exist_ok=True)
    print_system_info()
    
    monitor = ResourceMonitor(cpu_threshold=85.0, memory_threshold=85.0)
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
            'eval_dq_from_q': eval_dq_from_q,
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
    
    # 统计信息
    stats = {
        'total_configs': total_configs,
        'successful_configs': len(results),
        'failed_configs': total_configs - len(results),
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_formatted': time.strftime('%H:%M:%S', time.gmtime(elapsed_time)),
    }
    
    if results:
        wrmse_values = [r['metrics']['wrmse'] for r in results]
        latent_values = [r['latent_total'] for r in results]
        
        stats['wrmse_min'] = float(np.min(wrmse_values))
        stats['wrmse_max'] = float(np.max(wrmse_values))
        stats['wrmse_mean'] = float(np.mean(wrmse_values))
        stats['wrmse_median'] = float(np.median(wrmse_values))
        
        stats['latent_min'] = int(np.min(latent_values))
        stats['latent_max'] = int(np.max(latent_values))
        stats['latent_mean'] = float(np.mean(latent_values))
        
        # 找到 latent ≤ 1000 的最佳配置
        under_1k = [r for r in results if r['latent_total'] <= 1000]
        if under_1k:
            best_under_1k = min(under_1k, key=lambda x: x['metrics']['wrmse'])
            stats['best_under_1k'] = {
                'M': best_under_1k['M'],
                'degree': best_under_1k['degree'],
                'lambda': best_under_1k['lambda'],
                'latent_total': best_under_1k['latent_total'],
                'wrmse': best_under_1k['metrics']['wrmse'],
                'rmse_q': best_under_1k['metrics']['rmse_q'],
                'rmse_dq': best_under_1k['metrics']['rmse_dq'],
                'rmse_tau': best_under_1k['metrics']['rmse_tau'],
            }
    
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
        
        print(f"\n潜变量维度统计:")
        print(f"  最小: {stats['latent_min']}")
        print(f"  最大: {stats['latent_max']}")
        print(f"  平均: {stats['latent_mean']:.1f}")
        
        if 'best_under_1k' in stats:
            best = stats['best_under_1k']
            print(f"\n≤1000维 最佳配置:")
            print(f"  M: q={best['M']['q']}, dq={best['M']['dq']}, tau={best['M']['tau']}")
            print(f"  degree: q={best['degree']['q']}, dq={best['degree']['dq']}, tau={best['degree']['tau']}")
            print(f"  lambda: q={best['lambda']['q']}, dq={best['lambda']['dq']}, tau={best['lambda']['tau']}")
            print(f"  潜变量: {best['latent_total']}")
            print(f"  WRMSE: {best['wrmse']:.6f}")
            print(f"  RMSE: q={best['rmse_q']:.6f}, dq={best['rmse_dq']:.6f}, tau={best['rmse_tau']:.6f}")
        
        # 最佳配置 Top 10
        sorted_results = sorted(results, key=lambda x: x['metrics']['wrmse'])
        print(f"\n最佳配置 (Top 10):")
        for i, res in enumerate(sorted_results[:10], 1):
            M = res['M']
            deg = res['degree']
            wrmse = res['metrics']['wrmse']
            latent = res['latent_total']
            ratio = res['compression_ratio']
            dq_from_q = "✓" if res.get('dq_from_q_derivative', False) else ""
            print(f"{i}. M=({M['q']},{M['dq']},{M['tau']}) deg=({deg['q']},{deg['dq']},{deg['tau']}) | "
                  f"WRMSE={wrmse:.6f} | Latent={latent} | 压缩比={ratio:.2f}x {dq_from_q}")
    
    print(f"\n结果保存到: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='高级样条压缩网格扫描')
    
    # 数据参数
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    
    # 网格参数
    parser.add_argument('--Mq', nargs='+', type=int, default=[128])
    parser.add_argument('--Mdq', nargs='+', type=int, default=[16])
    parser.add_argument('--Mtau', nargs='+', type=int, default=[16])
    parser.add_argument('--deg-q', nargs='+', type=int, default=[3])
    parser.add_argument('--deg-dq', nargs='+', type=int, default=[3])
    parser.add_argument('--deg-tau', nargs='+', type=int, default=[3])
    
    # Lambda 参数（多档位）
    parser.add_argument('--lam-q', nargs='+', type=float, default=[1e-6])
    parser.add_argument('--lam-dq', nargs='+', type=float, default=[1e-6])
    parser.add_argument('--lam-tau', nargs='+', type=float, default=[1e-6])
    
    # 权重
    parser.add_argument('--wq', type=float, default=1.0)
    parser.add_argument('--wdq', type=float, default=0.25)
    parser.add_argument('--wtau', type=float, default=0.1)
    
    # 其他选项
    parser.add_argument('--eval-dq-from-q', action='store_true',
                        help='从 q 导数评估 dq（当 Mdq=0 时自动启用）')
    
    # 并行参数
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=10)
    
    args = parser.parse_args()
    
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
