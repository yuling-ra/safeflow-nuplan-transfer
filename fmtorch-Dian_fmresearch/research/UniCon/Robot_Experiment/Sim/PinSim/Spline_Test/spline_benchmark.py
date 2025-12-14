#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-Spline 压缩基准测试
对 q / dq / tau 三组分别做样条编码-解码
支持网格扫描：不同控制点数 M 与样条阶数 p
"""
import os
import argparse
import time
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from spline_utils import (
    normalized_times, bspline_design_matrix, bspline_deriv_design_matrix,
    ridge_pinv, encode_theta, decode_signal
)
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv,
    group_rmse, weighted_rmse, save_json
)


def config_hash(d: Dict[str, Any]) -> str:
    """生成配置哈希"""
    s = repr(sorted(d.items())).encode('utf-8')
    return hashlib.md5(s).hexdigest()[:10]


def run_spline_compression(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                           Mq: int, Mdq: int, Mtau: int,
                           deg_q: int, deg_dq: int, deg_tau: int,
                           lam_q: float, lam_dq: float, lam_tau: float,
                           wq: float, wdq: float, wtau: float,
                           eval_dq_from_q: bool = False) -> Dict[str, Any]:
    """
    运行样条压缩-重建
    
    参数:
        q, dq, tau: (N, T, J) 原始数据
        Mq, Mdq, Mtau: 控制点数量
        deg_q, deg_dq, deg_tau: 样条阶数
        lam_q, lam_dq, lam_tau: 岭回归正则化参数
        wq, wdq, wtau: 加权 RMSE 权重
        eval_dq_from_q: 是否从 q 的导数评估 dq
    
    返回:
        结果字典
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
    
    # 3. 构造样条基矩阵与伪逆
    Bq = bspline_design_matrix(t, Mq, deg_q)      # (T, Mq)
    Pq = ridge_pinv(Bq, lam_q)                    # (Mq, T)
    
    Bdq = bspline_design_matrix(t, Mdq, deg_dq)   # (T, Mdq)
    Pdq = ridge_pinv(Bdq, lam_dq)                 # (Mdq, T)
    
    Btau = bspline_design_matrix(t, Mtau, deg_tau) # (T, Mtau)
    Ptau = ridge_pinv(Btau, lam_tau)               # (Mtau, T)
    
    # 可选：q 的导数基矩阵
    Bq_dot = None
    if eval_dq_from_q:
        Bq_dot = bspline_deriv_design_matrix(t, Mq, deg_q, order=1)
    
    # 4. 编码-解码（批量处理所有样本）
    rmse_q_list = []
    rmse_dq_list = []
    rmse_tau_list = []
    rmse_dq_from_q_list = []
    
    for i in range(N):
        # q
        theta_q = encode_theta(Pq, qz[i])       # (Mq, J)
        qz_hat = decode_signal(Bq, theta_q)    # (T, J)
        q_rec = zscore_inv(qz_hat[None, :, :], mq, sq)[0]
        rmse_q_list.append(np.sqrt(np.mean((q[i] - q_rec) ** 2)))
        
        # dq
        theta_dq = encode_theta(Pdq, dqz[i])
        dqz_hat = decode_signal(Bdq, theta_dq)
        dq_rec = zscore_inv(dqz_hat[None, :, :], mdq, sdq)[0]
        rmse_dq_list.append(np.sqrt(np.mean((dq[i] - dq_rec) ** 2)))
        
        # tau
        theta_tau = encode_theta(Ptau, tauz[i])
        tauz_hat = decode_signal(Btau, theta_tau)
        tau_rec = zscore_inv(tauz_hat[None, :, :], mtau, stau)[0]
        rmse_tau_list.append(np.sqrt(np.mean((tau[i] - tau_rec) ** 2)))
        
        # 可选：从 q 的导数评估 dq
        if eval_dq_from_q and Bq_dot is not None:
            dqz_from_q = decode_signal(Bq_dot, theta_q)
            dq_from_q = zscore_inv(dqz_from_q[None, :, :], mdq, sdq)[0]
            rmse_dq_from_q_list.append(np.sqrt(np.mean((dq[i] - dq_from_q) ** 2)))
    
    # 5. 计算平均指标
    rmse_q_avg = float(np.mean(rmse_q_list))
    rmse_dq_avg = float(np.mean(rmse_dq_list))
    rmse_tau_avg = float(np.mean(rmse_tau_list))
    
    # 加权 RMSE（需要重建所有数据）
    # 为简化，这里用平均 RMSE 计算
    wrmse = np.sqrt(wq * rmse_q_avg**2 + wdq * rmse_dq_avg**2 + wtau * rmse_tau_avg**2)
    
    # 6. 构建结果
    latent_total = Mq * J + Mdq * J + Mtau * J
    orig_size = N * T * J * 3
    comp_size = N * latent_total
    compression_ratio = orig_size / comp_size
    
    result = {
        'method': 'Spline',
        'M': {'q': int(Mq), 'dq': int(Mdq), 'tau': int(Mtau), 'total': int(Mq + Mdq + Mtau)},
        'degree': {'q': int(deg_q), 'dq': int(deg_dq), 'tau': int(deg_tau)},
        'lambda': {'q': float(lam_q), 'dq': float(lam_dq), 'tau': float(lam_tau)},
        'latent_total': int(latent_total),
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
    
    if eval_dq_from_q and rmse_dq_from_q_list:
        result['metrics']['rmse_dq_from_q'] = float(np.mean(rmse_dq_from_q_list))
    
    return result


def main():
    parser = argparse.ArgumentParser(description='B-Spline 压缩基准测试')
    
    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='NPZ 数据文件路径')
    parser.add_argument('--out', type=str, required=True, help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数（快速测试）')
    
    # 样条参数 - 控制点数量（网格扫描）
    parser.add_argument('--Mq', nargs='+', type=int, default=[16, 32, 48, 64, 96, 128],
                        help='q 的控制点数量列表')
    parser.add_argument('--Mdq', nargs='+', type=int, default=[16, 32, 48, 64],
                        help='dq 的控制点数量列表')
    parser.add_argument('--Mtau', nargs='+', type=int, default=[16, 32, 48, 64],
                        help='tau 的控制点数量列表')
    
    # 样条阶数（网格扫描）
    parser.add_argument('--deg-q', nargs='+', type=int, default=[3, 4, 5],
                        help='q 的样条阶数列表')
    parser.add_argument('--deg-dq', nargs='+', type=int, default=[3],
                        help='dq 的样条阶数列表')
    parser.add_argument('--deg-tau', nargs='+', type=int, default=[3],
                        help='tau 的样条阶数列表')
    
    # 正则化参数
    parser.add_argument('--lam-q', type=float, default=1e-6, help='q 的岭回归正则化')
    parser.add_argument('--lam-dq', type=float, default=1e-6, help='dq 的岭回归正则化')
    parser.add_argument('--lam-tau', type=float, default=1e-6, help='tau 的岭回归正则化')
    
    # 权重
    parser.add_argument('--wq', type=float, default=1.0, help='q 的权重')
    parser.add_argument('--wdq', type=float, default=0.25, help='dq 的权重')
    parser.add_argument('--wtau', type=float, default=0.1, help='tau 的权重')
    
    # 其他选项
    parser.add_argument('--eval-dq-from-q', action='store_true',
                        help='从 q 的导数评估 dq')
    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式（小规模配置）')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick_test:
        print("[快速测试模式] 使用简化配置")
        args.max_samples = 10
        args.Mq = [16, 32]
        args.Mdq = [16]
        args.Mtau = [16]
        args.deg_q = [3]
        args.deg_dq = [3]
        args.deg_tau = [3]
        args.out = './quick_test_results'
    
    os.makedirs(args.out, exist_ok=True)
    
    # 打印系统信息
    from resource_monitor import print_system_info
    print_system_info()
    
    # 加载数据
    print(f"\n[数据] 加载 {args.data}...")
    dataset = load_dataset(args.data, max_samples=args.max_samples)
    q = dataset['q']
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    print(f"[数据] 形状: N={N}, T={T}, J={J}")
    
    # 保存元数据
    meta = {
        'args': vars(args),
        'data_shape': {'N': N, 'T': T, 'J': J},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_json(meta, os.path.join(args.out, 'run_meta.json'))
    
    # 网格扫描
    results = []
    total_configs = len(args.Mq) * len(args.Mdq) * len(args.Mtau) * \
                   len(args.deg_q) * len(args.deg_dq) * len(args.deg_tau)
    
    print(f"\n[扫描] 总配置数: {total_configs}")
    print("="*70)
    
    pbar = tqdm(total=total_configs, desc="Spline 基准测试")
    
    for Mq in args.Mq:
        for Mdq in args.Mdq:
            for Mtau in args.Mtau:
                for deg_q in args.deg_q:
                    for deg_dq in args.deg_dq:
                        for deg_tau in args.deg_tau:
                            try:
                                result = run_spline_compression(
                                    q, dq, tau,
                                    Mq, Mdq, Mtau,
                                    deg_q, deg_dq, deg_tau,
                                    args.lam_q, args.lam_dq, args.lam_tau,
                                    args.wq, args.wdq, args.wtau,
                                    args.eval_dq_from_q
                                )
                                
                                results.append(result)
                                
                                # 保存单个结果
                                tag = f"M({Mq},{Mdq},{Mtau})_deg({deg_q},{deg_dq},{deg_tau})"
                                fname = f"result_Spline_{tag}_{config_hash(result)}.json"
                                save_json(result, os.path.join(args.out, fname))
                                
                                pbar.set_postfix({
                                    'M': f"{Mq},{Mdq},{Mtau}",
                                    'deg': f"{deg_q},{deg_dq},{deg_tau}",
                                    'WRMSE': f"{result['metrics']['wrmse']:.6f}"
                                })
                                
                            except Exception as e:
                                print(f"\n[错误] M=({Mq},{Mdq},{Mtau}), deg=({deg_q},{deg_dq},{deg_tau}): {e}")
                            
                            pbar.update(1)
    
    pbar.close()
    
    # 保存汇总结果
    save_json(results, os.path.join(args.out, 'results_summary.json'))
    print(f"\n[完成] 结果保存到 {args.out}/results_summary.json")
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"总配置数: {len(results)}")
    
    if results:
        # 按 WRMSE 排序
        sorted_results = sorted(results, key=lambda x: x['metrics']['wrmse'])
        
        print("\n最佳配置 (Top 5):")
        for i, res in enumerate(sorted_results[:5], 1):
            M = res['M']
            deg = res['degree']
            wrmse = res['metrics']['wrmse']
            latent = res['latent_total']
            ratio = res['compression_ratio']
            print(f"{i}. M=({M['q']},{M['dq']},{M['tau']}) deg=({deg['q']},{deg['dq']},{deg['tau']}) | "
                  f"WRMSE={wrmse:.6f} | Latent={latent} | 压缩比={ratio:.2f}x")


if __name__ == '__main__':
    main()
