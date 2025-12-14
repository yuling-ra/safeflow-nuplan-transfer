#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多头 PCA 网格搜索
对 (Kq, Kdq, Ktau) × (rq, rdq, rtau) × filters 做 sweep
支持多进程并行加速 + 资源监控
"""
import os, argparse, itertools, sys, subprocess, time
from glob import glob
from multiprocessing import Pool, cpu_count
from utils import save_json
from resource_monitor import ResourceMonitor, print_system_info


def parse_triplet_list(items):
    """
    解析三元组列表，如 ["400,50,50", "360,70,70"] -> [(400,50,50),...]
    """
    out = []
    for s in items:
        parts = s.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid triplet: {s}")
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
        out.append((a, b, c))
    return out


def run_single_config(args_tuple):
    """
    运行单个配置（用于多进程）
    返回: (success, config_tag, error_msg)
    """
    (Kq, Kdq, Ktau, rq, rdq, rtau, filt, data_path, out_dir, 
     wq, wdq, wtau, max_samples, save_mapping) = args_tuple
    
    tag = f"K({Kq},{Kdq},{Ktau})_r({rq},{rdq},{rtau})_f({filt})"
    
    cmd = [
        sys.executable, 'compress_multigroup_pca.py',
        '--data', data_path,
        '--out', out_dir,
        '--Kq', str(Kq),
        '--Kdq', str(Kdq),
        '--Ktau', str(Ktau),
        '--rq', str(rq),
        '--rdq', str(rdq),
        '--rtau', str(rtau),
        '--q-filter', filt,
        '--dq-filter', filt,
        '--tau-filter', filt,
        '--wq', str(wq),
        '--wdq', str(wdq),
        '--wtau', str(wtau)
    ]
    
    if max_samples:
        cmd += ['--max-samples', str(max_samples)]
    
    if save_mapping:
        cmd += ['--save-mapping']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (True, tag, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Return code: {e.returncode}\nStderr: {e.stderr[:200]}"
        return (False, tag, error_msg)


def main():
    parser = argparse.ArgumentParser(description='Sweep multi-head PCA configurations')
    
    # 数据
    parser.add_argument('--data', type=str, required=True, help='NPZ data path')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples for quick test')
    
    # DCT K 网格
    parser.add_argument('--Kq', type=int, nargs='+', default=[256, 384, 512],
                        help='DCT K values for q')
    parser.add_argument('--Kdq', type=int, nargs='+', default=[128, 192],
                        help='DCT K values for dq')
    parser.add_argument('--Ktau', type=int, nargs='+', default=[128, 192],
                        help='DCT K values for tau')
    
    # Latent 三元组
    parser.add_argument('--latent-triplets', nargs='+',
                        default=['100,25,25', '200,50,50', '400,50,50', '360,70,70', '320,90,90'],
                        help='Latent dimension triplets (rq,rdq,rtau)')
    
    # 滤波器
    parser.add_argument('--filters', nargs='+', default=['none'],
                        help='Filter specs to test (e.g., none, ema:9, sg:21,3)')
    
    # 权重
    parser.add_argument('--wq', type=float, default=1.0)
    parser.add_argument('--wdq', type=float, default=0.25)
    parser.add_argument('--wtau', type=float, default=0.1)
    
    # 其他
    parser.add_argument('--save-mapping', action='store_true', help='Save PCA mappings')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--cpu-threshold', type=float, default=85.0,
                        help='CPU usage threshold for safety (default: 85%%)')
    parser.add_argument('--memory-threshold', type=float, default=85.0,
                        help='Memory usage threshold for safety (default: 85%%)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Process jobs in batches (0=all at once, >0=batch size)')
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # 打印系统信息
    print_system_info()
    
    # 初始化资源监控器
    monitor = ResourceMonitor(
        cpu_threshold=args.cpu_threshold,
        memory_threshold=args.memory_threshold
    )
    
    # 解析 latent 三元组
    triplets = parse_triplet_list(args.latent_triplets)
    
    # 生成所有组合
    jobs = list(itertools.product(args.Kq, args.Kdq, args.Ktau, triplets, args.filters))
    
    # 动态确定安全的 worker 数量
    requested_workers = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
    num_workers = monitor.get_safe_worker_count(requested_workers)
    
    print("="*70)
    print(f"Multi-Head PCA Sweep (Parallel)")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print(f"Total configurations: {len(jobs)}")
    print(f"Parallel workers: {num_workers} (available CPUs: {cpu_count()})")
    print(f"  Kq: {args.Kq}")
    print(f"  Kdq: {args.Kdq}")
    print(f"  Ktau: {args.Ktau}")
    print(f"  Latent triplets: {len(triplets)}")
    print(f"  Filters: {args.filters}")
    print("="*70)
    
    # 准备参数元组
    job_args = []
    for Kq, Kdq, Ktau, (rq, rdq, rtau), filt in jobs:
        job_args.append((
            Kq, Kdq, Ktau, rq, rdq, rtau, filt,
            args.data, args.out,
            args.wq, args.wdq, args.wtau,
            args.max_samples, args.save_mapping
        ))
    
    # 多进程执行（带资源监控和分批处理）
    print(f"\nStarting parallel execution with {num_workers} workers...")
    print(f"Resource monitoring: CPU < {args.cpu_threshold}%, Memory < {args.memory_threshold}%")
    if args.batch_size > 0:
        print(f"Batch mode: Processing {args.batch_size} jobs at a time")
    print("")
    
    start_time = time.time()
    completed = 0
    failed = 0
    aborted = False
    
    # 分批处理（如果指定了batch_size）
    batch_size = args.batch_size if args.batch_size > 0 else len(job_args)
    num_batches = (len(job_args) + batch_size - 1) // batch_size
    
    try:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(job_args))
            batch_jobs = job_args[start_idx:end_idx]
            
            if num_batches > 1:
                print(f"\n--- Batch {batch_idx + 1}/{num_batches} ({len(batch_jobs)} jobs) ---")
            
            # 批次开始前检查资源
            safe, msg = monitor.check_resources()
            if not safe:
                print(f"\n[Resource Monitor] System overloaded before batch start: {msg}")
                print("[Resource Monitor] Waiting for resources...")
                if not monitor.wait_for_resources(timeout=120):
                    print("[Resource Monitor] Timeout waiting for resources. Aborting.")
                    aborted = True
                    break
            
            with Pool(processes=num_workers) as pool:
                for success, tag, error in pool.imap_unordered(run_single_config, batch_jobs):
                    completed += 1
                    if success:
                        print(f"[{completed}/{len(jobs)}] ✓ {tag}")
                    else:
                        failed += 1
                        print(f"[{completed}/{len(jobs)}] ✗ {tag}")
                        if error:
                            print(f"    Error: {error}")
                    
                    # 定期检查资源
                    if completed % 3 == 0:
                        safe, msg = monitor.check_resources()
                        if not safe:
                            print(f"\n[Resource Monitor] ⚠ WARNING: {msg}")
                            print("[Resource Monitor] Consider reducing --num-workers or using --batch-size")
                    
                    # 显示进度和预估时间
                    if completed % 5 == 0 or completed == len(jobs):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(jobs) - completed) / rate if rate > 0 else 0
                        
                        # 显示系统状态
                        import psutil
                        cpu_now = psutil.cpu_percent(interval=0.1)
                        mem_now = psutil.virtual_memory().percent
                        
                        print(f"    Progress: {completed}/{len(jobs)} ({100*completed/len(jobs):.1f}%) | "
                              f"Rate: {rate:.2f} cfg/s | ETA: {eta/60:.1f} min")
                        print(f"    System: CPU {cpu_now:.1f}%, Memory {mem_now:.1f}%")
                        print("")
            
            # 批次间短暂休息，让系统恢复
            if batch_idx < num_batches - 1:
                print("[Resource Monitor] Cooling down between batches...")
                time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\n[Interrupted] User stopped the sweep")
        aborted = True
    except Exception as e:
        print(f"\n[Error] Sweep failed: {e}")
        aborted = True
    
    total_time = time.time() - start_time
    
    # 汇总结果
    print("\n" + "="*70)
    print("Aggregating results...")
    print("="*70)
    
    results = []
    try:
        import orjson
        use_orjson = True
    except ImportError:
        import json
        use_orjson = False
    
    for fp in sorted(glob(os.path.join(args.out, 'result_MHPCA_*.json'))):
        try:
            if use_orjson:
                with open(fp, 'rb') as f:
                    res = orjson.loads(f.read())
            else:
                with open(fp, 'r') as f:
                    res = json.load(f)
            results.append(res)
        except Exception as e:
            print(f"[warn] Failed to load {fp}: {e}")
    
    # 按 WRMSE 排序
    results.sort(key=lambda r: (r['metrics']['wrmse'], r['metrics']['rmse_q']))
    
    # 保存汇总
    summary_path = os.path.join(args.out, 'results_summary_multigroup.json')
    save_json(results, summary_path)
    
    print(f"\n[Complete] Saved {len(results)} results to: {summary_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {completed - failed}/{completed}")
    print(f"Failed: {failed}/{completed}")
    if aborted:
        print(f"Status: ⚠ ABORTED - Partial results saved")
    print(f"Average rate: {completed/total_time:.2f} configs/second")
    
    # 打印 Top 10
    print("\n" + "="*70)
    print("Top 10 Configurations (by WRMSE)")
    print("="*70)
    
    for i, r in enumerate(results[:10], 1):
        K = r['K']
        lat = r['latent']
        metrics = r['metrics']
        print(f"{i:2d}. K=({K['q']:3d},{K['dq']:3d},{K['tau']:3d}) "
              f"r=({lat['q']:3d},{lat['dq']:2d},{lat['tau']:2d}) | "
              f"WRMSE={metrics['wrmse']:.6f} | "
              f"q={metrics['rmse_q']:.6f}, dq={metrics['rmse_dq']:.6f}, tau={metrics['rmse_tau']:.6f}")
    
    print("\n" + "="*70)
    print("Sweep Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

