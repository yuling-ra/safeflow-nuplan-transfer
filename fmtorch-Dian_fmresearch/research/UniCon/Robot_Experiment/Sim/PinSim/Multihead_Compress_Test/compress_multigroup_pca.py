#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多头 PCA 压缩：q / dq / tau 三组分别做 DCT→PCA→IDCT
支持可选预滤波（none / EMA / Savitzky-Golay / Butterworth）
可导出映射包用于后续推理
"""
import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv,
    dct_truncate, idct_reconstruct, flatten_coeffs, unflatten_coeffs,
    rmse, weighted_rmse, save_json
)
from filters import apply_filter


def run_multihead_pca(data_path: str, 
                      Kq: int, Kdq: int, Ktau: int,
                      rq: int, rdq: int, rtau: int,
                      q_filter: str = 'none',
                      dq_filter: str = 'none',
                      tau_filter: str = 'none',
                      wq: float = 1.0,
                      wdq: float = 0.25,
                      wtau: float = 0.1,
                      out_dir: str = None,
                      save_mapping: bool = False,
                      max_samples: int = None):
    """
    运行多头 PCA 压缩-重建
    
    参数:
        data_path: npz 数据文件路径
        Kq, Kdq, Ktau: DCT 截断系数个数（每组）
        rq, rdq, rtau: PCA 潜变量维度（每组）
        q_filter, dq_filter, tau_filter: 预滤波器规格
        wq, wdq, wtau: 加权 RMSE 权重
        out_dir: 输出目录
        save_mapping: 是否保存映射参数
        max_samples: 最大样本数（用于快速测试）
    
    返回:
        结果字典
    """
    
    # 1. 加载数据
    dataset = load_dataset(data_path, max_samples=max_samples)
    q = dataset['q']      # (N, T, 7)
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    
    print(f"[Data] N={N}, T={T}, J={J}")
    
    # 2. 可选预滤波（在原始尺度）
    print(f"[Filter] q: {q_filter}, dq: {dq_filter}, tau: {tau_filter}")
    q_f = apply_filter(q, q_filter)
    dq_f = apply_filter(dq, dq_filter)
    tau_f = apply_filter(tau, tau_filter)
    
    # 3. 分组 Z-score 标准化
    print("[Normalize] Computing z-score per group...")
    mq, sq = zscore_fit_stats(q_f)
    mdq, sdq = zscore_fit_stats(dq_f)
    mtau, stau = zscore_fit_stats(tau_f)
    
    qz = zscore_apply(q_f, mq, sq)
    dqz = zscore_apply(dq_f, mdq, sdq)
    tauz = zscore_apply(tau_f, mtau, stau)
    
    # 4. DCT 截断（每组独立）
    print(f"[DCT] Kq={Kq}, Kdq={Kdq}, Ktau={Ktau}")
    Cq = dct_truncate(qz, Kq)      # (N, Kq, 7)
    Cdq = dct_truncate(dqz, Kdq)   # (N, Kdq, 7)
    Ctau = dct_truncate(tauz, Ktau) # (N, Ktau, 7)
    
    # 展平
    Xq = flatten_coeffs(Cq)      # (N, Kq*7)
    Xdq = flatten_coeffs(Cdq)    # (N, Kdq*7)
    Xtau = flatten_coeffs(Ctau)  # (N, Ktau*7)
    
    # 5. PCA 降维（每组独立）
    print(f"[PCA] rq={rq}, rdq={rdq}, rtau={rtau}")
    
    # 检查维度有效性
    max_rq = min(N, Xq.shape[1])
    max_rdq = min(N, Xdq.shape[1])
    max_rtau = min(N, Xtau.shape[1])
    
    if rq > max_rq or rdq > max_rdq or rtau > max_rtau:
        print(f"[warn] Requested r exceeds max components:")
        print(f"  rq={rq} (max={max_rq}), rdq={rdq} (max={max_rdq}), rtau={rtau} (max={max_rtau})")
        rq = min(rq, max_rq)
        rdq = min(rdq, max_rdq)
        rtau = min(rtau, max_rtau)
        print(f"  Adjusted to: rq={rq}, rdq={rdq}, rtau={rtau}")
    
    pca_q = PCA(n_components=rq, svd_solver='randomized' if N > rq else 'full', random_state=42)
    pca_dq = PCA(n_components=rdq, svd_solver='randomized' if N > rdq else 'full', random_state=42)
    pca_tau = PCA(n_components=rtau, svd_solver='randomized' if N > rtau else 'full', random_state=42)
    
    Zq = pca_q.fit_transform(Xq)      # (N, rq)
    Zdq = pca_dq.fit_transform(Xdq)   # (N, rdq)
    Ztau = pca_tau.fit_transform(Xtau) # (N, rtau)
    
    evr_q = float(np.sum(pca_q.explained_variance_ratio_))
    evr_dq = float(np.sum(pca_dq.explained_variance_ratio_))
    evr_tau = float(np.sum(pca_tau.explained_variance_ratio_))
    
    print(f"  EVR: q={evr_q:.6f}, dq={evr_dq:.6f}, tau={evr_tau:.6f}")
    
    # 6. PCA 逆变换
    print("[Reconstruct] Inverse PCA...")
    Xq_hat = pca_q.inverse_transform(Zq)
    Xdq_hat = pca_dq.inverse_transform(Zdq)
    Xtau_hat = pca_tau.inverse_transform(Ztau)
    
    # 反展平
    Cq_hat = unflatten_coeffs(Xq_hat, Kq, J)
    Cdq_hat = unflatten_coeffs(Xdq_hat, Kdq, J)
    Ctau_hat = unflatten_coeffs(Xtau_hat, Ktau, J)
    
    # 7. IDCT 重建
    print("[IDCT] Reconstructing time domain...")
    qz_rec = idct_reconstruct(Cq_hat, T)
    dqz_rec = idct_reconstruct(Cdq_hat, T)
    tauz_rec = idct_reconstruct(Ctau_hat, T)
    
    # 8. 反标准化
    q_rec = zscore_inv(qz_rec, mq, sq)
    dq_rec = zscore_inv(dqz_rec, mdq, sdq)
    tau_rec = zscore_inv(tauz_rec, mtau, stau)
    
    # 9. 计算误差
    print("[Metrics] Computing reconstruction errors...")
    rmse_q = rmse(q, q_rec)
    rmse_dq = rmse(dq, dq_rec)
    rmse_tau = rmse(tau, tau_rec)
    wrmse = weighted_rmse(q, dq, tau, q_rec, dq_rec, tau_rec, wq, wdq, wtau)
    
    print(f"  RMSE: q={rmse_q:.6f}, dq={rmse_dq:.6f}, tau={rmse_tau:.6f}")
    print(f"  WRMSE: {wrmse:.6f}")
    
    # 10. 构建结果
    latent_total = rq + rdq + rtau
    orig_size = N * T * J * 3  # q, dq, tau
    comp_size = N * latent_total
    compression_ratio = orig_size / comp_size
    
    result = {
        'method': 'MultiHeadPCA',
        'filters': {'q': q_filter, 'dq': dq_filter, 'tau': tau_filter},
        'K': {'q': int(Kq), 'dq': int(Kdq), 'tau': int(Ktau)},
        'latent': {'q': int(rq), 'dq': int(rdq), 'tau': int(rtau), 'total': int(latent_total)},
        'metrics': {
            'wrmse': float(wrmse),
            'rmse_q': float(rmse_q),
            'rmse_dq': float(rmse_dq),
            'rmse_tau': float(rmse_tau)
        },
        'explained_variance_ratio': {
            'q': float(evr_q),
            'dq': float(evr_dq),
            'tau': float(evr_tau)
        },
        'shapes': {
            'N': int(N),
            'T': int(T),
            'J': int(J),
            'Dq': int(Kq * J),
            'Ddq': int(Kdq * J),
            'Dtau': int(Ktau * J)
        },
        'weights': {'wq': float(wq), 'wdq': float(wdq), 'wtau': float(wtau)},
        'compression_ratio': float(compression_ratio)
    }
    
    # 11. 保存结果
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = f"K({Kq},{Kdq},{Ktau})_r({rq},{rdq},{rtau})_f({q_filter}|{dq_filter}|{tau_filter})"
        result_path = os.path.join(out_dir, f"result_MHPCA_{tag}.json")
        save_json(result, result_path)
        print(f"\n[Saved] Result: {result_path}")
        
        # 保存映射参数
        if save_mapping:
            mapping_path = os.path.join(out_dir, f"mapping_MHPCA_{tag}.npz")
            np.savez_compressed(
                mapping_path,
                # Z-score 参数
                mq=mq, sq=sq, mdq=mdq, sdq=sdq, mtau=mtau, stau=stau,
                # PCA 参数
                Wq=pca_q.components_.astype(np.float32),
                mu_q=pca_q.mean_.astype(np.float32),
                Wdq=pca_dq.components_.astype(np.float32),
                mu_dq=pca_dq.mean_.astype(np.float32),
                Wtau=pca_tau.components_.astype(np.float32),
                mu_tau=pca_tau.mean_.astype(np.float32),
                # DCT K 参数
                Kq=np.int32(Kq), Kdq=np.int32(Kdq), Ktau=np.int32(Ktau),
                T=np.int32(T), J=np.int32(J)
            )
            print(f"[Saved] Mapping: {mapping_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Multi-head PCA compression test')
    
    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='NPZ data file path')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples for quick test')
    
    # 预滤波器
    parser.add_argument('--q-filter', type=str, default='none', help='Filter spec for q')
    parser.add_argument('--dq-filter', type=str, default='none', help='Filter spec for dq')
    parser.add_argument('--tau-filter', type=str, default='none', help='Filter spec for tau')
    
    # DCT 参数
    parser.add_argument('--Kq', type=int, default=256, help='DCT coeffs for q')
    parser.add_argument('--Kdq', type=int, default=128, help='DCT coeffs for dq')
    parser.add_argument('--Ktau', type=int, default=128, help='DCT coeffs for tau')
    
    # PCA 潜变量
    parser.add_argument('--rq', type=int, default=400, help='PCA latent dim for q')
    parser.add_argument('--rdq', type=int, default=50, help='PCA latent dim for dq')
    parser.add_argument('--rtau', type=int, default=50, help='PCA latent dim for tau')
    
    # 权重
    parser.add_argument('--wq', type=float, default=1.0, help='Weight for q in WRMSE')
    parser.add_argument('--wdq', type=float, default=0.25, help='Weight for dq in WRMSE')
    parser.add_argument('--wtau', type=float, default=0.1, help='Weight for tau in WRMSE')
    
    # 其他
    parser.add_argument('--save-mapping', action='store_true', help='Save PCA mapping parameters')
    
    args = parser.parse_args()
    
    result = run_multihead_pca(
        data_path=args.data,
        Kq=args.Kq, Kdq=args.Kdq, Ktau=args.Ktau,
        rq=args.rq, rdq=args.rdq, rtau=args.rtau,
        q_filter=args.q_filter,
        dq_filter=args.dq_filter,
        tau_filter=args.tau_filter,
        wq=args.wq, wdq=args.wdq, wtau=args.wtau,
        out_dir=args.out,
        save_mapping=args.save_mapping,
        max_samples=args.max_samples
    )
    
    print("\n" + "="*70)
    print("Multi-Head PCA Test Complete!")
    print("="*70)


if __name__ == '__main__':
    main()


