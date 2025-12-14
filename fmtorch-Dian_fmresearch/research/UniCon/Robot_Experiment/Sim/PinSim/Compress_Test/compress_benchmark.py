#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCT→PCA 与 DCT→VAE 的统一基准脚本。
- 输入: test_complete.npz (包含 q_log, dq_log, tau_log)
- 输出: out_dir 下 results_summary.json + 各配置 result_*.json

用法示例：
  # 小批量快速测试（5个样本）
  python compress_benchmark.py --quick-test
  
  # 完整测试
  python compress_benchmark.py \
    --data test_complete.npz \
    --out ./results \
    --methods PCA VAE \
    --K 48 64 \
    --latent 16 32 64 \
    --vae-arch 2x512 3x1024 \
    --beta 0.0 0.001 \
    --epochs 60 --batch-size 128 --lr 1e-3
"""
import os, argparse, time, hashlib
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

import torch
import torch.utils.data as tud
from sklearn.decomposition import PCA

from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv,
    apply_group_weights, remove_group_weights,
    dct_truncate, idct_reconstruct, flatten_coeffs, unflatten_coeffs,
    group_rmse, weighted_rmse, save_json
)
from models_vae import MLPVAE


def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_hash(d: Dict[str, Any]) -> str:
    s = repr(sorted(d.items())).encode('utf-8')
    return hashlib.md5(s).hexdigest()[:10]


def run_pca(C: np.ndarray, K: int, F: int, T: int, cfg: Dict[str, Any], meta: Dict[str, Any], out_dir: str, X_orig: np.ndarray,
            wq: float, wdq: float, wtau: float):
    """C: (N,K,F) in z-scored & weighted space."""
    N = C.shape[0]
    X = flatten_coeffs(C)  # (N, K*F)

    r = int(cfg['latent'])
    # Ensure r is valid for PCA
    max_components = min(N, X.shape[1])
    if r > max_components:
        print(f"  [warn] latent={r} > max_components={max_components}, skipping PCA")
        return None
    
    pca = PCA(n_components=r, svd_solver='randomized' if N > r else 'full', random_state=42)
    Z = pca.fit_transform(X)
    Xh = pca.inverse_transform(Z)

    Chat = unflatten_coeffs(Xh, K, F)
    Xrec_nw = idct_reconstruct(Chat, T)  # 回到 z-scored & weighted 空间

    # 反权重 & 反zscore → 原尺度
    Xrec_z = remove_group_weights(Xrec_nw, wq, wdq, wtau)
    Xrec = zscore_inv(Xrec_z, meta['mean'], meta['std'])

    # 评估
    g = group_rmse(X_orig, Xrec)
    wr = weighted_rmse(X_orig, Xrec)

    res = {
        'method': 'PCA',
        'K': K,
        'latent': r,
        'explained_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),
        'metrics': {
            'wrmse': wr,
            **g
        },
        'shapes': {'N': int(N), 'T': int(T), 'F': int(F), 'D': int(K*F)}
    }
    res['config'] = cfg

    fname = f"result_PCA_K{K}_r{r}_{config_hash(res)}.json"
    save_json(res, os.path.join(out_dir, fname))
    return res


def train_vae(C: np.ndarray, K: int, F: int, T: int, cfg: Dict[str, Any], meta: Dict[str, Any], out_dir: str, X_orig: np.ndarray,
              wq: float, wdq: float, wtau: float, device: str):
    """VAE on (N, K*F)."""
    N = C.shape[0]
    X = flatten_coeffs(C)

    in_dim = X.shape[1]
    latent = int(cfg['latent'])

    # 解析架构字符串 e.g., "2x512"
    arch = cfg['vae_arch']
    try:
        layers, hidden = arch.lower().split('x')
        num_layers = int(layers)
        hidden_dim = int(hidden)
    except Exception:
        num_layers, hidden_dim = 2, 512

    beta = float(cfg.get('beta', 0.001))
    epochs = int(cfg.get('epochs', 60))
    bs = int(cfg.get('batch_size', 128))
    lr = float(cfg.get('lr', 1e-3))

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    dl = tud.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    model = MLPVAE(in_dim, latent, hidden_dim, num_layers, dropout=float(cfg.get('dropout', 0.0))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm(range(epochs), desc=f"VAE K={K} r={latent} arch={arch} beta={beta}")
    for _ in pbar:
        epoch_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(xb)
            loss, rec, kl = MLPVAE.loss_function(recon, xb, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        pbar.set_postfix({'loss': epoch_loss / N})

    # 评估
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        recon, mu, logvar = model(X_tensor)
        Xh = recon.cpu().numpy()

    Chat = unflatten_coeffs(Xh, K, F)
    Xrec_nw = idct_reconstruct(Chat, T)
    Xrec_z  = remove_group_weights(Xrec_nw, wq, wdq, wtau)
    Xrec    = zscore_inv(Xrec_z, meta['mean'], meta['std'])

    g = group_rmse(X_orig, Xrec)
    wr = weighted_rmse(X_orig, Xrec)

    # 存模型快照（可选）
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg, 'in_dim': in_dim, 'K': K, 'F': F},
               os.path.join(out_dir, f"vae_K{K}_r{latent}_{arch.replace('x','by')}.pt"))

    res = {
        'method': 'VAE',
        'K': K,
        'latent': latent,
        'vae_arch': arch,
        'beta': beta,
        'metrics': {
            'wrmse': wr,
            **g
        },
        'shapes': {'N': int(N), 'T': int(T), 'F': int(F), 'D': int(K*F)}
    }
    res['config'] = cfg

    fname = f"result_VAE_K{K}_r{latent}_{arch.replace('x','by')}_{config_hash(res)}.json"
    save_json(res, os.path.join(out_dir, fname))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='test_complete.npz', 
                        help='npz path with q_log,dq_log,tau_log')
    parser.add_argument('--out', type=str, default='./results')
    parser.add_argument('--methods', nargs='+', default=['PCA', 'VAE'])
    parser.add_argument('--K', nargs='+', type=int, default=[48, 64])
    parser.add_argument('--latent', nargs='+', type=int, default=[16, 32, 64])
    parser.add_argument('--vae-arch', nargs='+', default=['2x512', '3x1024'])
    parser.add_argument('--beta', nargs='+', type=float, default=[0.0, 0.001])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)

    # 加权标准化设置
    parser.add_argument('--wq', type=float, default=1.0)
    parser.add_argument('--wdq', type=float, default=0.25)
    parser.add_argument('--wtau', type=float, default=0.1)

    # 快速测试模式
    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式：只用5个样本，简化配置')

    args = parser.parse_args()

    # 快速测试模式配置
    if args.quick_test:
        print("[快速测试模式] 使用 10 个样本，简化配置")
        args.out = './quick_test_results'
        args.methods = ['PCA', 'VAE']
        args.K = [48]
        args.latent = [8]  # 小于样本数
        args.vae_arch = ['2x256']  # 更小的架构
        args.beta = [0.001]
        args.epochs = 10  # 更少的epochs
        quick_test_samples = 10
    else:
        quick_test_samples = None

    os.makedirs(args.out, exist_ok=True)
    seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device] {device}")

    # 加载数据 (N,T,21)
    print(f"[data] 加载 {args.data}...")
    d = load_dataset(args.data, max_samples=quick_test_samples)
    X = d['X']
    N, T, F = X.shape
    print(f"[data] 数据形状: N={N}, T={T}, F={F}")

    # z-score & 组权重（线性、可逆）
    mean, std = zscore_fit_stats(X)
    Xz  = zscore_apply(X, mean, std)
    Xzw = apply_group_weights(Xz, args.wq, args.wdq, args.wtau)

    meta = {'mean': mean, 'std': std, 'wq': args.wq, 'wdq': args.wdq, 'wtau': args.wtau}
    save_json({'args': vars(args), 'meta': {'N': N, 'T': T, 'F': F}}, os.path.join(args.out, 'run_meta.json'))

    summary = []

    for K in args.K:
        print(f"\n[DCT] 截断到 K={K}...")
        # DCT 截断
        C = dct_truncate(Xzw, K)  # (N,K,21)
        print(f"[DCT] DCT 系数形状: {C.shape}")
        
        for r in args.latent:
            if 'PCA' in args.methods:
                print(f"\n[PCA] K={K}, latent={r}")
                cfg = {
                    'latent': r,
                    'K': K,
                    'method': 'PCA',
                    'weights': {'wq': args.wq, 'wdq': args.wdq, 'wtau': args.wtau}
                }
                res = run_pca(C, K, F, T, cfg, meta, args.out, X, args.wq, args.wdq, args.wtau)
                if res is not None:
                    summary.append(res)
                    print(f"  WRMSE: {res['metrics']['wrmse']:.6f}")

            if 'VAE' in args.methods:
                for arch in args.vae_arch:
                    for beta in args.beta:
                        print(f"\n[VAE] K={K}, latent={r}, arch={arch}, beta={beta}")
                        cfg = {
                            'latent': r,
                            'K': K,
                            'method': 'VAE',
                            'vae_arch': arch,
                            'beta': beta,
                            'epochs': args.epochs,
                            'batch_size': args.batch_size,
                            'lr': args.lr,
                            'dropout': args.dropout,
                            'weights': {'wq': args.wq, 'wdq': args.wdq, 'wtau': args.wtau}
                        }
                        res = train_vae(C, K, F, T, cfg, meta, args.out, X, args.wq, args.wdq, args.wtau, device)
                        summary.append(res)
                        print(f"  WRMSE: {res['metrics']['wrmse']:.6f}")

    # 汇总保存
    save_json(summary, os.path.join(args.out, 'results_summary.json'))
    print(f"\n[完成] 结果保存到 {os.path.join(args.out, 'results_summary.json')}")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for res in summary:
        method = res['method']
        K = res['K']
        r = res['latent']
        wrmse = res['metrics']['wrmse']
        if method == 'PCA':
            evr = res['explained_variance_ratio']
            print(f"{method} K={K:3d} r={r:3d} | WRMSE={wrmse:.6f} | EVR={evr:.4f}")
        else:
            arch = res['vae_arch']
            beta = res['beta']
            print(f"{method} K={K:3d} r={r:3d} {arch:7s} β={beta:.4f} | WRMSE={wrmse:.6f}")


if __name__ == '__main__':
    main() 