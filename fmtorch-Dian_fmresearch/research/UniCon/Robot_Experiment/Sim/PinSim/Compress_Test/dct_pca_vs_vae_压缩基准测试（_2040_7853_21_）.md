# 项目结构与脚本

下面是完整可运行的最小项目，支持：
- 统一 **DCT→PCA(SVD)** 与 **DCT→VAE** 的对比评测
- 可调时间频段系数 K、潜维 r、VAE 架构(层数×宽度)、β-VAE 系数等
- 评测指标：WRMSE（总/分组q,dq,τ）、每组MSE、解释方差（PCA）等
- 统一输出 `results_summary.json` + 各配置单独 `result_*.json`

> 说明：先对每个通道做 DCT-II（保留前 K 个低频系数），再在系数空间做 PCA 或 VAE；解码后做 IDCT-III 回到时域，再按加权标准化反变换，计算重建误差。整个管线线性且可复现（除 VAE 随机性）。

---

## requirements.txt

```txt
numpy>=1.22
scipy>=1.10
scikit-learn>=1.2
torch>=2.1
tqdm>=4.66
orjson>=3.9
```

---

## utils.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, math
import orjson
import numpy as np
from scipy.fft import dct, idct
from typing import Dict, Tuple

# ------------------------
# 统计 & 变换工具
# ------------------------

def zscore_fit_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每个特征(通道)的 mean/std。
    X: (N, T, F)
    返回: mean(F,), std(F,)
    """
    # 沿 N,T 两轴聚合
    mean = X.mean(axis=(0,1), dtype=np.float64)
    std  = X.std(axis=(0,1), dtype=np.float64)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, None, :]) / std[None, None, :]


def zscore_inv(Xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return Xn * std[None, None, :] + mean[None, None, :]


def apply_group_weights(X: np.ndarray, w_q: float, w_dq: float, w_tau: float) -> np.ndarray:
    """
    X: (N, T, F=21), 顺序假设为 [q7 | dq7 | tau7]
    返回加权后的数组（线性尺度，不是方差加权）。
    """
    Xw = X.copy()
    Xw[:, :, 0:7]  *= w_q
    Xw[:, :, 7:14] *= w_dq
    Xw[:, :, 14:21]*= w_tau
    return Xw


def remove_group_weights(Xw: np.ndarray, w_q: float, w_dq: float, w_tau: float) -> np.ndarray:
    X = Xw.copy()
    X[:, :, 0:7]  /= max(w_q, 1e-12)
    X[:, :, 7:14] /= max(w_dq,1e-12)
    X[:, :, 14:21]/= max(w_tau,1e-12)
    return X


def dct_truncate(X: np.ndarray, K: int) -> np.ndarray:
    """
    对时间轴做 DCT-II(ortho)，截断到 K。
    X: (N, T, F)
    返回 C: (N, K, F)
    """
    C_full = dct(X, type=2, n=None, axis=1, norm='ortho')  # (N, T, F)
    K = min(K, C_full.shape[1])
    return C_full[:, :K, :].astype(np.float32, copy=False)


def idct_reconstruct(C: np.ndarray, T: int) -> np.ndarray:
    """
    IDCT-III 还原到时域长度 T。
    C: (N, K, F)
    返回 X_rec: (N, T, F)
    """
    # 先 pad 回长度 T，在 DCT 空间补零
    N, K, F = C.shape
    Z = np.zeros((N, T, F), dtype=np.float32)
    Z[:, :K, :] = C
    X_rec = idct(Z, type=3, n=None, axis=1, norm='ortho')
    return X_rec


def flatten_coeffs(C: np.ndarray) -> np.ndarray:
    """(N, K, F) -> (N, K*F)"""
    N, K, F = C.shape
    return C.reshape(N, K*F)


def unflatten_coeffs(vec: np.ndarray, K: int, F: int) -> np.ndarray:
    N = vec.shape[0]
    return vec.reshape(N, K, F)

# ------------------------
# 评估指标
# ------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def group_rmse(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    X,Y: (N,T,21) 组别: q(0:7), dq(7:14), tau(14:21)
    返回每组 RMSE
    """
    g = {}
    g['rmse_q']   = rmse(X[:, :, 0:7],   Y[:, :, 0:7])
    g['rmse_dq']  = rmse(X[:, :, 7:14],  Y[:, :, 7:14])
    g['rmse_tau'] = rmse(X[:, :, 14:21], Y[:, :, 14:21])
    g['rmse_all'] = rmse(X, Y)
    return g


def weighted_rmse(X: np.ndarray, Y: np.ndarray, wq=1.0, wdq=0.25, wtau=0.1) -> float:
    e_q   = np.mean((X[:, :, 0:7]   - Y[:, :, 0:7])   ** 2)
    e_dq  = np.mean((X[:, :, 7:14]  - Y[:, :, 7:14])  ** 2)
    e_tau = np.mean((X[:, :, 14:21] - Y[:, :, 14:21]) ** 2)
    w_mse = wq*e_q + wdq*e_dq + wtau*e_tau
    return float(np.sqrt(w_mse))

# ------------------------
# IO 工具
# ------------------------

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_dataset(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    # 期望键
    q  = data['q_log'].astype(np.float32)
    dq = data['dq_log'].astype(np.float32)
    tau= data['tau_log'].astype(np.float32)
    # 组合为 (N,T,21)
    X = np.concatenate([q, dq, tau], axis=2)
    return {
        'X': X,  # (N,T,21)
        'T': X.shape[1],
        'F': X.shape[2],
        'N': X.shape[0],
    }
```

---

## models_vae.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
from typing import Tuple

class MLPVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)

        # decoder
        dlayers = []
        d = latent_dim
        for i in range(num_layers):
            dlayers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                dlayers += [nn.Dropout(dropout)]
            d = hidden_dim
        dlayers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dlayers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1e-3):
        # MSE 重建 + beta * KL
        recon_loss = torch.mean((recon_x - x) ** 2)
        # KL(N(mu, sigma) || N(0,1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss.detach(), kl.detach()
```

---

## compress_benchmark.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCT→PCA 与 DCT→VAE 的统一基准脚本。
- 输入: 包含 q_log, dq_log, tau_log 的 npz
- 输出: out_dir 下 results_summary.json + 各配置 result_*.json

用法示例：
  python compress_benchmark.py \
    --data /path/to/data.npz \
    --out ./out_results \
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
    pca = PCA(n_components=r, svd_solver='randomized', random_state=42)
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
    parser.add_argument('--data', type=str, required=True, help='npz path with q_log,dq_log,tau_log')
    parser.add_argument('--out', type=str, required=True)
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

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据 (N,T,21)
    d = load_dataset(args.data)
    X = d['X']
    N, T, F = X.shape

    # z-score & 组权重（线性、可逆）
    mean, std = zscore_fit_stats(X)
    Xz  = zscore_apply(X, mean, std)
    Xzw = apply_group_weights(Xz, args.wq, args.wdq, args.wtau)

    meta = {'mean': mean, 'std': std, 'wq': args.wq, 'wdq': args.wdq, 'wtau': args.wtau}
    save_json({'args': vars(args), 'meta': {'N': N, 'T': T, 'F': F}}, os.path.join(args.out, 'run_meta.json'))

    summary = []

    for K in args.K:
        # DCT 截断
        C = dct_truncate(Xzw, K)  # (N,K,21)
        for r in args.latent:
            if 'PCA' in args.methods:
                cfg = {
                    'latent': r,
                    'K': K,
                    'method': 'PCA',
                    'weights': {'wq': args.wq, 'wdq': args.wdq, 'wtau': args.wtau}
                }
                res = run_pca(C, K, F, T, cfg, meta, args.out, X, args.wq, args.wdq, args.wtau)
                summary.append(res)

            if 'VAE' in args.methods:
                for arch in args.vae_arch:
                    for beta in args.beta:
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

    # 汇总保存
    save_json(summary, os.path.join(args.out, 'results_summary.json'))
    print(f"Saved summary to {os.path.join(args.out, 'results_summary.json')}")


if __name__ == '__main__':
    main()
```

---

## run_benchmark.sh

```bash
#!/usr/bin/env bash
set -e

DATA=${1:-/path/to/your_data.npz}
OUTDIR=${2:-./out_$(date +%F_%H%M%S)}

python3 compress_benchmark.py \
  --data "$DATA" \
  --out "$OUTDIR" \
  --methods PCA VAE \
  --K 48 64 \
  --latent 16 32 64 \
  --vae-arch 2x512 3x1024 \
  --beta 0.0 0.001 \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-3 \
  --wq 1.0 --wdq 0.25 --wtau 0.1

echo "Results at: $OUTDIR/results_summary.json"
```

---

## 使用说明（简要）
1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据：npz 含键 `q_log,dq_log,tau_log`，形状如 `(2040,7853,7)`。
3. 运行：
   - `bash run_benchmark.sh /path/to/data.npz ./out_dir`
   - 或直接 `python compress_benchmark.py --data ...` 自定义网格。
4. 查看结果：`out_dir/results_summary.json`（每个配置的 WRMSE 与分组RMSE）。

---

## 备注
- **可扩展**：你可以按需增加 K、潜维 r 取值；或在 `models_vae.py` 中加入 BN/skip 等 trick。
- **确定性**：PCA 管线完全确定；VAE 设定 `--seed` 固化初始随机性。
- **内存**：DCT 在时域维压到 K（如 64），特征维仅 K×21≈1344，易于在 GPU/CPU 上训练/评测。

