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



---

## Multi‑Head PCA（q / dq / τ 三头分组）

下面新增与升级：
1) **`compress_multigroup_pca.py`（升级版）**：支持三头 DCT→PCA，并新增**可选时域预滤波**（none / EMA / Savitzky–Golay / Butterworth 低通），以及**映射导出**。
2) **`sweep_multigroup_pca.py`（新）**：对 (Kq,Kdq,Ktau) 与 (rq,rdq,rtau) 做网格搜索，输出 `results_summary.json`。
3) **`run_sweep_multigroup.sh`（新）**：一键复现实验，默认给 q 更多带宽与维度，总 latent≤500。

---

### compress_multigroup_pca.py（升级版）
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, hashlib
import numpy as np
from typing import Dict, Any, Tuple
from scipy.fft import dct, idct
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.decomposition import PCA
import orjson

# ------------------------
# IO & utils
# ------------------------

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_dataset(npz_path: str):
    data = np.load(npz_path)
    q   = data['q_log'].astype(np.float32)    # (N,T,7)
    dq  = data['dq_log'].astype(np.float32)   # (N,T,7)
    tau = data['tau_log'].astype(np.float32)  # (N,T,7)
    assert q.shape == dq.shape == tau.shape
    N, T, J = q.shape
    return q, dq, tau, N, T, J


def zstats_fit(X):
    mean = X.mean(axis=(0,1), dtype=np.float64).astype(np.float32)
    std  = X.std(axis=(0,1), dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std

def zscore_apply(X, m, s):
    return (X - m[None, None, :]) / s[None, None, :]

def zscore_inv(Xn, m, s):
    return Xn * s[None, None, :] + m[None, None, :]

# ------------------------
# Optional temporal filters (per group)
# ------------------------

def apply_filter(X: np.ndarray, spec: str) -> np.ndarray:
    """Apply temporal filter along axis=1. spec formats:
    - 'none'
    - 'ema:win'          e.g., 'ema:9'
    - 'sg:win,poly'      e.g., 'sg:21,3' (win odd)
    - 'butter:cut,ord'   e.g., 'butter:0.05,4' (cut is 0..1 Nyquist)
    """
    if not spec or spec == 'none':
        return X
    kind = spec.split(':')[0]
    args = spec.split(':')[1] if ':' in spec else ''
    if kind == 'ema':
        win = int(args)
        alpha = 2.0 / (win + 1.0)
        Y = np.copy(X)
        # simple causal EMA; also do backward pass to make zero-phase-ish
        for _ in range(1):
            for c in range(X.shape[2]):
                for n in range(X.shape[0]):
                    s = 0.0
                    for t in range(X.shape[1]):
                        s = alpha * X[n, t, c] + (1 - alpha) * (s if t>0 else X[n, t, c])
                        Y[n, t, c] = s
            # backward pass
            Xb = Y[:, ::-1, :].copy()
            Yb = np.copy(Xb)
            for c in range(Xb.shape[2]):
                for n in range(Xb.shape[0]):
                    s = 0.0
                    for t in range(Xb.shape[1]):
                        s = alpha * Xb[n, t, c] + (1 - alpha) * (s if t>0 else Xb[n, t, c])
                        Yb[n, t, c] = s
            Y = Yb[:, ::-1, :]
        return Y.astype(np.float32)
    elif kind == 'sg':
        parts = args.split(',')
        win = int(parts[0]); poly = int(parts[1])
        Y = savgol_filter(X, window_length=win, polyorder=poly, axis=1, mode='interp')
        return Y.astype(np.float32)
    elif kind == 'butter':
        parts = args.split(',')
        cut = float(parts[0]); order = int(parts[1]) if len(parts)>1 else 4
        b, a = butter(order, cut, btype='low', fs=None, analog=False)
        # zero-phase filtering per series
        Y = np.empty_like(X)
        for c in range(X.shape[2]):
            for n in range(X.shape[0]):
                Y[n, :, c] = filtfilt(b, a, X[n, :, c], axis=0, method='gust')
        return Y.astype(np.float32)
    else:
        return X

# ------------------------
# DCT helpers
# ------------------------

def dct_trunc(X, K):
    C = dct(X, type=2, axis=1, norm='ortho')
    K = min(K, C.shape[1])
    return C[:, :K, :].astype(np.float32, copy=False)

def idct_recon(C, T):
    N, K, J = C.shape
    Z = np.zeros((N, T, J), dtype=np.float32)
    Z[:, :K, :] = C
    X = idct(Z, type=3, axis=1, norm='ortho')
    return X

def flatten(C):
    N, K, J = C.shape
    return C.reshape(N, K*J)

def unflatten(vec, K, J):
    N = vec.shape[0]
    return vec.reshape(N, K, J)

# ------------------------
# Metrics
# ------------------------

def rmse(A,B):
    return float(np.sqrt(np.mean((A-B)**2)))

def weighted_rmse(q_rec, dq_rec, tau_rec, q, dq, tau, wq=1.0, wdq=0.25, wtau=0.1):
    e_q   = np.mean((q - q_rec)   ** 2)
    e_dq  = np.mean((dq - dq_rec) ** 2)
    e_tau = np.mean((tau - tau_rec)** 2)
    return float(np.sqrt(wq*e_q + wdq*e_dq + wtau*e_tau))

# ------------------------
# Main
# ------------------------

def run(cfg):
    q, dq, tau, N, T, J = load_dataset(cfg.data)

    # 0) optional prefilters (original scale)
    q_f   = apply_filter(q,   cfg.q_filter)
    dq_f  = apply_filter(dq,  cfg.dq_filter)
    tau_f = apply_filter(tau, cfg.tau_filter)

    # 1) groupwise z-score
    mq, sq = zstats_fit(q_f)
    md, sd = zstats_fit(dq_f)
    mt, st = zstats_fit(tau_f)
    qz  = zscore_apply(q_f,  mq, sq)
    dqz = zscore_apply(dq_f, md, sd)
    tz  = zscore_apply(tau_f, mt, st)

    # 2) DCT truncation per group
    Cq  = dct_trunc(qz,  cfg.Kq)
    Cdq = dct_trunc(dqz, cfg.Kdq)
    Ct  = dct_trunc(tz,  cfg.Ktau)

    Xq, Xdq, Xt = flatten(Cq), flatten(Cdq), flatten(Ct)

    # 3) PCA per group
    pca_q  = PCA(n_components=cfg.rq,  svd_solver='randomized', random_state=42)
    pca_dq = PCA(n_components=cfg.rdq, svd_solver='randomized', random_state=42)
    pca_t  = PCA(n_components=cfg.rtau,svd_solver='randomized', random_state=42)

    Zq  = pca_q.fit_transform(Xq)
    Zdq = pca_dq.fit_transform(Xdq)
    Zt  = pca_t.fit_transform(Xt)

    # 4) inverse
    Xq_hat  = pca_q.inverse_transform(Zq)
    Xdq_hat = pca_dq.inverse_transform(Zdq)
    Xt_hat  = pca_t.inverse_transform(Zt)

    Cq_hat  = unflatten(Xq_hat,  cfg.Kq,  J)
    Cdq_hat = unflatten(Xdq_hat, cfg.Kdq, J)
    Ct_hat  = unflatten(Xt_hat,  cfg.Ktau, J)

    qz_rec  = idct_recon(Cq_hat,  T)
    dqz_rec = idct_recon(Cdq_hat, T)
    tz_rec  = idct_recon(Ct_hat,  T)

    q_rec   = zscore_inv(qz_rec,  mq, sq)
    dq_rec  = zscore_inv(dqz_rec, md, sd)
    tau_rec = zscore_inv(tz_rec,  mt, st)

    # 5) metrics
    rmse_q   = rmse(q,   q_rec)
    rmse_dq  = rmse(dq,  dq_rec)
    rmse_tau = rmse(tau, tau_rec)
    wrmse    = weighted_rmse(q_rec, dq_rec, tau_rec, q, dq, tau, cfg.wq, cfg.wdq, cfg.wtau)

    evr_q  = float(np.sum(pca_q.explained_variance_ratio_))
    evr_dq = float(np.sum(pca_dq.explained_variance_ratio_))
    evr_t  = float(np.sum(pca_t.explained_variance_ratio_))

    res = {
        'method': 'MultiHeadPCA',
        'filters': {'q': cfg.q_filter, 'dq': cfg.dq_filter, 'tau': cfg.tau_filter},
        'K': {'q': int(cfg.Kq), 'dq': int(cfg.Kdq), 'tau': int(cfg.Ktau)},
        'latent': {'q': int(cfg.rq), 'dq': int(cfg.rdq), 'tau': int(cfg.rtau)},
        'metrics': {'wrmse': wrmse, 'rmse_q': rmse_q, 'rmse_dq': rmse_dq, 'rmse_tau': rmse_tau},
        'explained_variance_ratio': {'q': evr_q, 'dq': evr_dq, 'tau': evr_t},
        'shapes': {'N': int(N), 'T': int(T), 'J': int(J),
                   'Dq': int(cfg.Kq*J), 'Ddq': int(cfg.Kdq*J), 'Dtau': int(cfg.Ktau*J),
                   'latent_total': int(cfg.rq+cfg.rdq+cfg.rtau)},
        'weights': {'wq': cfg.wq, 'wdq': cfg.wdq, 'wtau': cfg.wtau}
    }

    tag = f"K({cfg.Kq},{cfg.Kdq},{cfg.Ktau})_r({cfg.rq},{cfg.rdq},{cfg.rtau})_f({cfg.q_filter}|{cfg.dq_filter}|{cfg.tau_filter})"
    os.makedirs(cfg.out, exist_ok=True)
    save_json(res, os.path.join(cfg.out, f"result_MHPCA_{tag}.json"))

    if cfg.save_mapping:
        np.savez_compressed(
            os.path.join(cfg.out, f"mapping_MHPCA_{tag}.npz"),
            mq=mq, sq=sq, md=md, sd=sd, mt=mt, st=st,
            Wq=pca_q.components_.astype(np.float32), mu_q=pca_q.mean_.astype(np.float32),
            Wdq=pca_dq.components_.astype(np.float32), mu_dq=pca_dq.mean_.astype(np.float32),
            Wt=pca_t.components_.astype(np.float32),  mu_t=pca_t.mean_.astype(np.float32),
            Kq=np.int32(cfg.Kq), Kdq=np.int32(cfg.Kdq), Ktau=np.int32(cfg.Ktau)
        )

    print('Done.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--out',  type=str, required=True)
    # filters per group
    ap.add_argument('--q-filter',   type=str, default='none')
    ap.add_argument('--dq-filter',  type=str, default='none')
    ap.add_argument('--tau-filter', type=str, default='none')
    # DCT K per group
    ap.add_argument('--Kq',   type=int, default=96)
    ap.add_argument('--Kdq',  type=int, default=64)
    ap.add_argument('--Ktau', type=int, default=64)
    # latent per group
    ap.add_argument('--rq',   type=int, default=12)
    ap.add_argument('--rdq',  type=int, default=2)
    ap.add_argument('--rtau', type=int, default=2)
    # weights for WRMSE reporting
    ap.add_argument('--wq',   type=float, default=1.0)
    ap.add_argument('--wdq',  type=float, default=0.25)
    ap.add_argument('--wtau', type=float, default=0.1)
    ap.add_argument('--save-mapping', dest='save_mapping', action='store_true')
    cfg = ap.parse_args()
    os.makedirs(cfg.out, exist_ok=True)
    run(cfg)

if __name__ == '__main__':
    main()
```

---

### sweep_multigroup_pca.py（新）
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, itertools, json
import numpy as np
from glob import glob
from compress_multigroup_pca import main as run_once  # reuse main via CLI-style args? We'll call as a subprocess-like
import subprocess, sys
import orjson

"""
网格搜索：对 (Kq,Kdq,Ktau) × (rq,rdq,rtau) × filters 做 sweep。
为简化复用，内部通过 subprocess 调用 compress_multigroup_pca.py。
"""

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def parse_triplet_list(items):
    # items like ["400,50,50", "360,70,70"] -> [(400,50,50),...]
    out = []
    for s in items:
        a,b,c = s.split(',')
        out.append((int(a), int(b), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--out',  type=str, required=True)
    # lists
    ap.add_argument('--Kq',   type=int, nargs='+', default=[256,384,512])
    ap.add_argument('--Kdq',  type=int, nargs='+', default=[128,192])
    ap.add_argument('--Ktau', type=int, nargs='+', default=[128,192])
    ap.add_argument('--latent-triplets', nargs='+', default=['400,50,50','360,70,70','320,90,90'])
    ap.add_argument('--filters', nargs='+', default=['none'])  # or 'ema:9', 'sg:21,3'
    ap.add_argument('--wq', type=float, default=1.0)
    ap.add_argument('--wdq', type=float, default=0.25)
    ap.add_argument('--wtau',type=float, default=0.1)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    triplets = parse_triplet_list(args.latent_triplets)

    jobs = list(itertools.product(args.Kq, args.Kdq, args.Ktau, triplets, args.filters))

    results = []
    for Kq, Kdq, Kt, (rq, rdq, rtau), filt in jobs:
        tag = f"K({Kq},{Kdq},{Kt})_r({rq},{rdq},{rtau})_f({filt})"
        out_dir = args.out
        cmd = [
            sys.executable, 'compress_multigroup_pca.py',
            '--data', args.data, '--out', out_dir,
            '--Kq', str(Kq), '--Kdq', str(Kdq), '--Ktau', str(Kt),
            '--rq', str(rq), '--rdq', str(rdq), '--rtau', str(rtau),
            '--q-filter', filt, '--dq-filter', filt, '--tau-filter', filt,
            '--wq', str(args.wq), '--wdq', str(args.wdq), '--wtau', str(args.wtau)
        ]
        print('Running', tag)
        subprocess.run(cmd, check=True)

    # aggregate
    for fp in glob(os.path.join(args.out, 'result_MHPCA_*.json')):
        with open(fp, 'rb') as f:
            results.append(orjson.loads(f.read()))

    # sort by WRMSE then RMSE_q
    results.sort(key=lambda r: (r['metrics']['wrmse'], r['metrics']['rmse_q']))
    save_json(results, os.path.join(args.out, 'results_summary_multigroup.json'))
    print('Saved summary to', os.path.join(args.out, 'results_summary_multigroup.json'))

if __name__ == '__main__':
    main()
```

---

### run_sweep_multigroup.sh（新）
```bash
#!/usr/bin/env bash
set -e
DATA=${1:-/path/to/your_data.npz}
OUT=${2:-./out_mhpca_sweep}

python3 sweep_multigroup_pca.py \
  --data "$DATA" --out "$OUT" \
  --Kq 256 384 512 \
  --Kdq 128 192 \
  --Ktau 128 192 \
  --latent-triplets "400,50,50" "360,70,70" "320,90,90" \
  --filters none \
  --wq 1.0 --wdq 0.25 --wtau 0.1

echo "Summary: $OUT/results_summary_multigroup.json"
```

**建议的首轮高保真 Sweep（总 latent ≤ 500）**
- `Kq ∈ {256,384,512}`，`Kdq, Ktau ∈ {128,192}`（给 q 更高时间带宽）
- 潜维三元组：`(400,50,50)`, `(360,70,70)`, `(320,90,90)`（总维=500）
- 预滤波：先用 `none`（为了“尽可能重建”不抢频带）；若噪声大再加 `ema:9` 或 `sg:21,3` 试验

> 目标：**最大化重建**，尤其压低 `RMSE_q`；在你允许 FM 500 维的前提下，这套网格应显著优于单头 PCA 的 q 精度。必要时可把 `Kq` 提到 768/1024 再跑一轮。

