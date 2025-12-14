# 说明
这是一套可跑的基准，用 **B-Spline（样条）** 对 `q_log`（可选对 `dq_log`、`tau_log`）做 **encode→decode**，遍历不同**控制点个数 M（密度）**与**样条阶数 p**（厚度/次数），并画出 **参数量 vs RMSE** 的曲线。全流程确定、可复现：
- 统一把每条轨迹时间归一到 `[0,1]`；
- 使用 **开区间均匀结**（open-uniform）B-Spline；
- 编码 = 岭回归 (Tikhonov)：`θ = (BᵀB + λI)⁻¹ Bᵀ y`，每关节相同 `B` 与 `P=(BᵀB+λI)⁻¹Bᵀ` 预计算重用；
- 解码 = `ŷ = B θ`；
- 可选对 `dq` 用 **解析导数**（`B' θ`）评估 `RMSE_dq`；
- 可选对 `tau` 也做一套样条（或跳过）。

默认 **只对 q 做样条** 并测 `RMSE_q` 与可选 `RMSE_dq`；如需对 `dq`/`tau` 也做样条，可加开关。

---

## requirements.txt
```txt
numpy>=1.22
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
orjson>=3.9
tqdm>=4.66
```

---

## spline_utils.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import BSpline
from typing import Tuple

# --------- 时间与结向量 ---------

def normalized_times(T: int) -> np.ndarray:
    # 等距采样到 [0,1]
    if T <= 1:
        return np.array([0.0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    return t


def open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    # 开区间均匀结：首尾重复 degree+1 次，中间均匀
    # 节点个数 = n_ctrl + degree + 1
    k = degree
    m = n_ctrl + k + 1
    knots = np.zeros(m, dtype=np.float64)
    knots[k:m-k] = np.linspace(0.0, 1.0, m - 2*k, dtype=np.float64)
    knots[m-k:] = 1.0
    return knots

# --------- 设计矩阵 B 与导数 Bdot ---------

def bspline_design_matrix(Ts: np.ndarray, n_ctrl: int, degree: int) -> np.ndarray:
    """构造 B 设计矩阵，形状 (T, n_ctrl)。每列是一个基函数在 Ts 上的取值。"""
    knots = open_uniform_knots(n_ctrl, degree)
    T = Ts.shape[0]
    B = np.zeros((T, n_ctrl), dtype=np.float64)
    # 逐基列构造：把 coeff 设为 one-hot，调用 BSpline 求值
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=np.float64)
        coeff[j] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, j] = spl(Ts)
    return B


def bspline_deriv_design_matrix(Ts: np.ndarray, n_ctrl: int, degree: int, order: int = 1) -> np.ndarray:
    """构造导数设计矩阵 B'（或更高阶），形状 (T, n_ctrl)。"""
    knots = open_uniform_knots(n_ctrl, degree)
    T = Ts.shape[0]
    Bd = np.zeros((T, n_ctrl), dtype=np.float64)
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=np.float64)
        coeff[j] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False).derivative(nu=order)
        Bd[:, j] = spl(Ts)
    return Bd

# --------- 编码/解码 ---------

def ridge_pinv(B: np.ndarray, lam: float) -> np.ndarray:
    """返回 P = (B^T B + lam I)^{-1} B^T; 用于 θ = P @ y"""
    T, M = B.shape
    BtB = B.T @ B
    if lam > 0:
        BtB = BtB + lam * np.eye(M, dtype=np.float64)
    P = np.linalg.solve(BtB, B.T)  # (M,M)×(M,T) -> (M,T)
    return P


def encode_theta(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y: (T, J) -> θ: (M, J)"""
    return P @ y  # (M,T)×(T,J) = (M,J)


def decode_signal(B: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """B: (T,M), θ: (M,J) -> yhat: (T,J)"""
    return B @ theta
```

---

## spline_benchmark.py（主脚本：网格 sweep + 画图）
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, itertools
import numpy as np
import orjson
from tqdm import tqdm
import matplotlib.pyplot as plt

from spline_utils import (
    normalized_times, bspline_design_matrix, bspline_deriv_design_matrix,
    ridge_pinv, encode_theta, decode_signal
)

# ----------------- IO -----------------

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_dataset(npz_path):
    data = np.load(npz_path)
    q   = data['q_log'].astype(np.float64)    # (N,T,7)
    dq  = data.get('dq_log', None)
    tau = data.get('tau_log', None)
    if dq is not None:
        dq = dq.astype(np.float64)
    if tau is not None:
        tau = tau.astype(np.float64)
    return q, dq, tau

# ----------------- Metrics -----------------

def rmse(A,B):
    return float(np.sqrt(np.mean((A-B)**2)))

# ----------------- Benchmark -----------------

def run_once(q, dq, tau, deg_q, Mq, lam_q, eval_dq=False, fit_dq=False, deg_dq=3, Mdq=32, lam_dq=1e-6,
             fit_tau=False, deg_tau=3, Mtau=32, lam_tau=1e-6):
    N, T, J = q.shape
    t = normalized_times(T)

    # q: basis & pinv
    Bq = bspline_design_matrix(t, Mq, deg_q)         # (T, Mq)
    Pq = ridge_pinv(Bq, lam_q)                       # (Mq, T)

    # optional derivative basis for dq evaluation
    Bq_dot = bspline_deriv_design_matrix(t, Mq, deg_q, order=1) if eval_dq else None

    # encode all samples/joints
    rmse_q_list = []
    rmse_dq_list = []

    for i in range(N):
        # θ_q: (Mq,7)
        theta_q = encode_theta(Pq, q[i])
        q_hat = decode_signal(Bq, theta_q)  # (T,7)
        rmse_q_list.append(rmse(q[i], q_hat))

        if eval_dq and dq is not None:
            dq_hat = decode_signal(Bq_dot, theta_q)
            rmse_dq_list.append(rmse(dq[i], dq_hat))

    metrics = {
        'rmse_q': float(np.mean(rmse_q_list)),
    }
    if eval_dq and dq is not None:
        metrics['rmse_dq_from_q'] = float(np.mean(rmse_dq_list))

    # optionally: fit dq, tau each with their own spline (rarely需要)
    if fit_dq and dq is not None:
        Bdq = bspline_design_matrix(t, Mdq, deg_dq)
        Pdq = ridge_pinv(Bdq, lam_dq)
        rmse_dq2 = []
        for i in range(N):
            theta_dq = encode_theta(Pdq, dq[i])
            dq_hat2 = decode_signal(Bdq, theta_dq)
            rmse_dq2.append(rmse(dq[i], dq_hat2))
        metrics['rmse_dq_direct'] = float(np.mean(rmse_dq2))

    if fit_tau and tau is not None:
        Btau = bspline_design_matrix(t, Mtau, deg_tau)
        Ptau = ridge_pinv(Btau, lam_tau)
        rmse_tau_list = []
        for i in range(N):
            theta_tau = encode_theta(Ptau, tau[i])
            tau_hat = decode_signal(Btau, theta_tau)
            rmse_tau_list.append(rmse(tau[i], tau_hat))
        metrics['rmse_tau'] = float(np.mean(rmse_tau_list))

    # parameter counts
    params = {
        'q': int(J*Mq),
        'dq': int(J*Mdq) if fit_dq else 0,
        'tau': int(J*Mtau) if fit_tau else 0,
    }
    metrics['param_total'] = params['q'] + params['dq'] + params['tau']

    return metrics, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)

    # sweep ranges
    ap.add_argument('--deg-q', type=int, nargs='+', default=[3,4,5])
    ap.add_argument('--Mq', type=int, nargs='+', default=[16,32,48,64,96,128,160])
    ap.add_argument('--lam-q', type=float, default=1e-6)

    # whether to evaluate dq from q's derivative
    ap.add_argument('--eval-dq', action='store_true')

    # optionally fit dq/tau with their own splines
    ap.add_argument('--fit-dq', action='store_true')
    ap.add_argument('--deg-dq', type=int, nargs='+', default=[3])
    ap.add_argument('--Mdq', type=int, nargs='+', default=[16,32,48,64])
    ap.add_argument('--lam-dq', type=float, default=1e-6)

    ap.add_argument('--fit-tau', action='store_true')
    ap.add_argument('--deg-tau', type=int, nargs='+', default=[3])
    ap.add_argument('--Mtau', type=int, nargs='+', default=[16,32,48,64])
    ap.add_argument('--lam-tau', type=float, default=1e-6)

    ap.add_argument('--plot', action='store_true', help='画 参数量 vs RMSE 曲线')

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    q, dq, tau = load_dataset(args.data)

    results = []

    # sweep q (必选)
    for deg_q in args.deg_q:
        for Mq in args.Mq:
            # 如果还要扫 dq/tau 的自己参数就再套一层产品，但默认不扫，避免爆炸
            deg_dq_list = args.deg_dq if args.fit_dq else [None]
            Mdq_list = args.Mdq if args.fit_dq else [None]
            deg_tau_list = args.deg_tau if args.fit_tau else [None]
            Mtau_list = args.Mtau if args.fit_tau else [None]

            for deg_dq in deg_dq_list:
                for Mdq in Mdq_list:
                    for deg_tau in deg_tau_list:
                        for Mtau in Mtau_list:
                            metrics, params = run_once(
                                q, dq, tau,
                                deg_q=deg_q, Mq=Mq, lam_q=args.lam_q,
                                eval_dq=args.eval_dq,
                                fit_dq=args.fit_dq and (deg_dq is not None) and (Mdq is not None),
                                deg_dq=deg_dq if deg_dq is not None else 3,
                                Mdq=Mdq if Mdq is not None else 32,
                                lam_dq=args.lam_dq,
                                fit_tau=args.fit_tau and (deg_tau is not None) and (Mtau is not None),
                                deg_tau=deg_tau if deg_tau is not None else 3,
                                Mtau=Mtau if Mtau is not None else 32,
                                lam_tau=args.lam_tau,
                            )

                            rec = {
                                'deg_q': deg_q,
                                'Mq': Mq,
                                'lam_q': args.lam_q,
                                'metrics': metrics,
                                'params': params,
                            }
                            if args.fit_dq:
                                rec.update({'deg_dq': deg_dq, 'Mdq': Mdq, 'lam_dq': args.lam_dq})
                            if args.fit_tau:
                                rec.update({'deg_tau': deg_tau, 'Mtau': Mtau, 'lam_tau': args.lam_tau})
                            results.append(rec)

    # 保存汇总
    save_json(results, os.path.join(args.out, 'results_spline_summary.json'))

    # 画图：参数量 vs RMSE（支持多条：rmse_q、rmse_dq_from_q、rmse_dq_direct、rmse_tau）
    if args.plot:
        # 收集 x=param_total, y=各 RMSE
        xs = np.array([r['metrics']['param_total'] for r in results], dtype=float)
        # 用不同样式区分 deg_q
        degs = sorted(set(r['deg_q'] for r in results))

        plt.figure(figsize=(9,6))
        for d in degs:
            idx = [i for i,r in enumerate(results) if r['deg_q']==d]
            x = xs[idx]
            y = np.array([results[i]['metrics']['rmse_q'] for i in idx], dtype=float)
            order = np.argsort(x)
            plt.plot(x[order], y[order], marker='o', label=f'q: degree {d}')
        plt.xlabel('Total parameters (latent dims)')
        plt.ylabel('RMSE_q')
        plt.title('Parameters vs RMSE_q (B-Spline encode-decode)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'curve_params_vs_rmse_q.png'), dpi=160)

        # 可选：dq from q
        if any('rmse_dq_from_q' in r['metrics'] for r in results):
            plt.figure(figsize=(9,6))
            for d in degs:
                idx = [i for i,r in enumerate(results) if r['deg_q']==d and 'rmse_dq_from_q' in r['metrics']]
                if not idx:
                    continue
                x = xs[idx]
                y = np.array([results[i]['metrics']['rmse_dq_from_q'] for i in idx], dtype=float)
                order = np.argsort(x)
                plt.plot(x[order], y[order], marker='o', label=f'dq (from q), degree {d}')
            plt.xlabel('Total parameters (latent dims)')
            plt.ylabel('RMSE_dq (from q)')
            plt.title('Parameters vs RMSE_dq_from_q')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'curve_params_vs_rmse_dq_from_q.png'), dpi=160)

        # 可选：dq、tau 直接拟合
        if any('rmse_dq_direct' in r['metrics'] for r in results):
            plt.figure(figsize=(9,6))
            x = np.array([r['metrics']['param_total'] for r in results if 'rmse_dq_direct' in r['metrics']], dtype=float)
            y = np.array([r['metrics']['rmse_dq_direct'] for r in results if 'rmse_dq_direct' in r['metrics']], dtype=float)
            order = np.argsort(x)
            plt.plot(x[order], y[order], marker='o')
            plt.xlabel('Total parameters (latent dims)')
            plt.ylabel('RMSE_dq (direct)')
            plt.title('Parameters vs RMSE_dq (direct spline)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'curve_params_vs_rmse_dq_direct.png'), dpi=160)

        if any('rmse_tau' in r['metrics'] for r in results):
            plt.figure(figsize=(9,6))
            x = np.array([r['metrics']['param_total'] for r in results if 'rmse_tau' in r['metrics']], dtype=float)
            y = np.array([r['metrics']['rmse_tau'] for r in results if 'rmse_tau' in r['metrics']], dtype=float)
            order = np.argsort(x)
            plt.plot(x[order], y[order], marker='o')
            plt.xlabel('Total parameters (latent dims)')
            plt.ylabel('RMSE_tau')
            plt.title('Parameters vs RMSE_tau (direct spline)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'curve_params_vs_rmse_tau.png'), dpi=160)

        print('Saved plots to', args.out)

if __name__ == '__main__':
    main()
```

---

## run_spline_sweep.sh（一键跑 & 画图）
```bash
#!/usr/bin/env bash
set -e
DATA=${1:-/path/to/your_data.npz}
OUT=${2:-./out_spline_$(date +%F_%H%M%S)}

# 基线：仅对 q 做样条，评估 q 与 dq(from q)，扫描 p 与 M。
python3 spline_benchmark.py \
  --data "$DATA" \
  --out "$OUT" \
  --deg-q 3 4 5 \
  --Mq 16 32 48 64 96 128 160 \
  --lam-q 1e-6 \
  --eval-dq \
  --plot

echo "Summary: $OUT/results_spline_summary.json"
echo "Curves:  $OUT/curve_params_vs_rmse_q.png (and *_dq_*.png if enabled)"
```

---

## 说明与小贴士
- **参数量（latent）**：若只对 `q` 做样条，`param_total = 7×Mq`。若对 `dq`/`tau` 也拟合，额外 + `7×Mdq`、`7×Mtau`。
- **度数 p**：`3/4/5`（三/四/五次）。五次更平滑、拟合更强但可能略过拟合；建议先看 `p=3,5` 的对比曲线。
- **正则 λ**：默认 `1e-6`；若高频抖动，可增至 `1e-4`、`1e-3` 再复跑。
- **速度（dq）评估**：如果勾选 `--eval-dq`，`dq` 是通过 `B'θ` 解析求导，不额外增加 latent；也可以 `--fit-dq` 单独拟合。
- **时间复杂度**：每个 (p,M) 只需构造一次 `B` 与 `P`，然后对 N=2040 条样本批量乘法即可。`M≤160` 时在 3090Ti/CPU 都很稳。
- **可扩展**：若你要“固定总 latent ≈ 1000”，可以把 `Mq` 值限制在 `~140`（`7×140=980`）；或者改为按能量自适应每关节的控制点数（可加权分配函数）。

