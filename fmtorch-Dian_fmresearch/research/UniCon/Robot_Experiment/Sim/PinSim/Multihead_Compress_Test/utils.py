#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, math
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


def weighted_rmse(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                  q_rec: np.ndarray, dq_rec: np.ndarray, tau_rec: np.ndarray,
                  wq=1.0, wdq=0.25, wtau=0.1) -> float:
    e_q   = np.mean((q - q_rec) ** 2)
    e_dq  = np.mean((dq - dq_rec) ** 2)
    e_tau = np.mean((tau - tau_rec) ** 2)
    w_mse = wq*e_q + wdq*e_dq + wtau*e_tau
    return float(np.sqrt(w_mse))

# ------------------------
# IO 工具
# ------------------------

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        import orjson
        with open(path, 'wb') as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
    except ImportError:
        # Fallback to standard json
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)


def load_dataset(npz_path: str, max_samples: int = None) -> Dict[str, np.ndarray]:
    """
    加载 test_complete.npz 格式的数据
    max_samples: 如果指定，只加载前 N 个样本（用于快速测试）
    """
    data = np.load(npz_path)
    # 期望键: q_log, dq_log, tau_log
    q  = data['q_log'].astype(np.float32)
    dq = data['dq_log'].astype(np.float32)
    tau= data['tau_log'].astype(np.float32)
    
    # 如果指定了 max_samples，只取前 N 个
    if max_samples is not None and max_samples > 0:
        q = q[:max_samples]
        dq = dq[:max_samples]
        tau = tau[:max_samples]
    
    assert q.shape == dq.shape == tau.shape, "Shape mismatch!"
    N, T, J = q.shape
    
    return {
        'q': q,     # (N, T, 7)
        'dq': dq,   # (N, T, 7)
        'tau': tau, # (N, T, 7)
        'N': N,
        'T': T,
        'J': J,
    }


