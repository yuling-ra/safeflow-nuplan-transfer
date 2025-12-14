#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数：数据加载、标准化、评估指标、IO
"""
import os
import numpy as np
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
    """应用 z-score 标准化"""
    return (X - mean[None, None, :]) / std[None, None, :]


def zscore_inv(Xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """逆 z-score 标准化"""
    return Xn * std[None, None, :] + mean[None, None, :]


# ------------------------
# 评估指标
# ------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """均方误差"""
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(mse(a, b)))


def group_rmse(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
               q_rec: np.ndarray, dq_rec: np.ndarray, tau_rec: np.ndarray) -> Dict[str, float]:
    """
    计算每组的 RMSE
    """
    return {
        'rmse_q': rmse(q, q_rec),
        'rmse_dq': rmse(dq, dq_rec),
        'rmse_tau': rmse(tau, tau_rec),
    }


def weighted_rmse(q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                  q_rec: np.ndarray, dq_rec: np.ndarray, tau_rec: np.ndarray,
                  wq: float = 1.0, wdq: float = 0.25, wtau: float = 0.1) -> float:
    """
    加权 RMSE
    """
    e_q   = np.mean((q - q_rec) ** 2)
    e_dq  = np.mean((dq - dq_rec) ** 2)
    e_tau = np.mean((tau - tau_rec) ** 2)
    w_mse = wq*e_q + wdq*e_dq + wtau*e_tau
    return float(np.sqrt(w_mse))


# ------------------------
# IO 工具
# ------------------------

def save_json(obj, path: str):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        import orjson
        with open(path, 'wb') as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
    except ImportError:
        # Fallback to standard json
        import json
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
    
    return {
        'q': q,      # (N, T, 7)
        'dq': dq,    # (N, T, 7)
        'tau': tau,  # (N, T, 7)
        'N': q.shape[0],
        'T': q.shape[1],
        'J': q.shape[2],
    }
