#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选的时域预滤波器
"""
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def apply_filter(X: np.ndarray, spec: str) -> np.ndarray:
    """
    Apply temporal filter along axis=1. spec formats:
    - 'none'
    - 'ema:win'          e.g., 'ema:9'
    - 'sg:win,poly'      e.g., 'sg:21,3' (win must be odd)
    - 'butter:cut,ord'   e.g., 'butter:0.05,4' (cut is 0..1 Nyquist fraction)
    
    X: (N, T, J) 数组
    返回: 滤波后的 (N, T, J) 数组
    """
    if not spec or spec == 'none':
        return X
    
    kind = spec.split(':')[0]
    args = spec.split(':')[1] if ':' in spec else ''
    
    if kind == 'ema':
        return _apply_ema(X, args)
    elif kind == 'sg':
        return _apply_savgol(X, args)
    elif kind == 'butter':
        return _apply_butterworth(X, args)
    else:
        print(f"[warn] Unknown filter '{spec}', returning original data")
        return X


def _apply_ema(X: np.ndarray, args: str) -> np.ndarray:
    """EMA with forward-backward pass for zero-phase effect"""
    win = int(args)
    alpha = 2.0 / (win + 1.0)
    Y = np.copy(X)
    
    # Forward pass
    for c in range(X.shape[2]):
        for n in range(X.shape[0]):
            s = X[n, 0, c]
            for t in range(X.shape[1]):
                s = alpha * X[n, t, c] + (1 - alpha) * s
                Y[n, t, c] = s
    
    # Backward pass
    Xb = Y[:, ::-1, :].copy()
    Yb = np.copy(Xb)
    for c in range(Xb.shape[2]):
        for n in range(Xb.shape[0]):
            s = Xb[n, 0, c]
            for t in range(Xb.shape[1]):
                s = alpha * Xb[n, t, c] + (1 - alpha) * s
                Yb[n, t, c] = s
    
    Y = Yb[:, ::-1, :]
    return Y.astype(np.float32)


def _apply_savgol(X: np.ndarray, args: str) -> np.ndarray:
    """Savitzky-Golay filter"""
    parts = args.split(',')
    win = int(parts[0])
    poly = int(parts[1])
    
    if win % 2 == 0:
        win += 1  # Must be odd
        print(f"[warn] SG window adjusted to {win} (must be odd)")
    
    Y = savgol_filter(X, window_length=win, polyorder=poly, axis=1, mode='interp')
    return Y.astype(np.float32)


def _apply_butterworth(X: np.ndarray, args: str) -> np.ndarray:
    """Butterworth low-pass filter with zero-phase (filtfilt)"""
    parts = args.split(',')
    cut = float(parts[0])
    order = int(parts[1]) if len(parts) > 1 else 4
    
    b, a = butter(order, cut, btype='low', fs=None, analog=False)
    
    # Zero-phase filtering per series
    Y = np.empty_like(X)
    for c in range(X.shape[2]):
        for n in range(X.shape[0]):
            Y[n, :, c] = filtfilt(b, a, X[n, :, c], axis=0, method='gust')
    
    return Y.astype(np.float32)


