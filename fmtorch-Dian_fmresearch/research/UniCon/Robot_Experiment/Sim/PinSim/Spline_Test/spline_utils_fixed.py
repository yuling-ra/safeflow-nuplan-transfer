#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的B-Spline压缩工具模块 - 修复边界问题
增加边界权重以改善第一帧和最后一帧的重建质量
"""
import numpy as np
from scipy.interpolate import BSpline
from typing import Tuple


def normalized_times(T: int) -> np.ndarray:
    """等距采样到 [0,1]"""
    if T <= 1:
        return np.array([0.0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    return t


def open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """
    开区间均匀结：首尾重复 degree+1 次，中间均匀
    节点个数 = n_ctrl + degree + 1
    """
    k = degree
    m = n_ctrl + k + 1
    knots = np.zeros(m, dtype=np.float64)
    knots[k:m-k] = np.linspace(0.0, 1.0, m - 2*k, dtype=np.float64)
    knots[m-k:] = 1.0
    return knots


def bspline_design_matrix(Ts: np.ndarray, n_ctrl: int, degree: int) -> np.ndarray:
    """
    构造 B 设计矩阵，形状 (T, n_ctrl)。
    每列是一个基函数在 Ts 上的取值。
    """
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
    """
    构造导数设计矩阵 B'（或更高阶），形状 (T, n_ctrl)。
    """
    knots = open_uniform_knots(n_ctrl, degree)
    T = Ts.shape[0]
    Bd = np.zeros((T, n_ctrl), dtype=np.float64)
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=np.float64)
        coeff[j] = 1.0
        spl = BSpline(knots, coeff, degree, extrapolate=False).derivative(nu=order)
        Bd[:, j] = spl(Ts)
    return Bd


def ridge_pinv_with_boundary_weights(B: np.ndarray, lam: float, 
                                     boundary_weight: float = 10.0,
                                     boundary_frames: int = 10) -> np.ndarray:
    """
    返回加权的伪逆矩阵，给边界帧更高的权重
    
    公式: P = (B^T W B + lam I)^{-1} B^T W
    其中 W 是对角权重矩阵，边界帧权重更高
    
    参数:
    - B: 设计矩阵 (T, M)
    - lam: Ridge正则化参数
    - boundary_weight: 边界帧的权重倍数（相对于内部帧）
    - boundary_frames: 边界区域的帧数（前N帧和后N帧）
    
    返回:
    - P: 加权伪逆矩阵 (M, T)
    """
    T, M = B.shape
    
    # 构造权重向量（对角矩阵的对角元素）
    weights = np.ones(T, dtype=np.float64)
    
    # 前boundary_frames帧和后boundary_frames帧给更高权重
    # 使用渐变权重：离边界越近，权重越高
    for i in range(boundary_frames):
        # 线性衰减：从boundary_weight到1.0
        fade_factor = (boundary_frames - i) / boundary_frames
        weight = 1.0 + (boundary_weight - 1.0) * fade_factor
        
        weights[i] = weight  # 前边界
        weights[-(i+1)] = weight  # 后边界
    
    # 构造加权矩阵 B^T W B
    W_sqrt = np.sqrt(weights)  # 权重的平方根
    BW = B * W_sqrt[:, None]  # 广播：每行乘以对应权重
    
    BtWB = BW.T @ BW  # B^T W B
    
    if lam > 0:
        BtWB = BtWB + lam * np.eye(M, dtype=np.float64)
    
    # B^T W = (B^T W^{1/2}) @ W^{1/2}
    BtW = BW.T * W_sqrt[None, :]
    
    # 求解 (B^T W B + lam I)^{-1} B^T W
    P = np.linalg.solve(BtWB, BtW)  # (M, T)
    
    return P


def ridge_pinv(B: np.ndarray, lam: float) -> np.ndarray:
    """
    标准Ridge伪逆（向后兼容）
    返回 P = (B^T B + lam I)^{-1} B^T; 用于 θ = P @ y
    """
    T, M = B.shape
    BtB = B.T @ B
    if lam > 0:
        BtB = BtB + lam * np.eye(M, dtype=np.float64)
    P = np.linalg.solve(BtB, B.T)  # (M,M)×(M,T) -> (M,T)
    return P


def encode_theta(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    y: (T, J) -> θ: (M, J)
    """
    return P @ y  # (M,T)×(T,J) = (M,J)


def decode_signal(B: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    B: (T,M), θ: (M,J) -> yhat: (T,J)
    """
    return B @ theta


def compute_boundary_weights(T: int, boundary_weight: float = 10.0, 
                            boundary_frames: int = 10) -> np.ndarray:
    """
    计算边界权重向量（用于可视化和调试）
    
    返回:
    - weights: 形状 (T,) 的权重数组
    """
    weights = np.ones(T, dtype=np.float64)
    
    for i in range(boundary_frames):
        fade_factor = (boundary_frames - i) / boundary_frames
        weight = 1.0 + (boundary_weight - 1.0) * fade_factor
        weights[i] = weight
        weights[-(i+1)] = weight
    
    return weights











