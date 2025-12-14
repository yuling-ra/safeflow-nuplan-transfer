#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-Spline 压缩工具模块
提供样条基函数构造、编码/解码功能
"""
import numpy as np
from scipy.interpolate import BSpline
from typing import Tuple

# --------- 时间与结向量 ---------

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

# --------- 设计矩阵 B 与导数 Bdot ---------

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

# --------- 编码/解码 ---------

def ridge_pinv(B: np.ndarray, lam: float) -> np.ndarray:
    """
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
