# dmpc_fm_cbf/cbf_project_velocity_sgfm.py
# ===============================================================
# Self-contained SGFM-CBF projector (XY only)
# - Avoids dependency on cbf_project_velocity_sgfm_.py
# - Projects v_nom to satisfy barrier constraint against circles
# - Works with sampler_5h.py signature: x_norm_1x2T / v_nom_1x2T
# ===============================================================

from __future__ import annotations
from typing import Optional, List, Dict
import torch


@torch.no_grad()
def cbf_project_velocity_sgfm(
    *,
    x_norm_1x2N: Optional[torch.Tensor] = None,
    v_nom_1x2N: Optional[torch.Tensor] = None,
    x_norm_1x2T: Optional[torch.Tensor] = None,
    v_nom_1x2T: Optional[torch.Tensor] = None,
    scaler=None,  # not used (we operate in normalized/canonical XY directly)
    ellipses: Optional[List[Dict]] = None,
    gate_scalar: float = 1.0,
    gate_lf: float = 1.0,
    gate_hf: float = 1.0,
    passes: int = 10,
    extra_margin: float = 0.10,
    max_corr_norm: float = 2.0,
    alpha: float = 8.0,            # CBF gain
    eps: float = 1e-9,
    **_,
):
    """
    Inputs:
      x_norm_1x2T, v_nom_1x2T: (1,2,T)  OR  x_norm_1x2N, v_nom_1x2N: (1,2,N)
      ellipses: [{"center": (2,), "radius": float}, ...]  (circle obstacles)

    Output:
      v_safe: same shape as v_nom, projected to satisfy CBF approximately.

    Constraint (for each obstacle, each time index):
      h(p) = ||p-c||^2 - (r+m)^2
      grad h = 2(p-c)
      enforce: grad h · v + alpha * h >= 0
    We compute minimal correction along grad direction (closed form),
    and iterate 'passes' times for multiple obstacles.
    """
    x = x_norm_1x2T if x_norm_1x2T is not None else x_norm_1x2N
    v = v_nom_1x2T if v_nom_1x2T is not None else v_nom_1x2N
    if x is None or v is None:
        raise ValueError("Provide either (x_norm_1x2T, v_nom_1x2T) or (x_norm_1x2N, v_nom_1x2N).")

    ellipses = ellipses or []
    gate = float(max(gate_scalar, gate_lf, gate_hf))
    if gate <= 0.0 or len(ellipses) == 0:
        return v

    # shapes: (1,2,T)
    assert x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 2, f"x must be (1,2,T). got {tuple(x.shape)}"
    assert v.shape == x.shape, f"v must match x. got v={tuple(v.shape)} x={tuple(x.shape)}"

    v_safe = v.clone()

    # iterative projection for multiple obstacles / stronger effect
    for _ in range(int(max(1, passes))):
        corr = torch.zeros_like(v_safe)

        for e in ellipses:
            c = torch.as_tensor(e["center"], device=x.device, dtype=x.dtype).view(1, 2, 1)   # (1,2,1)
            r = float(e.get("radius", e.get("r", 0.3)))
            rm = r + float(extra_margin)

            d = x - c                              # (1,2,T)
            h = (d * d).sum(dim=1, keepdim=True) - (rm * rm)    # (1,1,T)
            grad = 2.0 * d                         # (1,2,T)

            # a = grad^T v + alpha h
            a = (grad * v_safe).sum(dim=1, keepdim=True) + float(alpha) * h  # (1,1,T)

            # If a >= 0 -> safe; else push along grad
            # minimal delta: delta = (-a) * grad / (||grad||^2)
            g2 = (grad * grad).sum(dim=1, keepdim=True).clamp_min(float(eps))  # (1,1,T)
            lam = (-a).clamp_min(0.0) / g2                                     # (1,1,T)
            delta = lam * grad                                                 # (1,2,T)

            corr = corr + delta

        # cap correction norm per time index (avoid explosions)
        corr_norm = torch.linalg.norm(corr, dim=1, keepdim=True).clamp_min(float(eps))  # (1,1,T)
        cap = float(max_corr_norm)
        scale = torch.clamp(cap / corr_norm, max=1.0)
        corr = corr * scale

        v_safe = v_safe + corr

    # gate (match sampler’s scheduling)
    return v + gate * (v_safe - v)
