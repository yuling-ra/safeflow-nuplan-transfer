# 
"""
DMPC + Flow Matching + CBF Framework for Multi-Agent Trajectory Planning

NOTE:
This package is under active development, so we keep __init__.py SAFE:
- Never hard-import optional modules that might not exist (e.g. models.py).
- Prefer explicit imports from submodules in notebooks/scripts.

Recommended usage:
    from dmpc_fm_cbf import sampler_5h
    from dmpc_fm_cbf.cbf_project_velocity_sgfm import cbf_project_velocity_sgfm
    from dmpc_fm_cbf.rollout_static import run_rollout_static
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------
# SAFE re-exports (only if the symbol really exists)
# ---------------------------------------------------------------------
__all__ = ["__version__"]

def _safe_export(module_path: str, names: list[str]):
    """
    Try to import `names` from module_path and add them to globals()/__all__.
    If anything fails, do nothing (keeps package importable).
    """
    try:
        mod = __import__(module_path, fromlist=names)
        for n in names:
            if hasattr(mod, n):
                globals()[n] = getattr(mod, n)
                __all__.append(n)
    except Exception:
        # swallow all import errors to keep package importable
        pass


# Core modules that DO exist in your tree
_safe_export("dmpc_fm_cbf.cbf", ["cbf_project_velocity_world", "ellipse_h_grad_batch"])
_safe_export("dmpc_fm_cbf.canonical", ["canonicalize", "uncanonicalize", "rot2"])
_safe_export("dmpc_fm_cbf.ode_solver", ["ODESolver", "FMODEFunc"])

# Optional / may or may not exist by name inside your repo
# (keep them SAFE — if not found, nothing breaks)
_safe_export("dmpc_fm_cbf.training", ["FlowMatchingTrainer", "TrainConfig"])
_safe_export("dmpc_fm_cbf.dataset", ["make_center_to_ring_dataset_tracking_oc", "add_velocity_fields"])
_safe_export("dmpc_fm_cbf.utils", ["set_seed", "get_device"])
_safe_export("dmpc_fm_cbf.cbf_project_velocity_sgfm", ["cbf_project_velocity_sgfm"])
_safe_export("dmpc_fm_cbf.sampler_5h", ["piecewise_sample_fm_5ch_with_cbf", "split_and_denormalize_5ch"])

