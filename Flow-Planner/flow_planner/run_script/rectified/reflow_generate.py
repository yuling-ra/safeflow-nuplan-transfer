#!/usr/bin/env python3
"""Generate deterministic ReFlow pairs with the existing SafeFlow sampler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch


def _project_root() -> Path:
    # In the installed layout this file is dmpc_fm_cbf/rectified/reflow_generate.py.
    return Path(__file__).resolve().parents[1]


def _imports():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rectified.fmtorch.models.models import SimpleUNet1D
    from rectified.dmpc_fm_cbf.sampler_5h import piecewise_sample_fm_5ch_with_cbf

    return SimpleUNet1D, piecewise_sample_fm_5ch_with_cbf


def _load_data(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if raw.shape == () and raw.dtype == object else raw
    if not isinstance(data, dict):
        raise ValueError("The dataset must be a .npy dictionary.")
    if "states" not in data or "controls" not in data or "goals" not in data:
        raise KeyError("Dataset requires 'states', 'controls', and 'goals'.")
    states = np.asarray(data["states"], dtype=np.float32)
    controls = np.asarray(data["controls"], dtype=np.float32)
    if states.ndim != 3 or states.shape[-1] < 5:
        raise ValueError(f"states must be (N,T,5+), got {states.shape}")
    if controls.shape[:2] != states.shape[:2] or controls.shape[-1] != 2:
        raise ValueError(f"controls must be (N,T,2), got {controls.shape}")
    # The stored ring data is in world coordinates. The checkpoint was trained
    # on the project's canonical goal-aligned frame, so reuse the exact same
    # conversion as scripts/train_5ch_checkpoint.py.
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rectified.dmpc_fm_cbf.canonical import canonicalize_episode_5d_noobs

    goals = np.asarray(data["goals"], dtype=np.float32)
    x1_list, cond_list = [], []
    for sample_states, sample_controls, goal in zip(states, controls, goals):
        x1_sample, cond_sample, _ = canonicalize_episode_5d_noobs(
            sample_states[:, :5], sample_controls, goal
        )
        x1_list.append(x1_sample)
        cond_list.append(cond_sample)
    x1 = np.stack(x1_list, axis=0).astype(np.float32)
    cond = np.stack(cond_list, axis=0).astype(np.float32)
    return {"x1": x1, "cond": cond}


def _model(checkpoint: Path, device: torch.device):
    SimpleUNet1D, _ = _imports()
    ckpt = torch.load(checkpoint, map_location=device)
    config = dict(ckpt.get("model_config", {}))
    config.setdefault("time_emb_dim", 128)
    config.setdefault("hidden_dim", 128)
    config.setdefault("cond_dim", 5)
    config.setdefault("in_channels", 5)
    config.setdefault("out_channels", 5)
    model = SimpleUNet1D(**config).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _obstacles(path: Path | None, count: int) -> List[list[dict[str, Any]]]:
    if path is None:
        return [[] for _ in range(count)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != count:
        raise ValueError(f"Obstacle JSON must be a list with exactly {count} samples.")
    return payload


def generate(args: argparse.Namespace) -> Path:
    _, sampler = _imports()
    device = torch.device(args.device)
    data = _load_data(Path(args.data))
    if args.limit > 0:
        data = {key: value[: args.limit] for key, value in data.items()}
    model = _model(Path(args.checkpoint), device)
    obstacles = _obstacles(Path(args.obstacles) if args.obstacles else None, len(data["x1"]))
    x0_all, x1_all = [], []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for index, (target, cond, ellipses) in enumerate(zip(data["x1"], data["cond"], obstacles)):
        # A fixed seed gives a reproducible deterministic coupling per sample.
        x0 = torch.randn((1, 5, target.shape[-1]), generator=generator, device=device)
        x1, _, _ = sampler(
            model_5ch=model,
            T_steps=target.shape[-1],
            device=str(device),
            num_segments=args.num_segments,
            ode_method=args.ode_method,
            noise_scale=args.noise_scale,
            x0_noise=x0,
            cond=cond,
            start_xy_norm=np.zeros(2, dtype=np.float32),
            goal_xy_norm=np.array([float(cond[0]), 0.0], dtype=np.float32),
            start_guidance_weight=args.start_guidance,
            goal_guidance_weight=args.goal_guidance,
            lock_start=True,
            use_cbf=args.cbf,
            ellipses=ellipses,
            tau0=args.cbf_tau0,
            tau1=args.cbf_tau1,
            cbf_kwargs={
                "passes": args.cbf_passes,
                "extra_margin": args.cbf_extra_margin,
                "max_corr_norm": args.cbf_max_corr_norm,
                "alpha": args.cbf_alpha,
            },
        )
        x0_all.append(x0.detach().cpu().numpy()[0].astype(np.float32))
        x1_all.append(x1.astype(np.float32))
        if (index + 1) % max(1, args.log_every) == 0:
            print(f"generated {index + 1}/{len(data['x1'])}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x0=np.stack(x0_all),
        x1=np.stack(x1_all),
        cond=data["cond"],
        source_checkpoint=str(args.checkpoint),
        cbf_enabled=np.array(bool(args.cbf)),
    )
    print(f"saved {output} with x0={np.stack(x0_all).shape}, x1={np.stack(x1_all).shape}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--obstacles", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--start-guidance", type=float, default=0.4)
    parser.add_argument("--goal-guidance", type=float, default=1.2)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--cbf-tau0", type=float, default=0.5)
    parser.add_argument("--cbf-tau1", type=float, default=0.9)
    parser.add_argument("--cbf-passes", type=int, default=8)
    parser.add_argument("--cbf-extra-margin", type=float, default=0.25)
    parser.add_argument("--cbf-max-corr-norm", type=float, default=2.0)
    parser.add_argument("--cbf-alpha", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Only generate this many samples; 0 means all.")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
