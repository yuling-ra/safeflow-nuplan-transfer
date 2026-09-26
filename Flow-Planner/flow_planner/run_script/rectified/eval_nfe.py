#!/usr/bin/env python3
"""Compare deterministic ODE function evaluations and endpoint quality."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rectified.fmtorch.models.models import SimpleUNet1D
    from rectified.dmpc_fm_cbf.sampler_5h import piecewise_sample_fm_5ch_with_cbf

    return SimpleUNet1D, piecewise_sample_fm_5ch_with_cbf


def _model(checkpoint: Path, device: torch.device):
    SimpleUNet1D, _ = _imports()
    ckpt = torch.load(checkpoint, map_location=device)
    config = dict(ckpt.get("model_config", {}))
    config.setdefault("time_emb_dim", 128); config.setdefault("hidden_dim", 128)
    config.setdefault("cond_dim", 5); config.setdefault("in_channels", 5); config.setdefault("out_channels", 5)
    model = SimpleUNet1D(**config).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt)); model.eval()
    return model


def _data(path: Path):
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if raw.shape == () and raw.dtype == object else raw
    states = np.asarray(data["states"], dtype=np.float32)
    controls = np.asarray(data["controls"], dtype=np.float32)
    from rectified.dmpc_fm_cbf.canonical import canonicalize_episode_5d_noobs
    goals = np.asarray(data["goals"], dtype=np.float32)
    targets, conditions = [], []
    for sample_states, sample_controls, goal in zip(states, controls, goals):
        target, condition, _ = canonicalize_episode_5d_noobs(
            sample_states[:, :5], sample_controls, goal
        )
        targets.append(target)
        conditions.append(condition)
    return np.stack(targets, axis=0).astype(np.float32), np.stack(conditions, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    _, sampler = _imports(); device = torch.device(args.device)
    model = _model(Path(args.checkpoint), device)
    target, cond = _data(Path(args.data))
    count = min(args.limit, len(target)); rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(target), count, replace=False)
    for segments in args.steps:
        errors, elapsed = [], []
        for index in indices:
            started = time.perf_counter()
            generated, _, _ = sampler(
                model_5ch=model, T_steps=target.shape[-1], device=str(device), num_segments=segments,
                ode_method=args.ode_method, noise_scale=0.0,
                x0_noise=torch.zeros((1, 5, target.shape[-1]), device=device), cond=cond[index],
                start_xy_norm=np.zeros(2, dtype=np.float32),
                goal_xy_norm=np.array([float(cond[index, 0]), 0.0], dtype=np.float32),
                start_guidance_weight=args.start_guidance, goal_guidance_weight=args.goal_guidance,
                lock_start=True, use_cbf=False, ellipses=[],
            )
            elapsed.append(time.perf_counter() - started)
            errors.append(float(np.linalg.norm(generated[:, -1] - target[index, :, -1])))
        print(f"segments={segments:>3} mean_endpoint_l2={np.mean(errors):.6f} mean_ms={1000*np.mean(elapsed):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", default="cpu"); parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7); parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--start-guidance", type=float, default=0.4); parser.add_argument("--goal-guidance", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
