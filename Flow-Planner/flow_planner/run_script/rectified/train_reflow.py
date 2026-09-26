#!/usr/bin/env python3
"""Warm-start training on deterministic Rectified Flow couplings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rectified.fmtorch.models.models import SimpleUNet1D

    return SimpleUNet1D


def _load_model(checkpoint: Path, device: torch.device):
    SimpleUNet1D = _imports()
    ckpt = torch.load(checkpoint, map_location=device)
    config = dict(ckpt.get("model_config", {}))
    config.setdefault("time_emb_dim", 128)
    config.setdefault("hidden_dim", 128)
    config.setdefault("cond_dim", 5)
    config.setdefault("in_channels", 5)
    config.setdefault("out_channels", 5)
    model = SimpleUNet1D(**config).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model, config


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    pairs = np.load(args.pairs, allow_pickle=True)
    x0 = torch.from_numpy(np.asarray(pairs["x0"], dtype=np.float32))
    x1 = torch.from_numpy(np.asarray(pairs["x1"], dtype=np.float32))
    cond = torch.from_numpy(np.asarray(pairs["cond"], dtype=np.float32))
    if x0.shape != x1.shape or x0.ndim != 3 or x0.shape[1] != 5:
        raise ValueError(f"Expected x0/x1 shape (N,5,T), got {x0.shape} and {x1.shape}")
    if cond.shape != (x0.shape[0], 5):
        raise ValueError(f"Expected cond shape (N,5), got {cond.shape}")
    loader = DataLoader(TensorDataset(x0, x1, cond), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    model, config = _load_model(Path(args.init_checkpoint), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for source, target, condition in loader:
            source, target, condition = source.to(device), target.to(device), condition.to(device)
            t = torch.rand(source.shape[0], device=device, generator=generator)
            t_view = t.view(-1, 1, 1)
            xt = (1.0 - t_view) * source + t_view * target
            velocity = target - source
            prediction = model(xt, t, cond=condition)
            loss = F.mse_loss(prediction, velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses))
        history.append(value)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={value:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "reflow_config": vars(args),
        "history": {"train_loss": history},
        "pair_file": str(args.pairs),
    }, output)
    print(f"saved {output}; best_loss={min(history):.6f}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
