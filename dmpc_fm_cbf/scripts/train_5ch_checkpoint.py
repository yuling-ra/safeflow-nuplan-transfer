#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the canonical 5-channel FM checkpoint used by the nuPlan transfer benchmark."
    )
    parser.add_argument("--n", type=int, default=100, help="Number of generated training trajectories.")
    parser.add_argument("--t", type=int, default=60, help="Trajectory horizon.")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate the ring dataset even if notebooks/cache already has it.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to notebooks/cache/model_vel_5ch_canonical_r.pt.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Dataset cache directory. Defaults to notebooks/cache.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' uses CUDA only when this PyTorch build supports the GPU architecture.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prepare_imports(root: Path) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_or_generate_dataset(
    *,
    cache_dir: Path,
    n: int,
    t: int,
    seed: int,
    force_regenerate: bool,
) -> tuple[dict, Path, Path]:
    from dmpc_fm_cbf.dataset_ import make_center_to_ring_dataset_tracking_oc

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "car_ring_track_oc_dataset_v1_v0rand.npy"
    data_path = cache_dir / "car_ring_track_oc_dataset_v1_v0rand_with_vel.npy"

    if data_path.exists() and not force_regenerate:
        print(f"[Data] loading {data_path}")
        return np.load(data_path, allow_pickle=True).item(), raw_path, data_path

    print(f"[Data] generating N={n}, T={t} into {raw_path}")
    data = make_center_to_ring_dataset_tracking_oc(
        N=n,
        T=t,
        radius=3.0,
        seed=seed,
        save_filename=str(raw_path),
        LOAD_SAVED_DATA=False,
        v0_min=0.15,
        v0_max=0.6,
    )

    dt = float(data["params"].get("dt", 0.1))
    states = np.asarray(data["states"], dtype=np.float32)
    data["vel_xy_Tm1"] = np.diff(states[:, :, 0:2], axis=1) / dt
    np.save(data_path, data, allow_pickle=True)
    print(f"[Data] saved {data_path}")
    return data, raw_path, data_path


def build_canonical_dataset(data: dict) -> tuple[np.ndarray, np.ndarray]:
    from dmpc_fm_cbf.canonical import canonicalize_episode_5d_noobs

    states = np.asarray(data["states"], dtype=np.float32)
    controls = np.asarray(data["controls"], dtype=np.float32)
    goals = np.asarray(data["goals"], dtype=np.float32)

    n, t, d = states.shape
    if d < 5:
        raise ValueError(f"states must be (N,T,5+). Got {states.shape}")
    if controls.shape != (n, t, 2):
        raise ValueError(f"controls must be {(n, t, 2)}. Got {controls.shape}")
    if goals.shape != (n, 2):
        raise ValueError(f"goals must be {(n, 2)}. Got {goals.shape}")

    x1_list = []
    cond_list = []
    for i in range(n):
        x1, cond, _ = canonicalize_episode_5d_noobs(states[i, :, :5], controls[i], goals[i])
        x1_list.append(x1)
        cond_list.append(cond)

    x1 = np.stack(x1_list, axis=0).astype(np.float32)
    cond = np.stack(cond_list, axis=0).astype(np.float32)
    print(f"[Canonical] X1={x1.shape}, cond={cond.shape}")
    return x1, cond


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("DEVICE=cuda was requested, but torch.cuda.is_available() is false.")
        return "cpu"

    if requested == "cuda":
        return "cuda"

    capability = torch.cuda.get_device_capability(0)
    current_arch = capability[0] * 10 + capability[1]
    supported_arches = []
    for arch in torch.cuda.get_arch_list():
        if arch.startswith("sm_"):
            try:
                supported_arches.append(int(arch.removeprefix("sm_")))
            except ValueError:
                pass

    if supported_arches and current_arch > max(supported_arches):
        name = torch.cuda.get_device_name(0)
        print(
            f"[Device] CUDA device {name} has sm_{current_arch}, "
            f"but this PyTorch build supports up to sm_{max(supported_arches)}. Falling back to CPU."
        )
        return "cpu"

    return "cuda"


def train(args: argparse.Namespace) -> Path:
    root = project_root()
    prepare_imports(root)

    from dmpc_fm_cbf.training import FlowMatchingTrainer, TrainConfig
    from fmtorch.models.simple1d_unet import SimpleUNet1D

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = args.cache_dir or root / "notebooks" / "cache"
    ckpt_path = args.checkpoint or cache_dir / "model_vel_5ch_canonical_r.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[Paths] project={root}")
    print(f"[Paths] cache={cache_dir}")
    print(f"[Paths] checkpoint={ckpt_path}")
    print(f"[Device] {device}")

    data, raw_path, data_path = load_or_generate_dataset(
        cache_dir=cache_dir,
        n=args.n,
        t=args.t,
        seed=args.seed,
        force_regenerate=args.force_regenerate,
    )
    x1, cond = build_canonical_dataset(data)

    dataset = TensorDataset(torch.from_numpy(x1), torch.from_numpy(cond))
    model = SimpleUNet1D(
        time_emb_dim=128,
        hidden_dim=128,
        cond_dim=5,
        in_channels=5,
        out_channels=5,
    )
    print(f"[Model] parameters={sum(p.numel() for p in model.parameters()):,}")

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        log_every=max(1, min(50, args.epochs)),
        grad_clip=1.0,
        scheduler_type="none",
    )
    trainer = FlowMatchingTrainer(model=model, config=cfg, device=device)
    history = trainer.train(dataset)

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "global_step": trainer.global_step,
            "best_loss": trainer.best_loss,
            "history": history,
            "model_config": {
                "time_emb_dim": 128,
                "hidden_dim": 128,
                "cond_dim": 5,
                "in_channels": 5,
                "out_channels": 5,
            },
            "data_config": {
                "raw_path": str(raw_path),
                "data_path": str(data_path),
                "channel_layout": ["x_can", "y_can", "theta_can", "a", "r_dot"],
                "cond_layout": ["goal_dist", "v0", "cos_theta0_can", "sin_theta0_can", "r0"],
            },
        },
        ckpt_path,
    )
    print(f"[Saved] {ckpt_path}")

    reloaded = torch.load(ckpt_path, map_location="cpu")
    state = reloaded["model_state_dict"]
    print(f"[Check] conv_in.weight={tuple(state['conv_in.weight'].shape)}")
    print(f"[Check] conv_out.2.weight={tuple(state['conv_out.2.weight'].shape)}")
    return ckpt_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
