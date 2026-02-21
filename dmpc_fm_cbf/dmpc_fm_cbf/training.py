"""
Training utilities for Flow Matching models.

Provides:
- FlowMatchingTrainer: Training loop for FM models
- TrajectoryDataset: PyTorch dataset for trajectory data
- Training utilities and loss functions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass
import time


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    log_every: int = 10
    save_every: int = 20
    grad_clip: float = 1.0
    warmup_steps: int = 100
    scheduler_type: str = "cosine"  # "cosine", "step", "none"
    

class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory training.
    
    Supports both position trajectories and velocity sequences.
    """
    
    def __init__(
        self,
        data: Dict,
        mode: str = "velocity",
        cond_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            data: Dataset dict with keys like 'xy', 'velocity', 'cond', 'goals'
            mode: 'velocity' or 'position'
            cond_keys: Keys to use for conditioning
        """
        self.mode = mode
        
        # Load trajectories
        if mode == "velocity" and "velocity" in data:
            self.trajs = data["velocity"]
        elif "xy" in data:
            self.trajs = data["xy"]
        else:
            raise KeyError("Dataset must have 'xy' or 'velocity' key")
        
        # Convert to tensor
        self.trajs = torch.tensor(self.trajs, dtype=torch.float32)
        
        # Load conditioning
        if "cond" in data:
            self.cond = torch.tensor(data["cond"], dtype=torch.float32)
        elif cond_keys:
            cond_list = []
            for key in cond_keys:
                if key in data:
                    c = data[key]
                    if c.ndim == 1:
                        c = c[:, None]
                    cond_list.append(c)
            if cond_list:
                self.cond = torch.tensor(np.concatenate(cond_list, axis=1), dtype=torch.float32)
            else:
                self.cond = None
        else:
            self.cond = None
        
        self.N = len(self.trajs)
    
    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        traj = self.trajs[idx]  # [T, 2]
        
        # Transpose to [2, T] for 1D conv
        traj = traj.transpose(0, 1)
        
        if self.cond is not None:
            return traj, self.cond[idx]
        return traj, None


class CondOTScheduler:
    """
    Conditional Optimal Transport scheduler for Flow Matching.
    
    Implements linear interpolation path: x_t = (1-t)*x_0 + t*x_1
    """
    
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Return noise level at time t (constant small value for OT)."""
        return 1e-5 * torch.ones_like(t)


class AffinePath:
    """
    Affine probability path for Flow Matching.
    
    x_t = (1-t)*x_0 + t*x_1
    dx_t = x_1 - x_0
    """
    
    def __init__(self, scheduler: Optional[CondOTScheduler] = None):
        self.scheduler = scheduler or CondOTScheduler()
    
    def sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> "PathSample":
        """
        Sample from the conditional path.
        
        Args:
            x_0: Source (noise) [B, ...]
            x_1: Target (data) [B, ...]
            t: Time [B]
        
        Returns:
            PathSample with x_t, dx_t, t
        """
        # Expand t for broadcasting
        if t.dim() == 1:
            t = t.view(-1, *([1] * (x_0.dim() - 1)))
        
        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0  # Velocity is constant along path
        
        return PathSample(x_t=x_t, dx_t=dx_t, t=t, x_0=x_0, x_1=x_1)


@dataclass
class PathSample:
    """Sample from a probability path."""
    x_t: torch.Tensor    # Interpolated point
    dx_t: torch.Tensor   # Target velocity
    t: torch.Tensor      # Time
    x_0: torch.Tensor    # Source (noise)
    x_1: torch.Tensor    # Target (data)


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching velocity models.
    
    Implements the conditional flow matching objective:
        L = E[||v_theta(x_t, t) - (x_1 - x_0)||^2]
    
    where x_t = (1-t)*x_0 + t*x_1 is the interpolated point.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainConfig] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model: Velocity model (e.g., SimpleUNet1D)
            config: Training configuration
            device: Compute device
        """
        self.model = model
        self.config = config or TrainConfig()
        self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = None  # Set in train()
        
        # Path
        self.path = AffinePath()
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}
    
    def _setup_scheduler(self, num_steps: int):
        """Setup learning rate scheduler."""
        cfg = self.config
        
        if cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_steps, eta_min=cfg.lr * 0.01
            )
        elif cfg.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_steps // 3, gamma=0.5
            )
        else:
            self.scheduler = None
    
    def _compute_loss(
        self,
        batch: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            batch: (x_1, cond) where x_1 is [B, 2, T]
        
        Returns:
            Scalar loss
        """
        x_1, cond = batch
        x_1 = x_1.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        
        B = x_1.shape[0]
        
        # Sample noise (source)
        x_0 = torch.randn_like(x_1)
        
        # Sample time uniformly
        t = torch.rand(B, device=self.device)
        
        # Get path sample
        path_sample = self.path.sample(x_0, x_1, t)
        
        # Predict velocity
        v_pred = self.model(path_sample.x_t, t.view(-1), cond=cond)
        
        # MSE loss against target velocity
        loss = F.mse_loss(v_pred, path_sample.dx_t)
        
        return loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with 'train_loss' and optionally 'val_loss'
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            loss = self._compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        
        metrics = {"train_loss": total_loss / num_batches}
        
        # Validation
        if val_loader is not None:
            val_loss = self.evaluate(val_loader)
            metrics["val_loss"] = val_loss
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            callback: Called after each epoch with (epoch, metrics)
        
        Returns:
            Training history
        """
        cfg = self.config
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.batch_size, shuffle=False
            )
        
        num_steps = cfg.num_epochs * len(train_loader)
        self._setup_scheduler(num_steps)
        
        print(f"Training for {cfg.num_epochs} epochs ({num_steps} steps)")
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Val samples: {len(val_dataset)}")
        
        start_time = time.time()
        
        for epoch in range(cfg.num_epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            self.history["train_loss"].append(metrics["train_loss"])
            if "val_loss" in metrics:
                self.history["val_loss"].append(metrics["val_loss"])
            
            # Update best
            if metrics["train_loss"] < self.best_loss:
                self.best_loss = metrics["train_loss"]
            
            # Logging
            if (epoch + 1) % cfg.log_every == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch+1}/{cfg.num_epochs} | loss={metrics['train_loss']:.4f}"
                if "val_loss" in metrics:
                    msg += f" | val={metrics['val_loss']:.4f}"
                msg += f" | lr={lr:.2e} | time={elapsed:.1f}s"
                print(msg)
            
            # Callback
            if callback is not None:
                callback(epoch, metrics)
        
        print(f"Training complete. Best loss: {self.best_loss:.4f}")
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.history,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float('inf'))
        self.history = ckpt.get("history", {"train_loss": [], "val_loss": []})
        print(f"Loaded checkpoint: {path}")


def train_fm_model(
    model: nn.Module,
    train_data: Dict,
    val_data: Optional[Dict] = None,
    config: Optional[TrainConfig] = None,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a Flow Matching model.
    
    Args:
        model: Velocity model
        train_data: Training data dict
        val_data: Validation data dict (optional)
        config: Training config
        device: Compute device
        save_path: Path to save final model
    
    Returns:
        Trained model and history
    """
    train_dataset = TrajectoryDataset(train_data, mode="velocity")
    val_dataset = TrajectoryDataset(val_data) if val_data else None
    
    trainer = FlowMatchingTrainer(model, config, device)
    history = trainer.train(train_dataset, val_dataset)
    
    if save_path:
        trainer.save_checkpoint(save_path)
    
    return trainer.model, history
