import numpy as np, torch, os, glob
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from typing import Union, List

@dataclass
class MinMaxScaler:
    min_: np.ndarray
    max_: np.ndarray
    def normalize(self, x):   return (2.0*(x - self.min_) / (self.max_-self.min_ + 1e-12) - 1.0)
    def denorm(self, x_norm): return ( (x_norm + 1.0)/2.0 * (self.max_-self.min_) + self.min_ )

class FullTrajDataset(Dataset):
    """
    输出:
      condition: x0s (B, cond_dim)
      target:    y = [actions (T×2), states (T+1×4)] flatten → (B, y_dim)
      都是 float32，且做了归一化（独立 scaler）
    """
    def __init__(self, path_or_paths: Union[str, List[str]], theta_sincos=False, cache_scaler=True):
        super().__init__()

        if isinstance(path_or_paths, str):
            if os.path.isdir(path_or_paths):
                paths = sorted(glob.glob(os.path.join(path_or_paths, "*.npz")))
                if not paths: raise ValueError(f"No .npz files found in directory: {path_or_paths}")
            else:
                paths = [path_or_paths]
        elif isinstance(path_or_paths, list):
            paths = path_or_paths
        else:
            raise TypeError(f"Expected path_or_paths to be str or list, but got {type(path_or_paths)}")

        self.sources = [os.path.basename(p) for p in paths]
        print(f"Loading data from {len(self.sources)} files: {self.sources}")

        all_states, all_actions, all_x0s = [], [], []
        for p in paths:
            try:
                with np.load(p) as data:
                    all_states.append(data["states"])
                    all_actions.append(data["actions"])
                    all_x0s.append(data["x0s"])
            except Exception as e:
                print(f"[Warning] Could not load data from {p}. Skipping. Error: {e}")
                continue
        
        if not all_states:
            raise ValueError("No valid data could be loaded from the provided paths.")

        self.states  = np.concatenate(all_states, axis=0)
        self.actions = np.concatenate(all_actions, axis=0)
        self.x0s     = np.concatenate(all_x0s, axis=0)
        
        print(f"Successfully loaded and combined data. Total trajectories: {self.states.shape[0]}")

        # 条件维度处理（theta 可选 sincos）
        if theta_sincos:
            self.cond = self._theta_to_sincos(self.x0s)  # (N,5)
        else:
            self.cond = self.x0s                          # (N,4)

        N = self.states.shape[0]
        # 目标向量：flatten(actions, states)
        self.y_raw = np.concatenate([
            self.actions.reshape(N, -1),     # (N, 100*2)
            self.states.reshape(N,  -1)      # (N, 101*4)
        ], axis=1).astype(np.float64)

        # scaler
        if cache_scaler:
            y_min, y_max = self.y_raw.min(axis=0), self.y_raw.max(axis=0)
            c_min, c_max = self.cond.min(axis=0),  self.cond.max(axis=0)
        else:
            # 也可以外部提供
            y_min, y_max = self.y_raw.min(0), self.y_raw.max(0)
            c_min, c_max = self.cond.min(0),  self.cond.max(0)
        self.y_scaler   = MinMaxScaler(y_min, y_max)
        self.c_scaler   = MinMaxScaler(c_min, c_max)

        self.y_dim   = self.y_raw.shape[1]
        self.cond_dim= self.cond.shape[1]

        self.T_actions, self.T_states = self.actions.shape[1], self.states.shape[1]
        self.action_dim, self.state_dim = self.actions.shape[2], self.states.shape[2]
        A = self.T_actions * self.action_dim
        S = self.T_states  * self.state_dim
        self.slices = {
            "actions": (0, A),
            "states":  (A, A+S)
        }
        self.cond_schema = (["x","y","theta","v"] if not theta_sincos
                            else ["x","y","sin_theta","cos_theta","v"])
        self.spec = {
            "T_actions": self.T_actions,
            "T_states":  self.T_states,
            "action_dim": self.action_dim,
            "state_dim":  self.state_dim,
            "slices":     self.slices,
            "cond_schema": self.cond_schema,
            "y_layout":   ["actions", "states"],
            "theta_sincos": theta_sincos,
        }

    def _theta_to_sincos(self, X):  # X (N,4) with theta at index=2
        Y = X.copy()
        theta = Y[:,2]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        Y = np.concatenate([Y[:, :2], sin_t[:,None], cos_t[:,None], Y[:, 3:4]], axis=1)  # (N,5)
        return Y

    def __len__(self): return self.y_raw.shape[0]

    def __getitem__(self, idx):
        cond = self.c_scaler.normalize(self.cond[idx]).astype(np.float32)
        y    = self.y_scaler.normalize(self.y_raw[idx]).astype(np.float32)
        return torch.from_numpy(cond), torch.from_numpy(y)

def build_loaders(cfg):
    data_path = cfg.data.get("path", None)

    # Construct the default path relative to this file's location to make it robust.
    # __file__ is in .../src/, so we go up one level and then into dataset/.
    if not data_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "dataset")
        print(f"cfg.data.path not specified, defaulting to '{data_path}'")
    
    # If the path (default or specified) is a directory, check if it contains any .npz files
    if isinstance(data_path, str) and os.path.isdir(data_path) and not glob.glob(os.path.join(data_path, "*.npz")):
         raise FileNotFoundError(f"Dataset directory '{data_path}' is empty or contains no .npz files.")

    ds = FullTrajDataset(data_path, theta_sincos=cfg.data.theta_sincos, cache_scaler=cfg.data.cache_scaler)
    N = len(ds); n_val = int(N * cfg.data.val_split)
    n_train = N - n_val
    g = torch.Generator().manual_seed(123)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.workers,
                              pin_memory=True, drop_last=True, persistent_workers=cfg.data.workers>0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.workers,
                              pin_memory=True, drop_last=False, persistent_workers=cfg.data.workers>0)
    return train_loader, val_loader, ds
