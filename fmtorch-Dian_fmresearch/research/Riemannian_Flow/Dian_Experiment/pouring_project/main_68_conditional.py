import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import json
from pathlib import Path
import bitsandbytes as bnb
from datetime import datetime
import zipfile
from scipy.spatial.transform import Rotation as R
import torch.optim as optim

# fmtorch components
from fmtorch.scheduler import CondOTScheduler
from fmtorch.path import AffinePath

# Local project components
from preprocess import (align_to_first_frame, 
                         compute_delta_xi, reconstruct_incremental_pose,
                         se3_log, se3_exp)
from model import EnhancedVectorFieldNet

# ==============================================================================
# Custom Data Loading for JSON files in ZIP
# ==============================================================================

def quaternion_to_matrix(quat):
    """Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    return R.from_quat(quat).as_matrix()

def load_trajectory_from_json_in_zip(zip_path, json_file):
    """Loads a single trajectory from a JSON file inside a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file) as f:
            data = json.load(f)
    trajectory_raw = data['trajectory']
    T_seq = np.zeros((len(trajectory_raw), 4, 4))
    for i, frame in enumerate(trajectory_raw):
        T_seq[i, :3, 3] = frame[:3]
        # Dataset provides quaternion in [w, x, y, z]; convert to [x, y, z, w]
        quat_wxyz = frame[3:]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        T_seq[i, :3, :3] = quaternion_to_matrix(quat_xyzw)
        T_seq[i, 3, 3] = 1.0
    return T_seq, {'traj': T_seq}

def get_json_files_from_zip(zip_path):
    """Lists all .json files in a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        return [name for name in z.namelist() if name.endswith('.json')]

# ==============================================================================
# Training Function with Custom Logic
# ==============================================================================

def train_model(model, loader, config, device, artifacts_dir):
    """Trains the model with early stopping and conditional checkpointing."""
    model = model.to(device)
    path = AffinePath(scheduler=CondOTScheduler())
    
    optimizer = optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-5)
    )
    
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=config.get('lr', 1e-3) * 0.01)
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=config['scheduler_factor'], 
            patience=config['scheduler_patience']
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model.train()
    
    best_loss = float('inf')
    epochs_no_improve = 0
    checkpointing_active = False
    
    print("\n--- Starting Training ---")
    for epoch in range(config['n_epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            x1, cond, lengths = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            x0, t = torch.randn_like(x1), torch.rand(x1.shape[0], device=device)
            s = path.sample(t=t, x_0=x0, x_1=x1)
            
            # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
            pred_v = model(s.x_t, s.t, condition=cond)
            # mask padded frames
            B, L, C = pred_v.shape
            frame_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            mask = (frame_idx < lengths.unsqueeze(1)).float().unsqueeze(-1)
            mse = (pred_v - s.dx_t) ** 2
            loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 0.5))
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step()
        
        # --- Check for improvement ---
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0
            print(f"Epoch {epoch+1:4d} | New Best Loss: {best_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {time.time() - epoch_start_time:.2f}s")

            # Conditional checkpointing logic
            if not checkpointing_active and best_loss < 0.08:
                print("  -> Loss threshold reached. Checkpointing is now active.")
                checkpointing_active = True

            if checkpointing_active:
                best_model_path = artifacts_dir / 'best_model.pt'
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Model checkpoint saved to {best_model_path}")

        else:
            epochs_no_improve += 1

        # Periodic status update
        if (epoch + 1) % 50 == 0:
             current_lr = optimizer.param_groups[0]['lr']
             print(f"Epoch {epoch+1:4d} | Avg Loss: {avg_epoch_loss:.6f} | Best: {best_loss:.6f} | LR: {current_lr:.6f} | No improve: {epochs_no_improve} epochs")

        # --- Early stopping logic ---
        if epochs_no_improve >= config['early_stop_patience']:
            print(f"\nEarly stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
            
    print("--- Training Finished ---")
    return best_loss

# ==============================================================================
# Main Orchestration
# ==============================================================================

def main():
    # --- 1. Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    script_dir = Path(__file__).parent
    run_name = f"main_68_conditional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir = script_dir / 'training_runs' / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved to: {artifacts_dir}")

    # --- 2. Data Loading ---
    all_trajectories = []  # whitened Z_delta tensors (var-len)
    all_conditions = []
    all_lengths = []
    # for global whitening accumulation
    delta_list = []
    raw_T_list = []
    T0_list = []
    dataset_zips = ['dataset/gestor_pouring_dataset_6.zip', 'dataset/gestor_pouring_dataset_7.zip']
    
    for zip_file in dataset_zips:
        zip_path = script_dir / zip_file
        json_files = get_json_files_from_zip(zip_path)
        for json_f in json_files:
            T_seq_raw, _ = load_trajectory_from_json_in_zip(zip_path, json_f)
            T_seq_aligned, T_0 = align_to_first_frame(T_seq_raw)
            delta_xi = compute_delta_xi(T_seq_aligned)  # (L_i, 6)
            delta_list.append(delta_xi)
            raw_T_list.append(T_seq_raw)
            T0_list.append(T_0)
            
            # Store condition (log of the first frame's pose)
            cond_6d = se3_log(T_0)
            all_conditions.append(torch.from_numpy(cond_6d).float())

    # --- Global whitening over all trajectories ---
    if len(delta_list) == 0:
        raise RuntimeError("No trajectories found for training.")
    delta_concat = np.concatenate(delta_list, axis=0)
    omega_all = delta_concat[:, :3]
    v_all = delta_concat[:, 3:]
    mu_omega = omega_all.mean(axis=0)
    sigma_omega = omega_all.std(axis=0) + 1e-8
    mu_v = v_all.mean(axis=0)
    sigma_v = v_all.std(axis=0) + 1e-8

    whitening_stats_delta = {
        'mu_omega': mu_omega.tolist(), 'sigma_omega': sigma_omega.tolist(),
        'mu_v': mu_v.tolist(), 'sigma_v': sigma_v.tolist()
    }
    with open(artifacts_dir / '05_whiten_delta.json', 'w') as f:
        json.dump(whitening_stats_delta, f, indent=2)
    print("Global whitening stats saved.")

    # --- Self-check: reconstruct from delta_xi and compare to raw ---
    def rotation_geodesic_deg(R1, R2):
        Rdiff = R1.T @ R2
        trace = np.clip((np.trace(Rdiff) - 1) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(trace))

    rot_err_means, trans_err_means = [], []
    for T_raw, T0, delta_xi in zip(raw_T_list, T0_list, delta_list):
        N = T_raw.shape[0]
        T_recon_aligned = np.zeros_like(T_raw)
        T_recon_aligned[0] = np.eye(4)
        for k in range(1, N):
            dT = se3_exp(delta_xi[k-1])
            T_recon_aligned[k] = T_recon_aligned[k-1] @ dT
        T_recon_world = np.array([T0 @ T for T in T_recon_aligned])

        rot_err = []
        trans_err = []
        for k in range(N):
            rot_err.append(rotation_geodesic_deg(T_raw[k, :3, :3], T_recon_world[k, :3, :3]))
            trans_err.append(np.linalg.norm(T_raw[k, :3, 3] - T_recon_world[k, :3, 3]))
        rot_err_means.append(np.mean(rot_err))
        trans_err_means.append(np.mean(trans_err))

    rot_mean = float(np.mean(rot_err_means))
    trans_mean = float(np.mean(trans_err_means))
    print(f"Self-check (delta -> recon) | mean rot err: {rot_mean:.3f} deg, mean trans err: {trans_mean*100:.2f} cm")

    # Reasonable thresholds (adjust if needed)
    rot_mean_thresh_deg = 1.0
    trans_mean_thresh_m = 0.01
    if rot_mean > rot_mean_thresh_deg or trans_mean > trans_mean_thresh_m:
        raise RuntimeError(
            f"Preprocess self-check failed: mean rot {rot_mean:.3f} deg > {rot_mean_thresh_deg} or "
            f"mean trans {trans_mean:.4f} m > {trans_mean_thresh_m}. Please inspect preprocessing.")

    # Apply whitening and record lengths
    for delta_xi in delta_list:
        Z_delta = np.zeros_like(delta_xi)
        Z_delta[:, :3] = (delta_xi[:, :3] - mu_omega) / sigma_omega
        Z_delta[:, 3:] = (delta_xi[:, 3:] - mu_v) / sigma_v
        all_lengths.append(Z_delta.shape[0])
        all_trajectories.append(torch.from_numpy(Z_delta).float())

    max_len = max(traj.shape[0] for traj in all_trajectories)
    padded_trajectories = [F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in all_trajectories]
    data_tensor = torch.stack(padded_trajectories)
    condition_tensor = torch.stack(all_conditions)
    lengths_tensor = torch.tensor(all_lengths, dtype=torch.long)

    print(f"Loaded {data_tensor.shape[0]} trajectories, padded to length {max_len}.")
    print(f"Condition tensor shape: {condition_tensor.shape}")

    # --- 3. Setup Model and Training Config ---
    config = {
        'lr': 0.0001,
        'scheduler': 'cosine',
        'batch_size': 32,
        'hidden_dim': 512,
        'num_layers': 4,
        'n_epochs': 3500,
        'early_stop_patience': 500,
        'bridge_k': 8,  # From previous window for K/V
        'bridge_q': 8,  # From current window for Q
        'data_dim': data_tensor.shape[2],
        'cond_dim': condition_tensor.shape[1],
        'seq_len': max_len,
        'time_embed_dim': 256,
        'dropout': 0.1,
        'weight_decay': 1e-5,
        'grad_clip': 0.5,
        'num_heads': 4,
        'reduction': 16,
        'window': 96,
        'fourier_k': 32
    }

    with open(artifacts_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Training configuration saved.")

    dataset = torch.utils.data.TensorDataset(data_tensor, condition_tensor, lengths_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = EnhancedVectorFieldNet(**{k: v for k, v in config.items() if k in [
        'seq_len', 'data_dim', 'hidden_dim', 'num_layers', 'time_embed_dim', 
        'dropout', 'num_heads', 'reduction', 'window', 'fourier_k', 
        'bridge_k', 'bridge_q', 'cond_dim'
    ]})

    # --- 4. Run Training ---
    best_loss = train_model(model, loader, config, device, artifacts_dir)
    
    print(f"\nTraining complete. Final best loss: {best_loss:.6f}")
    print(f"Best model and config saved in: {artifacts_dir}")

if __name__ == "__main__":
    main()
