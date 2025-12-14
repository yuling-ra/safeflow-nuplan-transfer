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
                         preprocess_incremental_pose, reconstruct_incremental_pose)
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
        quat = frame[3:]
        T_seq[i, :3, :3] = quaternion_to_matrix(quat)
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
            x1 = batch[0].to(device)
            x0, t = torch.randn_like(x1), torch.rand(x1.shape[0], device=device)
            s = path.sample(t=t, x_0=x0, x_1=x1)
            
            # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
            pred_v = model(s.x_t, s.t)
            loss = F.mse_loss(pred_v, s.dx_t)

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
    run_name = f"main_67_firsttry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir = script_dir / 'training_runs' / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved to: {artifacts_dir}")

    # --- 2. Data Loading ---
    all_trajectories = []
    dataset_zips = ['dataset/gestor_pouring_dataset_6.zip', 'dataset/gestor_pouring_dataset_7.zip']
    
    for zip_file in dataset_zips:
        zip_path = script_dir / zip_file
        json_files = get_json_files_from_zip(zip_path)
        for json_f in json_files:
            T_seq_raw, _ = load_trajectory_from_json_in_zip(zip_path, json_f)
            T_seq_aligned, _ = align_to_first_frame(T_seq_raw)
            # Note: We need a dummy dir for preprocess, but stats won't be used from there
            Z_delta, _ = preprocess_incremental_pose(T_seq_aligned, artifacts_dir)
            all_trajectories.append(torch.from_numpy(Z_delta).float())

    max_len = max(traj.shape[0] for traj in all_trajectories)
    padded_trajectories = [F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in all_trajectories]
    data_tensor = torch.stack(padded_trajectories)
    print(f"Loaded {data_tensor.shape[0]} trajectories, padded to length {max_len}.")

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

    dataset = torch.utils.data.TensorDataset(data_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = EnhancedVectorFieldNet(**{k: v for k, v in config.items() if k in ['seq_len', 'data_dim', 'hidden_dim', 'num_layers', 'time_embed_dim', 'dropout', 'num_heads', 'reduction', 'window', 'fourier_k', 'bridge_k', 'bridge_q']})

    # --- 4. Run Training ---
    best_loss = train_model(model, loader, config, device, artifacts_dir)
    
    print(f"\nTraining complete. Final best loss: {best_loss:.6f}")
    print(f"Best model and config saved in: {artifacts_dir}")

if __name__ == "__main__":
    main()
