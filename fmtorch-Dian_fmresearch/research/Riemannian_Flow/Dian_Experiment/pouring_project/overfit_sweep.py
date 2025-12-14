import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchdiffeq
import time
import json
from pathlib import Path
import contextlib
import bitsandbytes as bnb
from datetime import datetime
import itertools
import zipfile
import io
from scipy.spatial.transform import Rotation as R


# fmtorch components
from fmtorch.scheduler import CondOTScheduler
from fmtorch.path import AffinePath

# Local project components
from preprocess import (align_to_first_frame, 
                         preprocess_incremental_pose, reconstruct_incremental_pose)
from model import EnhancedVectorFieldNet
from visualization import (evaluate_reconstruction, plot_sweep_results, 
                            plot_fm_vs_recon_error, plot_final_trajectory_comparison,
                            visualize_3d_trajectories)

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
            
    # Assuming the format is [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    trajectory_raw = data['trajectory']
    
    T_seq = np.zeros((len(trajectory_raw), 4, 4))
    for i, frame in enumerate(trajectory_raw):
        T_seq[i, :3, 3] = frame[:3]
        # Scipy expects quaternion as [x, y, z, w]
        quat = frame[3:]
        T_seq[i, :3, :3] = quaternion_to_matrix(quat)
        T_seq[i, 3, 3] = 1.0
        
    return T_seq, {'traj': T_seq} # Mimic original data structure

def get_json_files_from_zip(zip_path):
    """Lists all .json files in a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        return [name for name in z.namelist() if name.endswith('.json')]

# ==============================================================================
# Training and Inference Functions
# ==============================================================================

def train_flow_matching(model, loader, config, device):
    """Trains the vector field model using Flow Matching across the entire dataset."""
    model = model.to(device)
    path = AffinePath(scheduler=CondOTScheduler())
    
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-5)
    )
    
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=config.get('lr', 1e-3) * 0.01)
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model.train()
    loss_history, best_loss = [], float('inf')
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            
            x1 = batch[0].to(device)
            x0, t = torch.randn_like(x1), torch.rand(x1.shape[0], device=device)
            s = path.sample(t=t, x_0=x0, x_1=x1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                pred_v = model(s.x_t, s.t)
                loss = F.mse_loss(pred_v, s.dx_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        best_loss = min(best_loss, avg_epoch_loss)
        
        if scheduler:
            scheduler.step(avg_epoch_loss if isinstance(scheduler, ReduceLROnPlateau) else None)
            
        if (epoch + 1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{config['n_epochs']} | Avg Loss: {avg_epoch_loss:.6f} | Best: {best_loss:.6f} | LR: {lr:.6f} | Time: {time.time() - start_time:.1f}s")
    
    return best_loss, loss_history

# ==============================================================================
# Main Orchestration
# ==============================================================================

def main():
    # --- 1. Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir / 'sweep_artifacts' / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved to: {artifacts_dir}")

    # --- 2. Data Loading and Preprocessing ---
    # Load all trajectories from datasets 6 and 7
    all_trajectories = []
    dataset_zips = ['dataset/gestor_pouring_dataset_6.zip', 'dataset/gestor_pouring_dataset_7.zip']
    
    for zip_file in dataset_zips:
        zip_path = script_dir / zip_file
        json_files = get_json_files_from_zip(zip_path)
        for json_f in json_files:
            T_seq_raw, _ = load_trajectory_from_json_in_zip(zip_path, json_f)
            # Find the longest trajectory to pad others
            T_seq_aligned, _ = align_to_first_frame(T_seq_raw)
            Z_delta, _ = preprocess_incremental_pose(T_seq_aligned, artifacts_dir)
            all_trajectories.append(torch.from_numpy(Z_delta).float())

    # Pad all trajectories to the same length
    max_len = max(traj.shape[0] for traj in all_trajectories)
    padded_trajectories = [F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in all_trajectories]
    data_tensor = torch.stack(padded_trajectories) # [num_trajectories, max_len, data_dim]
    print(f"Loaded and processed {data_tensor.shape[0]} trajectories. Padded to length {max_len}.")

    # --- 3. Hyperparameter Sweep Definition ---
    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4],
        'scheduler': ['cosine', 'plateau'],
        'batch_size': [16, 32, 64], # This will now mean number of trajectories per batch
        'hidden_dim': [256, 512],
        'num_layers': [4, 8],
    }

    base_config = {
        'data_dim': data_tensor.shape[2], 
        'n_epochs': 5000, 
        'time_embed_dim': 256, 
        'dropout': 0.1, 
        'weight_decay': 1e-5, 
        'grad_clip': 1.0, 
        'num_heads': 4, 
        'reduction': 16, 
        'window': 96, 
        'fourier_k': 32
    }

    keys, values = zip(*param_grid.items())
    sweep_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    sweep_results = []
    print(f"\nStarting hyperparameter sweep with {len(sweep_configs)} configurations...")

    for i, hyperparams in enumerate(sweep_configs):
        config = {**base_config, **hyperparams}
        config['seq_len'] = max_len
        
        run_name = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
        print(f"\n[Sweep {i+1}/{len(sweep_configs)}] Config: {run_name}")

        # Create data loader for the current batch size
        dataset = torch.utils.data.TensorDataset(data_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        model = EnhancedVectorFieldNet(**{k: v for k, v in config.items() if k in ['seq_len', 'data_dim', 'hidden_dim', 'num_layers', 'time_embed_dim', 'dropout', 'num_heads', 'reduction', 'window', 'fourier_k']})
        
        # Train on the full dataset by passing the loader
        best_loss, loss_history = train_flow_matching(model, loader, config, device)
        
        result = {
            'name': run_name, 
            'config': config, 
            'best_loss': best_loss, 
            'final_loss': loss_history[-1] if loss_history else None,
            # 'loss_history': loss_history # optional, can make JSON large
        }
        sweep_results.append(result)

        # Save results intermittently
        with open(artifacts_dir / 'sweep_results.json', 'w') as f:
            json.dump(sweep_results, f, indent=2)
    
    # --- 4. Finalization ---
    print("\nSweep complete.")
    # Sort results by best loss
    sweep_results.sort(key=lambda r: r['best_loss'])
    
    print("Top 5 configurations:")
    for res in sweep_results[:5]:
        print(f"  - {res['name']}: Best Loss = {res['best_loss']:.6f}")

    final_results_path = artifacts_dir / 'sweep_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
        
    print(f"\nAll sweep results saved to {final_results_path}")

if __name__ == "__main__":
    main()
