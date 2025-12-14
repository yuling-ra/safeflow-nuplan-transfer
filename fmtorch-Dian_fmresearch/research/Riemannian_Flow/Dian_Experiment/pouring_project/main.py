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

# fmtorch components
from fmtorch.scheduler import CondOTScheduler
from fmtorch.path import AffinePath

# Local project components
from .preprocess import (load_pouring_data, align_to_first_frame, 
                         preprocess_incremental_pose, reconstruct_incremental_pose)
from .model import EnhancedVectorFieldNet
from .visualization import (evaluate_reconstruction, plot_sweep_results, 
                            plot_fm_vs_recon_error, plot_final_trajectory_comparison,
                            visualize_3d_trajectories)

# ==============================================================================
# Training and Inference Functions
# ==============================================================================

def train_flow_matching(model, data, config, device):
    """Trains the vector field model using Flow Matching."""
    model = model.to(device)
    data = data.to(device)
    path = AffinePath(scheduler=CondOTScheduler())
    
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=config['lr'] * 0.01)
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model.train()
    loss_history, best_loss = [], float('inf')
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config['acc_steps']):
            x1 = data.expand(config['micro_batch_size'], -1, -1)
            x0, t = torch.randn_like(x1), torch.rand(config['micro_batch_size'], device=device)
            s = path.sample(t=t, x_0=x0, x_1=x1)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                pred_v = model(s.x_t, s.t)
                loss = F.mse_loss(pred_v, s.dx_t) / config['acc_steps']
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        
        loss_val = accumulated_loss * config['acc_steps']
        loss_history.append(loss_val)
        best_loss = min(best_loss, loss_val)
        
        if scheduler:
            scheduler.step(loss_val if isinstance(scheduler, ReduceLROnPlateau) else None)
            
        if (epoch + 1) % 500 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{config['n_epochs']} | Loss: {loss_val:.6f} | Best: {best_loss:.6f} | LR: {lr:.6f} | Time: {time.time() - start_time:.1f}s")
    
    return best_loss, loss_history

def run_inference(model, device):
    """Generates a trajectory from noise using the trained model."""
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(1, 479, 6, device=device)
        
        def ode_func(t, x_flat):
            x = x_flat.view(1, 479, 6)
            t_tensor = torch.tensor([float(t)], device=device)
            v = model(x, t_tensor)
            return v.view(-1)
        
        t_span = torch.tensor([0.0, 1.0], device=device)
        result = torchdiffeq.odeint(
            ode_func, x0.view(-1), t_span, method='dopri5', rtol=1e-5, atol=1e-5
        )
        Z_delta_sampled = result[-1].view(1, 479, 6).cpu().numpy()[0]
    print("Inference complete.")
    return Z_delta_sampled

# ==============================================================================
# Main Orchestration
# ==============================================================================

def main():
    # --- 1. Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir / 'artifacts' / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved to: {artifacts_dir}")

    # --- 2. Data Loading and Preprocessing ---
    pkl_path = script_dir.parent.parent / 'MMLfD-Tutorial_Modified/datasets/pouring_data/1_water_200.pkl'
    T_seq_raw, raw_data_dict = load_pouring_data(pkl_path)
    T_seq_aligned, T_0 = align_to_first_frame(T_seq_raw)
    Z_delta, whiten_stats = preprocess_incremental_pose(T_seq_aligned, artifacts_dir)
    Z_delta_tensor = torch.from_numpy(Z_delta).float().unsqueeze(0)

    # --- 3. Hyperparameter Sweep and Training ---
    base_config = {'seq_len': 479, 'data_dim': 6, 'n_epochs': 5000, 'micro_batch_size': 8, 'acc_steps': 64, 'time_embed_dim': 256, 'dropout': 0.0, 'weight_decay': 1e-5, 'grad_clip': 1.0, 'num_heads': 2, 'reduction': 16, 'window': 96, 'fourier_k': 32}
    sweep_configs = [
        {**base_config, 'hidden_dim': 256, 'num_layers': 4, 'lr': 1e-3, 'scheduler': 'cosine', 'name': 'h256_l4_cosine'},
        {**base_config, 'hidden_dim': 256, 'num_layers': 4, 'lr': 1e-3, 'scheduler': 'plateau', 'name': 'h256_l4_plateau'}
    ]
    
    sweep_results, best_overall_loss = [], float('inf')
    for i, config in enumerate(sweep_configs):
        print(f"\n[Sweep {i+1}/{len(sweep_configs)}] Config: {config['name']}")
        model = EnhancedVectorFieldNet(**{k: v for k, v in config.items() if k in ['seq_len', 'data_dim', 'hidden_dim', 'num_layers', 'time_embed_dim', 'dropout', 'num_heads', 'reduction', 'window', 'fourier_k']})
        best_loss, loss_history = train_flow_matching(model, Z_delta_tensor, config, device)
        
        result = {'name': config['name'], 'config': config, 'best_loss': best_loss, 'final_loss': loss_history[-1], 'loss_history': loss_history}
        sweep_results.append(result)
        
        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_model_path = artifacts_dir / 'best_model.pt'
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'best_loss': best_loss, 'whiten_stats': whiten_stats, 'T_0': T_0}, best_model_path)
            print(f"  ✓ New best model saved to: {best_model_path}")

    # --- 4. Load Best Model and Run Inference ---
    checkpoint = torch.load(best_model_path, map_location=device)
    best_config = checkpoint['config']
    model = EnhancedVectorFieldNet(**{k: v for k, v in best_config.items() if k in ['seq_len', 'data_dim', 'hidden_dim', 'num_layers', 'time_embed_dim', 'dropout', 'num_heads', 'reduction', 'window', 'fourier_k']})
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    Z_delta_sampled = run_inference(model, device)

    # --- 5. Reconstruct Trajectory and Evaluate ---
    T_fm_abs = reconstruct_incremental_pose(Z_delta_sampled, checkpoint['whiten_stats'], checkpoint['T_0'], T_seq_aligned[0])
    metrics_fm, rot_err_fm, trans_err_fm = evaluate_reconstruction(T_seq_raw, T_fm_abs, 'FM-ΔSE(3)')
    print("\n--- FM Generation Metrics ---")
    print(json.dumps(metrics_fm, indent=2))
    
    # Also get recon metrics for comparison
    T_recon_delta_abs = reconstruct_incremental_pose(Z_delta, whiten_stats, T_0, T_seq_aligned[0])
    _, rot_err_delta, trans_err_delta = evaluate_reconstruction(T_seq_raw, T_recon_delta_abs, 'Recon-ΔSE(3)')

    # --- 6. Visualization ---
    results_summary = sorted([{'name': r['name'], **r['config'], 'best_loss': r['best_loss']} for r in sweep_results], key=lambda x: x['best_loss'])
    plot_sweep_results(results_summary, sweep_results, artifacts_dir / 'sweep_results.png')
    plot_fm_vs_recon_error(rot_err_delta, trans_err_delta, rot_err_fm, trans_err_fm, artifacts_dir / 'fm_vs_recon_error.png')
    plot_final_trajectory_comparison(T_seq_raw, T_recon_delta_abs, T_fm_abs, artifacts_dir / 'final_trajectory_comparison.png')
    visualize_3d_trajectories(
        [T_seq_raw, T_fm_abs], ["Ground Truth", "FM Generated"], 
        [[0.1, 0.8, 0.1], [0.9, 0.2, 0.2]], raw_data_dict, skip_size=20,
        save_path=artifacts_dir / 'trajectories_for_visualizer.npz'
    )
    print(f"\nExperiment complete. All artifacts saved in {artifacts_dir}")

if __name__ == "__main__":
    main()
