import torch, numpy as np, os, sys, yaml
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add src to sys.path
# No longer needed if run from `train/` directory, but kept for robustness
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import FullTrajDataset
from src.inference import split_full_vector
from src.models import FlowUNet_CBAM, build_model
from raceline_core import load_map, MapData, OCPConfig, simulate_vehicle_bc


def visualize_inference_result(
    model_output_states, actions, 
    actual_rollout_states, generated_rollout_states, 
    map_data: MapData, dataset_name: str, save_dir: str
):
    """
    在赛道地图上可视化生成的状态轨迹，并单独绘制动作历史。
    同时对比显示基于动力学模型的两种前向推演轨迹。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 22), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Inference & Dynamics Rollout for {os.path.basename(dataset_name)}', fontsize=16)

    # --- 1. 状态轨迹可视化 (on Map) ---
    ax1 = axes[0]
    ax1.imshow(map_data.grid_map, cmap='gray', origin='lower', alpha=0.6)
    ax1.plot(map_data.raceline_grid[:, 0], map_data.raceline_grid[:, 1], 'g--', linewidth=1.5, label='Reference Raceline')
    
    # 绘制三条轨迹
    ax1.plot(model_output_states[:, 0], model_output_states[:, 1], 'r-', linewidth=2.5, label='Model Output Trajectory', zorder=10)
    ax1.plot(actual_rollout_states[:, 0], actual_rollout_states[:, 1], 'c--', linewidth=2, label='Rollout from Actual x0', zorder=9)
    ax1.plot(generated_rollout_states[:, 0], generated_rollout_states[:, 1], 'm:', linewidth=2, label='Rollout from Generated x0', zorder=8)
    
    # 标记起点
    ax1.plot(actual_rollout_states[0, 0], actual_rollout_states[0, 1], 'g^', markersize=12, label='Start (Actual Data)', markeredgecolor='k')
    ax1.plot(model_output_states[0, 0], model_output_states[0, 1], 'bo', markersize=10, label='Start (Model Output)')
    
    ax1.set_title('State Rollout Comparison (Model Output vs. Dynamics Simulation)')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_aspect('equal')

    # --- 2. 动作和速度历史可视化 ---
    ax2 = axes[1]
    state_time_steps = np.arange(model_output_states.shape[0])
    action_time_steps = np.arange(actions.shape[0])
    
    # 速度
    line1 = ax2.plot(state_time_steps, model_output_states[:, 3], 'b-', label='Velocity (pix/s)', linewidth=2)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Velocity', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 加速度和转向 (共享X轴)
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(action_time_steps, actions[:, 0], 'r-', label='Acceleration', linewidth=2, alpha=0.7)
    line3 = ax2_twin.plot(action_time_steps, actions[:, 1], 'g-', label='Steering (rad)', linewidth=2, alpha=0.7)
    ax2_twin.set_ylabel('Control Inputs')
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title('Action Rollout (Controls and Velocity over Time)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 保存图像 ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    save_path = os.path.join(save_dir, f'inference_visualization_{base_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close(fig)


def main():
    # --- 1. 配置路径 ---
    base_dir = os.path.dirname(__file__)
    trial_output_dir = os.path.join(base_dir, 'scripts/outputs/cbam_base_run/trial_13_lr_0.0001_mc_64/run_20250923_173413')
    
    ckpt_path = os.path.join(trial_output_dir, 'best.pt')
    cfg_path = os.path.join(trial_output_dir, 'cfg.yaml')
    
    # 数据集和地图路径
    dataset_dir = os.path.join(base_dir, 'dataset')
    map_path = os.path.join(base_dir, '..', 'nuerburgring_segment_map.npz') # Map is in parent `raceline` dir
    vis_output_dir = os.path.join(trial_output_dir, 'visualizations') # 保存可视化结果的目录
    
    dataset_names = [
        'ocp_dataset_incremental.npz',
        'ocp_dataset_incremental_99743ee3.npz',
        'ocp_dataset_inverse_2a1131b3.npz',
        'ocp_dataset_inverse_6dc436f8.npz' # Based on training logs
    ]
    dataset_paths = [os.path.join(dataset_dir, name) for name in dataset_names]

    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Loading config from: {cfg_path}")

    # --- 2. 加载配置和模型 ---
    with open(cfg_path, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # Add a default inference config if it's missing, as it's not in the training config
    default_infer_cfg = OmegaConf.create({
        "infer": {
            "ode": {
                "method": "dopri5",
                "atol": 1e-5,
                "rtol": 1e-5,
            }
        }
    })
    cfg = OmegaConf.merge(default_infer_cfg, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 3. 加载地图、数据 Scaler 和维度信息 ---
    print("\nLoading map data...")
    try:
        map_data = load_map(map_path)
        print(f"Map '{os.path.basename(map_path)}' loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Map file not found at '{map_path}'. Cannot generate visualizations.")
        return

    print("\nInitializing dataset to get data scalers and dimensions...")
    full_dataset = FullTrajDataset(path_or_paths=dataset_paths, theta_sincos=cfg.data.theta_sincos)
    y_scaler = full_dataset.y_scaler
    c_scaler = full_dataset.c_scaler
    y_dim = full_dataset.y_dim
    cond_dim = full_dataset.cond_dim
    print(f"Scalers created. y_dim={y_dim}, cond_dim={cond_dim}")

    # --- 2.b 加载模型 (修正) ---
    # 使用 build_model 函数来保证与训练时完全一致的模型结构
    model = build_model(cfg, y_dim=y_dim, cond_dim=cond_dim).to(device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model'])
    print("Model loaded successfully.")

    # --- 4. 从每个数据集中选择一个样本并进行推理 ---
    for i, (path, name) in enumerate(zip(dataset_paths, dataset_names)):
        print(f"\n--- Processing sample from: {name} ---")
        try:
            with np.load(path) as data:
                # 选择第一个 trajectory 的 x0s 作为 condition
                sample_x0s_raw = data['x0s'][0] 
        except Exception as e:
            print(f"Could not load data from {name}. Skipping. Error: {e}")
            continue

        # 归一化 condition
        # theta_sincos 转换
        if cfg.data.theta_sincos:
            theta = sample_x0s_raw[2]
            cond_raw = np.array([
                sample_x0s_raw[0], sample_x0s_raw[1],
                np.sin(theta), np.cos(theta),
                sample_x0s_raw[3]
            ])
        else:
            cond_raw = sample_x0s_raw

        # --- 5. 执行推理 (修正) ---
        # `sample_full_vector` 需要 `y` 作为初始噪声输入，这里我们直接从数据集中获取
        # 另外，模型的 forward 签名是 (y, t, condition)
        
        model.eval()
        with torch.no_grad():
            cond_norm = c_scaler.normalize(cond_raw).astype(np.float32)
            cond = torch.from_numpy(cond_norm).unsqueeze(0).to(device)
            
            # 模拟 `torchdiffeq.odeint` 的过程，但使用模型的 forward 签名
            def ode_fun(t, y_flat):
                y = y_flat.view(1, y_dim)
                t_tensor = torch.tensor([t.item()], device=device, dtype=torch.float32)
                # 关键修正：将 x, t, condition 对应到模型的 y, t, condition
                v = model(y, t_tensor, condition=cond)
                return v.view(-1)

            import torchdiffeq
            t_span = torch.tensor([0., 1.], device=device)
            x0 = torch.randn(1, y_dim, device=device) # Initial noise
            
            res = torchdiffeq.odeint(ode_fun, x0.view(-1), t_span,
                                     atol=cfg.infer.ode.atol, rtol=cfg.infer.ode.rtol,
                                     method=cfg.infer.ode.method)
            y_hat_normalized = res[-1].view(y_dim).cpu().numpy()

        # --- 6. 反归一化和解析结果 ---
        y_hat_denormalized = y_scaler.denorm(y_hat_normalized)
        actions, states = split_full_vector(y_hat_denormalized, full_dataset.spec)

        print("Inference successful. Result shapes:")
        print(f"  - Actions: {actions.shape}")
        print(f"  - States: {states.shape}")
        # print("Example output (first 5 steps):")
        # print("Actions:\n", actions[:5])
        # print("States:\n", states[:5])

        # --- 7. 为对比进行前向动力学模拟 ---
        print("Performing forward dynamics simulation for comparison...")
        sim_config = OCPConfig()
        L_pix = sim_config.wheelbase_m * map_data.resolution

        # 推演 1: 从数据集的 *真实* 初始状态开始
        actual_rollout_states = simulate_vehicle_bc(
            x0=sample_x0s_raw,
            u_seq=actions,
            dt=sim_config.dt,
            state_steps=sim_config.state_steps,
            L_pix=L_pix,
            delta_rate_max=sim_config.delta_rate_max
        )

        # 推演 2: 从模型生成的轨迹的 *第一个* 状态开始
        generated_rollout_states = simulate_vehicle_bc(
            x0=states[0],
            u_seq=actions,
            dt=sim_config.dt,
            state_steps=sim_config.state_steps,
            L_pix=L_pix,
            delta_rate_max=sim_config.delta_rate_max
        )

        # --- 8. 可视化结果 ---
        visualize_inference_result(
            model_output_states=states, 
            actions=actions, 
            actual_rollout_states=actual_rollout_states,
            generated_rollout_states=generated_rollout_states,
            map_data=map_data, 
            dataset_name=name, 
            save_dir=vis_output_dir
        )

if __name__ == "__main__":
    main() 