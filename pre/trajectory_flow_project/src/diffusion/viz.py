import matplotlib.pyplot as plt
import torch

def plot_trajectories(trajectories: torch.Tensor, labels: torch.Tensor, title: str = "Trajectories"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    class_names = ['circle', 'straight line', 'ellipse']
    colors = ['red', 'blue', 'green']

    # overview
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_trajs = trajectories[mask]
            for traj in class_trajs:
                axes[0].plot(traj[:, 0], traj[:, 1], color=colors[class_idx], alpha=0.6, linewidth=2)
    axes[0].set_title(f'{title} - overview')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # per class
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_trajs = trajectories[mask]
            for traj in class_trajs:
                axes[class_idx + 1].plot(traj[:, 0], traj[:, 1], color=colors[class_idx], alpha=0.7, linewidth=2)
        axes[class_idx + 1].set_title(f'{class_names[class_idx]} (n={int(mask.sum())})')
        axes[class_idx + 1].set_aspect('equal')
        axes[class_idx + 1].grid(True, alpha=0.3)
        axes[class_idx + 1].set_xlim(-1, 1)
        axes[class_idx + 1].set_ylim(-0.6, 0.6)
    
    plt.tight_layout()
    plt.show()
