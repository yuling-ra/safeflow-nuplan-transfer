import torch
from diffusion.train import train_diffusion_model
from diffusion.sampling import generate_trajectories
from diffusion.viz import plot_trajectories

def main():
    unet, cond_path, losses, ema = train_diffusion_model(
        num_epochs=30000, batch_size=128, lr=3e-4, use_ema=True
    )

    # swap to EMA weights for sampling (if available)
    if ema is not None:
        ema.copy_to(unet)

    gen_trajs, gen_labels = generate_trajectories(unet, cond_path, num_samples=12, num_steps=200, use_heun=True)
    plot_trajectories(gen_trajs, gen_labels, title="Generated trajectories")

    # (optional) compare with real
    real_trajs, real_labels = cond_path.p_data.sample(12)
    plot_trajectories(real_trajs.cpu(), real_labels.cpu(), title="Real trajectories")

if __name__ == "__main__":
    main()
