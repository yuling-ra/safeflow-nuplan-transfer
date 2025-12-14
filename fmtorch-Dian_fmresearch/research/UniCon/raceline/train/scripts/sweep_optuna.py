# scripts/sweep_optuna.py
import optuna, argparse, yaml, os, sys
from omegaconf import OmegaConf

# Add the src directory to the Python path to allow direct imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import run_training

def objective(trial: optuna.Trial, base_cfg: OmegaConf):
    """
    Optuna objective function. 
    This function takes a trial object, suggests hyperparameters,
    updates the configuration, runs the training, and returns the metric to optimize.
    """
    cfg = base_cfg.copy()

    # --- 1. Define the hyperparameter search space ---
    # You can add or modify any parameter from your YAML file here.
    
    # Optimizer parameters
    trial.suggest_float("train.lr", 1e-5, 1e-3, log=True)
    trial.suggest_float("train.weight_decay", 1e-4, 1e-1, log=True)
    
    # Model architecture parameters
    trial.suggest_int("model.width", 512, 2048, step=256)
    trial.suggest_int("model.depth", 3, 6)
    
    # UNet specific parameters
    trial.suggest_categorical("model.unet.model_channels", [32, 64, 96])
    trial.suggest_int("model.unet.num_res_blocks", 1, 3)

    # If using the CBAM model, you can also sweep its parameters
    if cfg.model.name == "flow_unet_cbam":
        trial.suggest_int("model.cbam.heads", 2, 8, step=2)
        trial.suggest_categorical("model.cbam.reduction", [8, 16, 32])

    # --- 2. Create a unique directory and experiment name for this trial ---
    trial_id = trial.number
    save_dir = os.path.join(cfg.log.save_dir, f"trial_{trial_id}")
    OmegaConf.update(cfg, "log.save_dir", save_dir, merge=True)
    
    if cfg.log.aim.enable:
        exp_name = f"{cfg.log.aim.experiment}_trial_{trial_id}"
        OmegaConf.update(cfg, "log.aim.experiment", exp_name, merge=True)

    print(f"\n===== Starting Trial {trial_id} =====")
    print(f"  - Params: {trial.params}")
    print(f"  - Save Directory: {save_dir}")
    print(f"================================\n")

    # --- 3. Run the training and return the value to optimize ---
    # The `train_loop` in engine.py must return the best validation loss.
    try:
        best_val_loss = run_training(cfg)
    except Exception as e:
        print(f"Trial {trial_id} failed with error: {e}")
        # Prune the trial if it fails (e.g., CUDA out of memory)
        raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to the base YAML configuration file.")
    p.add_argument("--n_trials", type=int, default=50, help="Number of trials to run.")
    p.add_argument("--storage", type=str, default="sqlite:///optuna_sweep.db", help="Optuna storage URL. Can be a file or a database URL.")
    p.add_argument("--study_name", type=str, default="racetrack_fm_sweep", help="Name of the study for organização.")
    args = p.parse_args()

    # Load base configuration
    with open(args.config, "r") as f:
        base_cfg = OmegaConf.create(yaml.safe_load(f))

    # Create or load the study
    study = optuna.create_study(
        direction="minimize",  # We want to minimize validation loss
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True
    )

    # Start the optimization
    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials=args.n_trials
    )

    # Print results
    print("\n===== Sweep Finished =====")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"\nBest trial:")
    print(f"  Value (min val loss): {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("==========================")


if __name__ == "__main__":
    main() 