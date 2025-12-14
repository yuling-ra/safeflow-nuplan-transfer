import optuna, argparse, yaml, os, sys
from omegaconf import OmegaConf

# Add the src directory to the Python path to allow direct imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import run_training

def objective(trial: optuna.Trial, base_cfg: OmegaConf):
    """
    Optuna objective function for this specific sweep.
    """
    cfg = base_cfg.copy()

    # --- 1. Define the hyperparameter search space ---
    # We are sweeping learning rate and model capacity (via model_channels).
    
    lr = trial.suggest_categorical("train.lr", [1e-4, 5e-5])
    model_channels = trial.suggest_categorical("model.unet.model_channels", [32, 64, 96])
    
    # Update the config with the suggested values
    OmegaConf.update(cfg, "train.lr", lr, merge=True)
    OmegaConf.update(cfg, "model.unet.model_channels", model_channels, merge=True)

    # --- 2. Create a unique directory and experiment name for this trial ---
    trial_id = trial.number
    save_dir = os.path.join(cfg.log.save_dir, f"trial_{trial_id}_lr_{lr}_mc_{model_channels}")
    OmegaConf.update(cfg, "log.save_dir", save_dir, merge=True)
    
    if cfg.log.aim.enable:
        exp_name = f"{cfg.log.aim.experiment}_trial_{trial_id}"
        OmegaConf.update(cfg, "log.aim.experiment", exp_name, merge=True)

    print(f"\n===== Starting Trial {trial_id} =====")
    print(f"  - Params: {trial.params}")
    print(f"  - Save Directory: {save_dir}")
    print(f"================================\n")

    # --- 3. Run the training and return the value to optimize ---
    try:
        best_val_loss = run_training(cfg)
    except Exception as e:
        print(f"Trial {trial_id} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main():
    # Base configuration for this sweep
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'cbam.yaml')
    study_name = "CBAM_LR_Capacity_Sweep"
    storage_path = "sqlite:///optuna_cbam_sweep.db"
    
    print(f"Starting sweep with base config: {config_path}")
    print(f"Study Name: {study_name}")
    print(f"Database: {storage_path}")

    # Load base configuration
    with open(config_path, "r") as f:
        base_cfg = OmegaConf.create(yaml.safe_load(f))

    # Define the search space for the grid sampler
    search_space = {
        "train.lr": [1e-4, 5e-5],
        "model.unet.model_channels": [32, 64, 96],
    }
    
    study = optuna.create_study(
        direction="minimize",
        storage=storage_path,
        study_name=study_name,
        load_if_exists=True,
        sampler=optuna.samplers.GridSampler(search_space)
    )

    # Enqueue trials to ensure all 8 combinations are run
    study.enqueue_trial({"train.lr": 1e-4, "model.unet.model_channels": 32})
    study.enqueue_trial({"train.lr": 1e-4, "model.unet.model_channels": 64})
    study.enqueue_trial({"train.lr": 1e-4, "model.unet.model_channels": 96})
    study.enqueue_trial({"train.lr": 5e-5, "model.unet.model_channels": 32})
    study.enqueue_trial({"train.lr": 5e-5, "model.unet.model_channels": 64})
    study.enqueue_trial({"train.lr": 5e-5, "model.unet.model_channels": 96})

    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials=6 # Run exactly 6 trials
    )

    print("\n===== Sweep Finished =====")
    
    # 检查是否有成功的试验
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        print(f"Best trial found in study '{study_name}':")
        best_trial = study.best_trial
        print(f"  Value (min val loss): {best_trial.value}")
        print(f"  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print(f"No successful trials found in study '{study_name}'.")
        print("All trials were pruned due to errors.")
        
        # 显示失败的试验信息
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        print(f"Total failed trials: {len(failed_trials)}")
        
    print("==========================")


if __name__ == "__main__":
    main() 