import os, json, torch, random, numpy as np
from omegaconf import OmegaConf

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try: torch.set_float32_matmul_precision('high')
    except: pass

def save_cfg(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cfg.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

def save_ckpt(path, payload):
    torch.save(payload, path)

def save_ckpt_wrap(save_dir, ds, cfg):
    def fn(name, model, optim, sched, scaler, ema, epoch, step, best, dataset, cfg_):
        payload = {
            "epoch": epoch, "global_step": step, "best_val": best,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.shadow if ema is not None else None,
            "scalers": {
                "y": {"type": "minmax", "min": dataset.y_scaler.min_, "max": dataset.y_scaler.max_},
                "cond": {"type": "minmax", "min": dataset.c_scaler.min_, "max": dataset.c_scaler.max_},
            },
            "data_spec": dataset.spec,
            "data_sources": getattr(dataset, "sources", None),
            "cfg": OmegaConf.to_container(cfg_, resolve=True)
        }
        os.makedirs(save_dir, exist_ok=True)
        save_ckpt(os.path.join(save_dir, name), payload)
    return fn
