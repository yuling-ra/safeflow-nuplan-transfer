import argparse, yaml, torch
from omegaconf import OmegaConf
from models import build_model
from data import build_loaders
from engine import train_loop
from logger import MultiLogger
from utils import seed_everything, save_cfg, save_ckpt_wrap

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    # 简易覆盖：例如 --override "train.optimizer.lr=3e-4,model.attn.enable_cbam=false"
    p.add_argument("--override", type=str, default="")
    return p.parse_args()

def apply_override(cfg, override_str):
    if not override_str: return cfg
    for kv in override_str.split(","):
        if not kv: continue
        k, v = kv.split("=")
        # 粗暴转型
        if v.lower() in ["true","false"]: v = v.lower()=="true"
        else:
            try:
                if "." in v: v=float(v)
                else: v=int(v)
            except: pass
        OmegaConf.update(cfg, k.strip(), v, merge=True)
    return cfg

def run_training(cfg):
    """
    Reusable training function that can be called from main script or sweep script.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.train.seed)

    # 数据
    train_loader, val_loader, ds = build_loaders(cfg)
    y_dim   = ds.y_dim
    cond_dim= ds.cond_dim

    # 模型
    model = build_model(cfg, y_dim=y_dim, cond_dim=cond_dim).to(device)

    # 日志
    loggers = MultiLogger(cfg.log.aim, cfg.log.csv, cfg.log.save_dir)
    save_dir = loggers.dir()
    save_cfg(cfg, save_dir)
    loggers.log_params({"y_dim": y_dim, "cond_dim": cond_dim, "model_params": sum(p.numel() for p in model.parameters())})

    # 訓練
    save_ckpt_fn = save_ckpt_wrap(save_dir, ds, cfg)
    best_val_loss = train_loop(model, train_loader, val_loader, ds, cfg, loggers, save_ckpt_fn, device)
    
    loggers.close()
    return best_val_loss

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    cfg = apply_override(cfg, args.override)

    run_training(cfg)

if __name__ == "__main__":
    main()
