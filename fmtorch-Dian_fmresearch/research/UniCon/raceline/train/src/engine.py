import math, time, torch, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
from fmtorch.path import AffinePath
from fmtorch.scheduler import CondOTScheduler

@dataclass
class EMAModel:
    model: torch.nn.Module
    decay: float=0.999
    def __post_init__(self):
        self.shadow = {k: v.detach().clone() for k,v in self.model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)
    def store(self):
        self.backup = {k: v.detach().clone() for k,v in self.model.state_dict().items()}
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)
    def restore(self):
        self.model.load_state_dict(self.backup, strict=False)

def build_optim_sched(model, cfg, total_steps: int):
    if cfg.train.optimizer.name.lower() == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.wd, betas=tuple(cfg.train.optimizer.betas)
        )
    else:
        raise NotImplementedError
    warmup = cfg.train.scheduler.warmup_steps
    min_lr = cfg.train.scheduler.min_lr
    base_lr= cfg.train.optimizer.lr

    def lr_lambda(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, (total_steps - warmup))
        return min_lr/base_lr + 0.5*(1+math.cos(math.pi*prog))*(1 - min_lr/base_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return optim, sched

def flow_matching_step(model, cond, target, device):
    # cond:(B,c), target:(B,D)
    cond, target = cond.to(device, non_blocking=True), target.to(device, non_blocking=True)
    B, D = target.shape
    x0 = torch.randn_like(target)
    t  = torch.rand(B, device=device)
    path = AffinePath(scheduler=CondOTScheduler())
    sample = path.sample(t=t, x_0=x0, x_1=target)
    v_pred = model(sample.x_t, sample.t, condition=cond)
    loss = F.mse_loss(v_pred, sample.dx_t)
    return loss

def train_loop(model, train_loader, val_loader, ds, cfg, loggers, save_ckpt_fn, device):
    epochs = cfg.train.epochs
    steps_per_epoch = math.ceil(len(train_loader.dataset) / (cfg.data.batch_size))
    total_steps = epochs * steps_per_epoch
    optim, sched = build_optim_sched(model, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.train.amp)
    ema = EMAModel(model, decay=cfg.train.ema.decay) if cfg.train.ema.enable else None

    best_val = float("inf"); global_step = 0

    for ep in range(epochs):
        model.train()
        t0 = time.time(); running = 0.0
        for it, (cond, y) in enumerate(train_loader):
            with autocast(enabled=cfg.train.amp):
                loss = flow_matching_step(model, cond, y, device)
            scaler.scale(loss/cfg.train.grad_accum_steps).backward()

            if (it+1) % cfg.train.grad_accum_steps == 0:
                if cfg.train.clip_grad_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
                sched.step(); global_step += 1
                if ema: ema.update(model)

            running += loss.item()
            if global_step % cfg.train.log_interval == 0:
                lr = sched.get_last_lr()[0]
                loggers.log_scalar("train/loss", running/cfg.train.log_interval, global_step)
                loggers.log_scalar("train/lr", lr, global_step)
                running = 0.0

            if cfg.train.val_interval_steps>0 and global_step % cfg.train.val_interval_steps == 0:
                val = validate(model, val_loader, device, ema=ema, cfg=cfg)
                loggers.log_scalar("val/loss", val, global_step)
                if val < best_val:
                    best_val = val
                    save_ckpt_fn("best.pt", model, optim, sched, scaler, ema, ep, global_step, best_val, ds, cfg)

        if (ep+1) % cfg.train.save_every_epoch == 0:
            save_ckpt_fn("last.pt", model, optim, sched, scaler, ema, ep, global_step, best_val, ds, cfg)

    return best_val

def validate(model, val_loader, device, ema=None, cfg=None):
    was_train = model.training
    if ema: ema.store(); ema.copy_to(model)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for cond, y in val_loader:
            loss = flow_matching_step(model, cond, y, device)
            tot += loss.item() * cond.size(0); n += cond.size(0)
    if ema: ema.restore()
    if was_train: model.train()
    return tot / max(1, n)
