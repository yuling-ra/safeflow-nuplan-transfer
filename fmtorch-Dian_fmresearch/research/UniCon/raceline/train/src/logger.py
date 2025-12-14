import os, csv, aim
from datetime import datetime

class AIMLogger:
    def __init__(self, enable=True, repo="aimlogs", experiment="default"):
        self.enable = enable
        self.run = None
        if enable:
            self.run = aim.Run(repo=repo, experiment=experiment)
    def log_scalar(self, name, val, step):
        if self.enable: self.run.track(val, name=name, step=step)
    def log_params(self, params: dict):
        if self.enable:
            self.run.set("hparams", params, strict=False)
    def log_figure(self, name, fig, step):
        if self.enable: self.run.track(aim.Figure(fig), name=name, step=step)
    def close(self):
        if self.run: self.run.close()

class CSVLogger:
    def __init__(self, enable=True, path=None):
        self.enable = enable
        self.path = path
        if enable:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(["step","name","value"])
    def log_scalar(self, name, val, step):
        if not self.enable: return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step, name, float(val)])
    def log_params(self, params: dict): pass
    def log_figure(self, name, fig, step): pass

class MultiLogger:
    def __init__(self, aim_cfg, csv_cfg, save_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(save_dir, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.aim = AIMLogger(aim_cfg.enable, aim_cfg.repo, aim_cfg.experiment)
        self.csv = CSVLogger(csv_cfg.enable, os.path.join(self.run_dir, "metrics.csv"))
    def log_scalar(self, name, val, step):
        self.aim.log_scalar(name, val, step)
        self.csv.log_scalar(name, val, step)
    def log_params(self, params: dict):
        self.aim.log_params(params)
    def dir(self): return self.run_dir
    def close(self):
        self.aim.close()
