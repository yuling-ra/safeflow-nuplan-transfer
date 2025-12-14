# traj2-project

把 `Traj2.ipynb` 自动整理成了一个可以直接在 VS Code 里跑/调试/打包的 Python 项目。

## 目录结构

```
traj2-project/
├── .vscode/              # VS Code 调试 & 任务配置
├── notebooks/            # 原始 Notebook 备份
│   └── Traj2.ipynb
├── scripts/
│   └── run.py            # 命令行入口（调试/运行用）
├── src/
│   └── traj2/
│       ├── __init__.py
│       └── main.py       # 从 Notebook 提取的代码（保留了 cell 标记）
├── tests/
│   └── test_smoke.py
├── pyproject.toml        # 最小可打包配置（`pip install .`）
├── requirements.txt      # 从导入语句里自动推断的依赖（如有）
└── README.md
```



或作为包入口：

```bash
pip install -e .
traj2-run


## 说明

-“轨迹扩散（1D Conv1d U-Net + ε-pred）”训练与DDIM 采样；输出12 类（或 3 类）轨迹样张；同时提供64×64×1 的栅格化展示与真实对比（2×3 面板）

-Diffusion 采样：学得 ε_θ 与（可选的）s_θ 后，按反向 SDE 或 概率流 ODE 更新；通用形式是
dX_t = [ u_θ(X_t) + (σ_t^2/2) s_θ(X_t) ] dt + σ_t dW_t（σ_t=0 退化成 ODE）

-2_3
训练阶段：

目标从直接预测 z 改为预测速度场 v = (z - x_t) / (1 - t)
损失函数使用 F.mse_loss(pred_v, target_v)
时间步避免正好为 0（加了小偏移）


生成阶段：

使用 Euler 方法积分速度场进行 ODE 求解
时间从 0 → 1（而不是 1 → 0）
每步更新：x = x + v * dt




