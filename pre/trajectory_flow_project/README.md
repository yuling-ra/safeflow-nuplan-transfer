# Trajectory Flow Project

A clean Python project that trains a 1D U-Net with Flow Matching to generate 2D trajectories (circle, line, ellipse).  
We split the original monolithic script into modules for data, schedules, conditional paths, model, training, sampling, and visualization.

## Quickstart

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install
pip install -r requirements.txt

# run
python -m scripts.train_and_sample
```

## Structure
```
trajectory_flow_project/
├─ src/
│  └─ diffusion/
│     ├─ __init__.py
│     ├─ data.py
│     ├─ schedule.py
│     ├─ cppath.py
│     ├─ viz.py
│     ├─ sampling.py
│     ├─ train.py
│     └─ model/
│        ├─ __init__.py
│        ├─ time_embed.py
│        └─ unet1d.py
├─ scripts/
│  └─ train_and_sample.py
├─ requirements.txt
└─ README.md
```
