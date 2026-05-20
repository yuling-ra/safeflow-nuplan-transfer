# nuPlan Official Baseline Setup

这个文档针对当前工作区：

- workspace root: `/new_world/cockatiel/ra`
- 你的项目包: `/new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf`
- nuPlan 数据:
  - db: `/new_world/cockatiel/ra/data`
  - maps: `/new_world/cockatiel/ra/maps`

目标分两部分：

1. 继续使用你现在已经写好的离线 transfer benchmark。
2. 安装官方 `nuplan-devkit`，并接上 `official_idm` / `official_pdm_closed` baseline。

## 1. 官方来源

下面的安装方式基于官方仓库说明：

- `nuplan-devkit` 官方安装文档：`https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md`
- `tuplan_garage` 官方 README：`https://github.com/autonomousvision/tuplan_garage`

官方说明里提到的关键点：

- `nuplan-devkit` 在 Ubuntu 上测试环境是 Python 3.9。
- 官方推荐用 conda，并通过 `pip install -e .` 安装本地 devkit。
- `tuplan_garage` 依赖已经安装好的 `nuplan-devkit`，并要求激活同一个 `nuplan` 环境后再 `pip install -e .`。

## 2. 建议环境

不要把 `nuplan-devkit` 直接装进你现在的 `openpi_liu` 环境。

原因：

- `nuplan-devkit` 对依赖版本比较敏感。
- 你当前项目里已经有自己的一套 `torch / pandas / jupyter` 依赖。
- 混装后很容易把你现有 notebook 环境弄脏。

建议新建一个独立环境，例如 `nuplan`.

## 3. 安装 nuplan-devkit

### 3.1 clone 官方仓库

建议放到：

- `/new_world/cockatiel/nuplan-devkit`

命令：

```bash
cd /new_world/cockatiel
git clone https://github.com/motional/nuplan-devkit.git
cd /new_world/cockatiel/nuplan-devkit
```

### 3.2 创建 conda 环境

官方文档建议：

```bash
conda env create -f environment.yml
conda activate nuplan
```

如果 `environment.yml` 创建失败，再退回手动方式：

```bash
conda create -n nuplan python=3.9 -y
conda activate nuplan
pip install -U pip setuptools wheel
```

### 3.3 安装 devkit

官方本地 editable 安装：

```bash
cd /new_world/cockatiel/nuplan-devkit
pip install -e .
```

如果官方 editable 安装没有自动拉齐依赖，再补：

```bash
pip install -r requirements_torch.txt
pip install -r requirements.txt
```

### 3.4 配置环境变量

```bash
export NUPLAN_DEVKIT_ROOT=/new_world/cockatiel/nuplan-devkit
export NUPLAN_DATA_ROOT=/new_world/cockatiel/ra/data
export NUPLAN_MAPS_ROOT=/new_world/cockatiel/ra/maps
export NUPLAN_EXP_ROOT=/new_world/cockatiel/ra/nuplan_exp
```

建议把这些也写进 `~/.bashrc` 或你常用的 shell rc 文件。

### 3.5 最小检查

```bash
python -c "import nuplan; print('nuplan import ok')"
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py --help
```

## 4. 安装 tuplan_garage

只有你要跑 `official_pdm_closed` 才需要这一步。

### 4.1 clone 仓库

建议放到：

- `/new_world/cockatiel/tuplan_garage`

```bash
cd /new_world/cockatiel
git clone https://github.com/autonomousvision/tuplan_garage.git
cd /new_world/cockatiel/tuplan_garage
```

### 4.2 在同一个 `nuplan` 环境里安装

```bash
conda activate nuplan
pip install -e .
```

### 4.3 配置环境变量

`tuplan_garage` README 里要求：

```bash
export NUPLAN_DEVKIT_ROOT=/new_world/cockatiel/nuplan-devkit
```

### 4.4 最小检查

```bash
python -c "import tuplan_garage; print('tuplan_garage import ok')"
```

## 5. 用当前项目生成官方 baseline 命令

你现在仓库里已经有这个脚本：

- [print_nuplan_sim_command.py](/new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf/scripts/print_nuplan_sim_command.py)

### 5.1 生成 official IDM 命令

```bash
conda activate nuplan
python /new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf/scripts/print_nuplan_sim_command.py \
  --root /new_world/cockatiel/ra \
  --devkit-root /new_world/cockatiel/nuplan-devkit \
  --planner official_idm
```

### 5.2 生成 official PDM-Closed 命令

```bash
conda activate nuplan
python /new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf/scripts/print_nuplan_sim_command.py \
  --root /new_world/cockatiel/ra \
  --devkit-root /new_world/cockatiel/nuplan-devkit \
  --planner official_pdm_closed
```

## 6. 你当前项目里的对应接口

### 6.1 离线 benchmark 主模块

- [nuplan_benchmark.py](/new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf/dmpc_fm_cbf/nuplan_benchmark.py)

它负责：

- 场景筛选
- sqlite 场景读取
- `FM+CBF / Pure FM / IDM-proxy`
- 指标聚合

### 6.2 官方 SDK 适配层

- [nuplan_sdk_adapter.py](/new_world/cockatiel/ra/safeflow-nuplan-transfer/dmpc_fm_cbf/dmpc_fm_cbf/nuplan_sdk_adapter.py)

它负责：

- `build_idm_planner()`
- `build_pdm_closed_planner()`
- `build_nuplan_scenario_builder()`
- `build_scenario_filter()`
- `build_run_simulation_command()`

## 7. 推荐执行顺序

### 阶段一：先保持你现有实验能跑

```python
from dmpc_fm_cbf.nuplan_benchmark import BenchmarkConfig, run_benchmark, aggregate_metrics

cfg = BenchmarkConfig(root="/new_world/cockatiel/ra")
metrics_df, traces_df = run_benchmark(cfg)
summary_df = aggregate_metrics(metrics_df)
```

这一步不依赖官方 SDK。

### 阶段二：装 SDK 跑官方 baseline

顺序：

1. 安装 `nuplan-devkit`
2. 安装 `tuplan_garage`（如果要 PDM-Closed）
3. 用 `print_nuplan_sim_command.py` 生成命令
4. 运行官方 `run_simulation.py`
5. 把官方输出结果回填到你自己的汇总表

## 8. 你现在最该做的事

先执行这组命令：

```bash
cd /new_world/cockatiel
git clone https://github.com/motional/nuplan-devkit.git
conda env create -f /new_world/cockatiel/nuplan-devkit/environment.yml
conda activate nuplan
pip install -e /new_world/cockatiel/nuplan-devkit
```

如果你也要 PDM-Closed，再执行：

```bash
git clone https://github.com/autonomousvision/tuplan_garage.git /new_world/cockatiel/tuplan_garage
pip install -e /new_world/cockatiel/tuplan_garage
```

## 9. 当前限制

我现在没有直接替你完成安装，因为这台会话环境当前：

- 没有 `nuplan` 包
- 没有 `tuplan_garage`
- 网络安装需要你本地实际执行

但安装后，你现在仓库里的接口和命令生成层已经准备好了。
