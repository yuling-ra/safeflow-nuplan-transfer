# SafeFlow nuPlan 闭环仿真

本目录中的集成将 5-channel SafeFlow checkpoint 作为 nuPlan `AbstractPlanner` 运行。代码、Hydra 配置和启动入口都位于 `safeflow-nuplan-transfer` 内，不再依赖 Flow-Planner。

## 文件位置

- planner adapter: `dmpc_fm_cbf/safeflow_nuplan_planner.py`
- Hydra planner 配置: `config/planner/safeflow_planner.yaml`
- 仿真入口: `scripts/launch_sim_safeflow.sh`
- NuBoard 入口: `scripts/launch_nuboard_safeflow.sh`
- 轻量日志入口: `scripts/run_nuplan_simulation.py`

轻量日志入口在序列化场景前临时移除 learned planner 和 map API，避免每个场景日志重复保存模型，也避免在 CPU-only NuBoard 进程中反序列化 CUDA 模型。

## 默认目录布局

启动脚本按当前仓库位置自动推导以下默认路径：

```text
<workspace>/
├── nuplan-devkit/
│   ├── nuplan-v1.1_mini/
│   └── nuplan-maps-v1.0/maps/
└── safeflow-nuplan-transfer/
    └── dmpc_fm_cbf/
        └── notebooks/cache/model_vel_5ch_canonical_r.pt
```

当前工作区符合这个布局。其他机器可以通过环境变量覆盖所有外部路径。

## 运行仿真

```bash
cd /new_world/cockatiel/TeleNas/DataExchange/yuling/safeflow-nuplan-transfer/dmpc_fm_cbf
./scripts/launch_sim_safeflow.sh
```

默认设置：

- challenge: `closed_loop_nonreactive_agents`
- scenario types: `starting_left_turn` 和 `starting_straight_traffic_light_intersection_traversal`
- 每类 1 个场景，总计最多 2 个
- planner device: 自动选择 CUDA，否则使用 CPU
- checkpoint: `notebooks/cache/model_vel_5ch_canonical_r.pt`
- output root: `/tmp/safeflow_exp`

先打印最终命令而不运行：

```bash
DRY_RUN=1 ./scripts/launch_sim_safeflow.sh
```

单场景 CPU smoke test：

```bash
PLANNER_DEVICE=cpu \
SCENARIO_TYPES=medium_magnitude_speed \
SCENARIO_LIMIT=1 \
./scripts/launch_sim_safeflow.sh
```

移动场景中首次出现 `SafeFlow inference active` 日志表示模型已经实际参与规划。默认的两个路口起步场景可能因红灯让 IDM reference 全程静止；此时 adapter 按设计保持停车并使用 reference fallback，不会为了触发模型而强行产生前进轨迹。

运行 pure FM 对照：

```bash
USE_CBF=false ./scripts/launch_sim_safeflow.sh
```

## 常用环境变量

| 变量 | 默认值或行为 |
| --- | --- |
| `PYTHON_BIN` | 优先使用本机 `nuplan_clean`，否则寻找 `python3`/`python` |
| `NUPLAN_DEVKIT_ROOT` | sibling `<workspace>/nuplan-devkit` |
| `NUPLAN_DATA_ROOT` | `$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini` |
| `NUPLAN_MAPS_ROOT` | `$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps` |
| `NUPLAN_EXP_ROOT` | `/tmp/safeflow_exp` |
| `CKPT_FILE` | canonical 5-channel checkpoint |
| `PLANNER_DEVICE` | `auto`，也可设为 `cpu` 或 `cuda` |
| `SCENARIO_TYPES` | Hydra list 内使用的逗号分隔场景类型 |
| `SCENARIOS_PER_TYPE` | 每类场景数量 |
| `SCENARIO_LIMIT` | 场景总数上限 |
| `CHALLENGE` | nuPlan simulation 配置名 |
| `FM_NUM_SEGMENTS` | Flow Matching ODE 分段数 |
| `ODE_METHOD` | ODE solver 方法，默认 `rk4` |
| `MODEL_TRAJECTORY_STEPS` | 输出轨迹点数，默认 60 |
| `USE_CBF` | `true` 或 `false` |

额外的 Hydra override 可以直接作为脚本参数追加，例如：

```bash
./scripts/launch_sim_safeflow.sh \
  planner.safeflow_planner.cbf_alpha=10.0 \
  planner.safeflow_planner.seed=11
```

## 打开 NuBoard

自动打开 `$NUPLAN_EXP_ROOT` 下最新的 `.nuboard` 文件：

```bash
./scripts/launch_nuboard_safeflow.sh
```

也可以显式指定文件：

```bash
./scripts/launch_nuboard_safeflow.sh /path/to/result.nuboard
```

后续 NuBoard Hydra 参数可以继续追加到命令末尾。

## Planner 行为

每个 simulation step 中，adapter 先由官方 IDM planner 生成 route-aware reference，再选择近距离局部目标。SafeFlow 在 canonical frame 中采样轨迹并执行 CBF 投影，最后按 reference progress 重新参数化为 nuPlan `InterpolatedTrajectory`。

如果单次 SafeFlow inference 出错，该 step 会记录异常并退回 IDM reference。日志首次出现 `SafeFlow inference active` 表示模型路径已经实际参与规划，而不只是成功加载。
