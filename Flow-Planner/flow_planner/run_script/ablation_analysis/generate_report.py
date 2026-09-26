#!/usr/bin/env python3
"""Generate figures and a CSV from the saved local ablation runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent

RUNS = {
    ("SafeFlow", "Straight"): Path("/tmp/safeflow_ablation_exp_straight_v2"),
    ("SafeFlow", "Curve"): Path("/tmp/safeflow_ablation_exp_curve"),
    ("SafeFlow", "Multi-vehicle"): Path("/tmp/safeflow_exp"),
    ("SafeFlow", "Pedestrian"): Path("/tmp/safeflow_exp"),
    ("ReFlow", "Straight"): Path("/tmp/safeflow_reflow_ablation/straight"),
    ("ReFlow", "Curve"): Path("/tmp/safeflow_reflow_ablation/curve"),
    ("ReFlow", "Multi-vehicle"): Path("/tmp/safeflow_reflow_ablation/multi_vehicle"),
    ("ReFlow", "Pedestrian"): Path("/tmp/safeflow_reflow_ablation/pedestrian"),
    ("Flow-Planner", "Straight"): Path("/tmp/flow_planner_ablation_exp_straight_scale80"),
    ("Flow-Planner", "Curve"): Path("/tmp/flow_planner_ablation_exp_curve_history_fix"),
    ("Flow-Planner", "Multi-vehicle"): Path("/tmp/flow_planner_ablation_exp_multi_history_fix2"),
    ("Flow-Planner", "Pedestrian"): Path("/tmp/flow_planner_ablation_exp_pedestrian_history_fix2"),
    ("PDM", "Straight"): Path("/tmp/pdm_ablation_exp_tuned_default"),
    ("PDM", "Curve"): Path("/tmp/pdm_ablation_exp_curve_default"),
    ("PDM", "Multi-vehicle"): Path("/tmp/pdm_ablation_exp_multi_dense_collision_free"),
    ("PDM", "Pedestrian"): Path("/tmp/pdm_ablation_exp_pedestrian_clean"),
    ("MPC-CBF", "Straight"): Path("/tmp/mpc_cbf_straight_final_rate"),
    ("MPC-CBF", "Curve"): Path("/tmp/mpc_cbf_curve_final"),
    ("MPC-CBF", "Multi-vehicle"): Path("/tmp/mpc_cbf_multi_smoke"),
    ("MPC-CBF", "Pedestrian"): Path("/tmp/mpc_cbf_pedestrian_smoke"),
}

SCENARIOS = ["Straight", "Curve", "Multi-vehicle", "Pedestrian"]
METHODS = ["SafeFlow", "ReFlow", "Flow-Planner", "PDM", "MPC-CBF"]
COLORS = {
    "SafeFlow": "#167c80",
    "ReFlow": "#7c3aed",
    "Flow-Planner": "#d97706",
    "PDM": "#315c9b",
    "MPC-CBF": "#b4432f",
}

# SafeFlow's older multi-vehicle and pedestrian roots contain several trials.
# These are the runs used in the saved NuBoard files referenced in the report.
SAFEFLOW_SELECTED = {
    "Multi-vehicle": Path(
        "/tmp/safeflow_exp/exp/simulation/closed_loop_nonreactive_agents/safeflow/"
        "all_scenarios/fm_cbf/model_vel_5ch_canonical_r_2026-08-20-10-47-28"
    ),
    "Pedestrian": Path(
        "/tmp/safeflow_exp/exp/simulation/closed_loop_nonreactive_agents/safeflow/"
        "all_scenarios/fm_cbf/model_vel_5ch_canonical_r_2026-08-20-21-56-55"
    ),
}


def metric_row(root: Path, name: str) -> pd.Series:
    paths = list(root.glob(f"**/metrics/{name}.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing {name} under {root}")
    return pd.read_parquet(paths[0]).iloc[0]


def read_run(planner: str, scenario: str, root: Path) -> dict:
    if planner == "SafeFlow" and scenario in SAFEFLOW_SELECTED:
        root = SAFEFLOW_SELECTED[scenario]

    progress = metric_row(root, "ego_progress_along_expert_route")
    collision = metric_row(root, "no_ego_at_fault_collisions")
    aggregate_paths = list(root.glob("**/aggregator_metric/*.parquet"))
    runner_paths = list(root.glob("**/runner_report.parquet"))
    aggregate = pd.read_parquet(aggregate_paths[0])
    aggregate = aggregate[aggregate["scenario"] == "final_score"].iloc[0]
    runner = pd.read_parquet(runner_paths[0]).iloc[0]

    timestamps = progress["time_series_timestamps"]
    return {
        "planner": planner,
        "scenario": scenario,
        "route_ratio": float(progress["ego_expert_progress_along_route_ratio_stat_value"]),
        "ego_progress_m": float(progress["ego_total_progress_along_route_stat_value"]),
        "collision_free": bool(collision["no_ego_at_fault_collisions_stat_value"]),
        "vehicle_collisions": int(collision["number_of_at_fault_collisions_with_vehicles_stat_value"]),
        "vru_collisions": int(collision["number_of_at_fault_collisions_with_VRUs_stat_value"]),
        "drivable": float(aggregate["drivable_area_compliance"]),
        "direction": float(aggregate["driving_direction_compliance"]),
        "comfortable": bool(aggregate["ego_is_comfortable"]),
        "making_progress": bool(aggregate["ego_is_making_progress"]),
        "speed_compliance": float(aggregate["speed_limit_compliance"]),
        "ttc_bound": float(aggregate["time_to_collision_within_bound"]),
        "score": float(aggregate["score"]),
        "simulation_s": float((timestamps[-1] - timestamps[0]) / 1e6),
        "inference_mean_ms": float(runner["compute_trajectory_runtimes_mean"] * 1000),
        "inference_median_ms": float(runner["compute_trajectory_runtimes_median"] * 1000),
        "wall_runtime_s": float(runner["duration"]),
    }


def grouped_bar(df: pd.DataFrame, value: str, ylabel: str, filename: str, ylim=None) -> None:
    x = np.arange(len(SCENARIOS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    for index, planner in enumerate(METHODS):
        values = [
            float(df[(df.planner == planner) & (df.scenario == scenario)][value].iloc[0])
            for scenario in SCENARIOS
        ]
        offset = (index - (len(METHODS) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=planner, color=COLORS[planner])
        for bar, number in zip(bars, values):
            label = f"{number:.2f}" if value != "inference_mean_ms" else f"{number:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x, SCENARIOS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=len(METHODS), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def binary_heatmap(df: pd.DataFrame, filename: str) -> None:
    values = np.array([
        [1 if bool(df[(df.planner == planner) & (df.scenario == scenario)].collision_free.iloc[0]) else 0
         for scenario in SCENARIOS]
        for planner in METHODS
    ])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS)
    ax.set_yticks(range(len(METHODS)), METHODS)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, "PASS" if values[row, col] else "FAIL", ha="center", va="center", fontsize=10)
    ax.set_title("No ego at-fault collision (one scenario per cell)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="1 = collision-free")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compliance_heatmap(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    values = np.array([
        [float(df[(df.planner == planner) & (df.scenario == scenario)][value].iloc[0])
         for scenario in SCENARIOS]
        for planner in METHODS
    ])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    image = ax.imshow(values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS)
    ax.set_yticks(range(len(METHODS)), METHODS)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="1 = fully compliant")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = []
    for (planner, scenario), root in RUNS.items():
        rows.append(read_run(planner, scenario, root))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False, float_format="%.6f")
    summary = (
        df.groupby("planner", sort=False)
        .agg(
            mean_route_ratio=("route_ratio", "mean"),
            mean_progress_m=("ego_progress_m", "mean"),
            collision_free_cells=("collision_free", "sum"),
            collision_free_rate=("collision_free", "mean"),
            mean_drivable_compliance=("drivable", "mean"),
            mean_comfort_compliance=("comfortable", "mean"),
            mean_score=("score", "mean"),
            mean_planning_latency_ms=("inference_mean_ms", "mean"),
            median_planning_latency_ms=("inference_median_ms", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    summary.to_csv(OUT / "summary_metrics.csv", index=False, float_format="%.6f")

    grouped_bar(df, "route_ratio", "Route progress ratio", "route_ratio.png", (0, 1.12))
    grouped_bar(df, "ego_progress_m", "Ego progress (m)", "ego_progress.png")
    grouped_bar(df, "inference_mean_ms", "Mean planning latency (ms)", "planning_latency.png")
    grouped_bar(df, "score", "Composite NuPlan score", "composite_score.png", (0, 1.12))
    grouped_bar(df, "drivable", "Drivable-area compliance", "drivable_area.png", (0, 1.12))
    binary_heatmap(df, "collision_free.png")
    compliance_heatmap(df, "comfortable", "Comfort compliance (1 = comfortable)", "comfort_compliance.png")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
