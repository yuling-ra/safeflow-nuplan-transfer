from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_METRIC_COLUMNS = [
    "method",
    "paper_bucket",
    "time_to_goal_s",
    "collision",
    "goal_reached",
    "min_inter_agent_dist_m",
    "road_boundary_violations",
    "final_dist_to_goal_m",
]


DEFAULT_METHOD_META = {
    "fm_cbf": {
        "display_name": "SafeFlow (FM+CBF)",
        "family": "ours",
        "training_regime": "zero_shot",
        "notes": "Zero-shot transfer",
    },
    "pure_fm": {
        "display_name": "Pure FM",
        "family": "ours_ablation",
        "training_regime": "zero_shot",
        "notes": "Zero-shot transfer",
    },
    "idm_proxy": {
        "display_name": "IDM-Proxy",
        "family": "classical",
        "training_regime": "no_learning",
        "notes": "Offline proxy baseline",
    },
    "official_idm": {
        "display_name": "IDM",
        "family": "classical",
        "training_regime": "no_learning",
        "notes": "Official nuPlan baseline",
    },
    "official_pdm_closed": {
        "display_name": "PDM-Closed",
        "family": "classical",
        "training_regime": "no_learning",
        "notes": "Official nuPlan / tuPlan baseline",
    },
    "diffusion_planner": {
        "display_name": "DiffusionPlanner",
        "family": "learned_baseline",
        "training_regime": "in_domain",
        "notes": "Typically trained on nuPlan",
    },
    "flow_planner": {
        "display_name": "FlowPlanner",
        "family": "learned_baseline",
        "training_regime": "in_domain",
        "notes": "Typically trained on nuPlan",
    },
    "fm_cbf_finetuned": {
        "display_name": "SafeFlow (Fine-tuned)",
        "family": "ours",
        "training_regime": "fine_tuned",
        "notes": "Fine-tuned on nuPlan mini",
    },
    "pure_fm_finetuned": {
        "display_name": "Pure FM (Fine-tuned)",
        "family": "ours_ablation",
        "training_regime": "fine_tuned",
        "notes": "Fine-tuned on nuPlan mini",
    },
}


@dataclass
class FigurePaths:
    zero_shot_barplot: Path
    regime_barplot: Path
    ablation_tau: Path
    ablation_k: Path
    ablation_margin: Path


def ensure_metric_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics dataframe missing required columns: {missing}")


def load_metrics_csv(path: Path, *, method_overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_metric_columns(df)
    if method_overrides:
        df["method"] = df["method"].replace(method_overrides)
    return df


def load_external_baseline_csv(path: Path) -> pd.DataFrame:
    """
    External CSV schema should contain the same metric columns as our benchmark output.
    Optional columns:
      - training_regime
      - display_name
      - notes
    """
    df = pd.read_csv(path)
    ensure_metric_columns(df)
    return df


def merge_metrics_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(tables, ignore_index=True)
    ensure_metric_columns(merged)
    return merged


def build_method_metadata(
    methods: Iterable[str],
    *,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    rows = []
    overrides = overrides or {}
    for method in sorted(set(methods)):
        meta = dict(DEFAULT_METHOD_META.get(method, {}))
        meta.update(overrides.get(method, {}))
        rows.append(
            {
                "method": method,
                "display_name": meta.get("display_name", method),
                "family": meta.get("family", "unknown"),
                "training_regime": meta.get("training_regime", "unknown"),
                "notes": meta.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def attach_method_metadata(metrics_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.merge(metadata_df, on="method", how="left")
    out["display_name"] = out["display_name"].fillna(out["method"])
    out["training_regime"] = out["training_regime"].fillna("unknown")
    out["family"] = out["family"].fillna("unknown")
    out["notes"] = out["notes"].fillna("")
    return out


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(
            ["paper_bucket", "method", "display_name", "training_regime", "family", "notes"],
            as_index=False,
        )
        .agg(
            time_to_goal_s=("time_to_goal_s", "mean"),
            collision_rate=("collision", "mean"),
            goal_reached_rate=("goal_reached", "mean"),
            min_inter_agent_dist_m=("min_inter_agent_dist_m", "mean"),
            road_boundary_violations=("road_boundary_violations", "mean"),
            final_dist_to_goal_m=("final_dist_to_goal_m", "mean"),
            episodes=("method", "count"),
        )
        .sort_values(["paper_bucket", "training_regime", "display_name"])
    )


def build_zero_shot_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    zero_shot_like = {"zero_shot", "no_learning", "in_domain"}
    out = summary_df[summary_df["training_regime"].isin(zero_shot_like)].copy()
    return out.sort_values(["paper_bucket", "training_regime", "display_name"])


def build_finetune_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df[summary_df["training_regime"].isin({"fine_tuned", "in_domain", "no_learning"})].copy()
    return out.sort_values(["paper_bucket", "training_regime", "display_name"])


def export_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False))


def export_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def _plot_metric_bars(
    summary_df: pd.DataFrame,
    methods_df: pd.DataFrame,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    buckets = list(summary_df["paper_bucket"].drop_duplicates())
    method_order = list(methods_df["display_name"])

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        pivot = (
            summary_df.pivot(index="paper_bucket", columns="display_name", values=metric_key)
            .reindex(index=buckets, columns=method_order)
        )
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric_title)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_zero_shot_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    zero_df = build_zero_shot_table(summary_df)
    methods_df = zero_df[["display_name"]].drop_duplicates()
    _plot_metric_bars(
        zero_df,
        methods_df,
        [
            ("collision_rate", "Collision Rate ↓"),
            ("goal_reached_rate", "Goal Reached Rate ↑"),
            ("road_boundary_violations", "Road Violations ↓"),
            ("min_inter_agent_dist_m", "Min Inter-Agent Dist ↑"),
        ],
        "Zero-shot / Baseline Comparison",
        output_path,
    )


def plot_regime_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    finetune_df = build_finetune_table(summary_df)
    methods_df = finetune_df[["display_name"]].drop_duplicates()
    _plot_metric_bars(
        finetune_df,
        methods_df,
        [
            ("collision_rate", "Collision Rate ↓"),
            ("goal_reached_rate", "Goal Reached Rate ↑"),
            ("time_to_goal_s", "Time to Goal ↓"),
            ("final_dist_to_goal_m", "Final Dist to Goal ↓"),
        ],
        "Regime Comparison: Zero-shot vs Fine-tuned",
        output_path,
    )


def plot_ablation(
    ablation_df: pd.DataFrame,
    *,
    x_col: str,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for bucket, bucket_df in ablation_df.groupby("paper_bucket"):
            ordered = bucket_df.sort_values(x_col)
            ax.plot(ordered[x_col], ordered[metric_key], marker="o", label=bucket)
        ax.set_title(metric_title)
        ax.set_xlabel(x_col)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figure_paths(output_dir: Path) -> FigurePaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return FigurePaths(
        zero_shot_barplot=output_dir / "zero_shot_comparison.png",
        regime_barplot=output_dir / "regime_comparison.png",
        ablation_tau=output_dir / "ablation_tau.png",
        ablation_k=output_dir / "ablation_k.png",
        ablation_margin=output_dir / "ablation_margin.png",
    )
