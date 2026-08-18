"""Shared strong/weak-scaling plotting helpers, used by both inference and training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from toolsbench.visualization.common import (
    COLORWAY,
    best_per_gpu,
    format_image_size,
    set_gpu_axis,
    style_axes,
    write_figure,
)


def resolve_scaling_baseline(
    group: pd.DataFrame,
    rows: pd.DataFrame,
    distribute_col: str,
) -> tuple[float, int, str]:
    """Pick the strong-scaling efficiency baseline for one group of runs.

    Prefers the 1-GPU non-distributed run if present (so a distributed 1-GPU
    point can show its own overhead below 100%); otherwise falls back to the
    smallest-GPU distributed row, using its own GPU count as the baseline.
    """
    baseline_candidates = group[(group["n_gpus"] == 1) & (~group[distribute_col])]
    if not baseline_candidates.empty:
        baseline_time = float(baseline_candidates.iloc[0]["avg_total_time_sec"])
        return baseline_time, 1, "1-GPU non-distributed"

    baseline = rows.loc[rows["n_gpus"].idxmin()]
    baseline_gpus = int(baseline["n_gpus"])
    baseline_time = float(baseline["avg_total_time_sec"])
    return baseline_time, baseline_gpus, f"{baseline_gpus}-GPU distributed"


def plot_weak_scaling_by_workload(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    distribute_col: str,
    mpix_col: str,
    title: str = "Weak Scaling Runtime Ratio",
    filename: str = "weak_scaling.png",
) -> str:
    """Group distributed runs by per-GPU workload and plot runtime ratio vs GPU count."""
    weak = summary[summary[distribute_col]].copy()
    weak["local_workload_mpix"] = weak[mpix_col] / weak["n_gpus"]
    weak["local_workload_label"] = weak["local_workload_mpix"].round(2)

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    workloads = (
        weak.groupby("local_workload_label")
        .filter(lambda group: len(group) >= 2)["local_workload_label"]
        .drop_duplicates()
        .sort_values(ascending=False)
    )
    for idx, workload in enumerate(workloads):
        rows = (
            weak[weak["local_workload_label"] == workload].sort_values("n_gpus").copy()
        )
        baseline_time = float(rows.iloc[0]["avg_total_time_sec"])
        rows["time_ratio"] = rows["avg_total_time_sec"] / baseline_time
        color = COLORWAY[idx % len(COLORWAY)]
        ax.plot(
            rows["n_gpus"],
            rows["time_ratio"],
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=color,
            label=f"{workload:g} Mpix/GPU",
        )

    set_gpu_axis(ax, weak["n_gpus"])
    ax.axhline(1, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    style_axes(
        ax,
        title,
        "Number of GPUs",
        "Average iteration time / smallest-GPU time",
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Local workload",
        ncols=1,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, filename)


def plot_strong_scaling_efficiency(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str,
    distribute_col: str,
    title: str = "Strong Scaling Efficiency",
    ylabel: str = "Parallel efficiency (%)",
    filename: str = "strong_scaling.png",
    group_label: str | None = None,
    legend_loc: str = "lower left",
    footnote: str | None = None,
) -> str:
    """Plot strong-scaling efficiency, one curve per `group_col` value.

    Used by both inference and training: each curve restricts itself to
    distributed runs (`distribute_col`) and is normalized against a baseline
    picked by `resolve_scaling_baseline` (1-GPU non-distributed if available).
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    max_efficiency = 100.0
    all_gpus = sorted(summary["n_gpus"].unique())

    for idx, (group_value, group) in enumerate(
        summary.groupby(group_col, dropna=False)
    ):
        rows = best_per_gpu(group[group[distribute_col]].copy())
        if rows.empty:
            continue
        baseline_time, baseline_gpus, baseline_note = resolve_scaling_baseline(
            group, rows, distribute_col
        )
        rows = rows.assign(
            efficiency_pct=(
                baseline_time
                * baseline_gpus
                / (rows["n_gpus"] * rows["avg_total_time_sec"])
                * 100
            )
        )
        max_efficiency = max(max_efficiency, float(rows["efficiency_pct"].max()))
        color = COLORWAY[idx % len(COLORWAY)]
        if group_label is None:
            group_name = format_image_size(group_value)
        else:
            group_name = f"{group_label} {group_value:g}"
        label = f"{group_name} ({baseline_note} baseline)"
        ax.plot(
            rows["n_gpus"],
            rows["efficiency_pct"],
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=color,
            label=label,
        )

    set_gpu_axis(ax, all_gpus)
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    ax.set_ylim(0, max(110, max_efficiency * 1.12))
    style_axes(ax, title, "Number of GPUs", ylabel)
    ax.legend(loc=legend_loc, ncols=1)
    if footnote:
        ax.text(
            0.99,
            0.03,
            footnote,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#6b7280",
            fontsize=10,
        )
    fig.tight_layout()
    return write_figure(fig, output_path, filename)
