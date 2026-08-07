"""Visualizations for distributed inference scaling experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    DEFAULT_OUTPUT_DIR,
    TIMING_WARMUP_ITERATIONS,
    best_per_gpu,
    configure_matplotlib,
    format_image_size,
    load_results,
    style_axes,
    summarize_configs,
    write_figure,
)
from toolsbench.visualization.scaling import (
    plot_strong_scaling_efficiency,
    plot_weak_scaling_by_workload,
)


def create_scaling_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create scaling visualizations from a benchopt inference result parquet."""
    configure_matplotlib()
    df, results_path = load_results(results)
    summary = summarize_configs(df)

    output_path = Path(output_dir) / f"scaling_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(
        output_path,
        [
            "step_time_breakdown.png",
            "memory_usage.png",
            "runtime_by_size.png",
        ],
    )

    plot_strong_scaling_efficiency(
        summary,
        output_path,
        group_col="p_dataset_image_size",
        distribute_col="p_solver_distribute_physics",
        title="Strong Scaling Efficiency",
        ylabel="Parallel efficiency vs smallest feasible GPU count (%)",
        legend_loc="upper right",
        footnote=(
            "100% means ideal scaling from each curve's baseline. Timings skip "
            f"first {TIMING_WARMUP_ITERATIONS} iterations."
        ),
    )
    plot_weak_scaling_by_workload(
        summary,
        output_path,
        distribute_col="p_solver_distribute_physics",
        mpix_col="image_mpix",
    )
    _plot_timing_breakdown(summary, output_path)
    return output_path


def _remove_stale_outputs(output_path: Path, filenames: list[str]) -> None:
    """Remove plots that are no longer produced by the current CLI."""
    for filename in filenames:
        path = output_path / filename
        if path.exists():
            path.unlink()


def _plot_timing_breakdown(summary: pd.DataFrame, output_path: Path) -> str:
    sizes = sorted(summary["p_dataset_image_size"].unique())
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(5.4 * len(sizes), 5.7),
        sharey=False,
        squeeze=False,
    )
    band_specs = [
        ("physics", "#2563eb"),
        ("denoising", "#f97316"),
        ("residual overhead", "#16a34a"),
    ]

    for col, image_size in enumerate(sizes):
        ax = axes[0, col]
        rows = best_per_gpu(
            summary[
                (summary["p_dataset_image_size"] == image_size)
                & (summary["p_solver_distribute_physics"])
            ].copy()
        ).copy()
        x = rows["n_gpus"].to_numpy(dtype=float)
        gradient = rows["avg_gradient_time_sec"].fillna(0).to_numpy(dtype=float)
        denoising = rows["avg_denoise_time_sec"].fillna(0).to_numpy(dtype=float)
        total = rows["avg_total_time_sec_raw"].fillna(0).to_numpy(dtype=float)
        measured = gradient + denoising
        residual = np.clip(total - measured, a_min=0, a_max=None)
        stacked = np.vstack([gradient, denoising, residual])

        ax.stackplot(
            x,
            stacked,
            labels=[label for label, _ in band_specs],
            colors=[color for _, color in band_specs],
            alpha=0.34,
            linewidth=0,
        )
        ax.plot(
            x,
            gradient,
            marker="o",
            markersize=4.8,
            linewidth=2.0,
            color="#2563eb",
            label="physics boundary",
        )
        ax.plot(
            x,
            measured,
            marker="o",
            markersize=4.8,
            linewidth=2.4,
            color="#f97316",
            label="physics + denoising",
        )
        ax.plot(
            x,
            total,
            marker="o",
            markersize=5.5,
            linewidth=3.0,
            color="#16a34a",
            label="total iteration",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(gpu)) for gpu in x])
        style_axes(
            ax,
            format_image_size(image_size),
            "Number of GPUs",
            "Average time / iteration (s)" if col == 0 else "",
        )
        ax.text(
            0.96,
            0.92,
            (
                f"iters {int(rows['timing_start_iter'].min())}-"
                f"{int(rows['timing_end_iter'].max())}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#6b7280",
            fontsize=10,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=False)
        if label in {"physics", "denoising", "residual overhead", "total iteration"}
    ]
    handles, labels = zip(*legend_items)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Inference Time Decomposition by Problem Size",
        x=0.02,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.02,
        "Green band is total iteration time not explained by physics + denoising.",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return write_figure(fig, output_path, "timing_breakdown_by_size.png")
