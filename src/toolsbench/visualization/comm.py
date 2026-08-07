"""Shared communication-time visualizations for training and inference PnP runs.

Both experiments profile per-section CUDA/communication time on a distributed
run (gradient/denoise for inference, forward/backward for training). Per the
POP methodology (see ``docs/comm-metrics-plan.md``), a section's raw
collective time is ``transfer + wait`` fused together: the collective starts
as soon as this rank arrives but cannot finish until every rank has joined, so
an early rank spins inside it. ``TorchProfiler._sync_comm_metrics`` already
separates the three parts at the source:

- ``{sec}_compute_sec``     mean over ranks, additive          (parallel term)
- ``{sec}_compute_max_sec`` max over ranks                     (critical path)
- ``{sec}_transfer_sec``    min over ranks, per collective     (pure transfer)
- ``{sec}_wait_sec``        mean(raw) - transfer               (average idle)
- ``{sec}_cuda_sec``        mean over ranks                    (== compute + transfer + wait)

Two result sets (2D and 3D) are combined into one figure per type, overlaid
on shared axes (2D solid lines / un-hatched bars, 3D dashed lines / hatched
bars) rather than one figure per dimensionality:

``comm_share.png``
    Stacked compute/transfer/wait cost per section, one paired bar (2D | 3D)
    per GPU count, in GPU-seconds relative to each series' own baseline.
``time_breakdown.png``
    Per-section compute_max / transfer / wait vs GPU count.
``wait_detail.png``
    Per-section load balance (LB), wait time, and wait-as-%-of-compute vs
    GPU count -- answers "are my GPUs unequal, and what does that cost".
``transfer_bandwidth.png``
    Transfer time vs GPU count, then the implied bandwidth of the same
    collectives -- the correctness check that ``transfer_sec`` measures real
    data movement and not leftover skew.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    TIMING_WARMUP_ITERATIONS,
    clear_png_outputs,
    configure_matplotlib,
    distributed_flag,
    load_results,
    make_output_path,
    mark_node_boundary,
    payload_bytes,
    set_gpu_axis,
    style_axes,
    write_figure,
)

Section = namedtuple(
    "Section", "name cuda_col compute_col compute_max_col transfer_col wait_col"
)

_COMPUTE_COLOR = "#2563eb"
_TRANSFER_COLOR = "#f97316"
_WAIT_COLOR = "#dc2626"

# (dark, light) per section -- compute / comm(=transfer+wait) shades of one
# hue, so the two phases stay legible on one bar without alpha blending.
_SECTION_SHADES = [
    ("#1d4ed8", "#93c5fd"),  # blue
    ("#c2410c", "#fdba74"),  # orange
    ("#15803d", "#86efac"),  # green
    ("#b91c1c", "#fca5a5"),  # red
]

_DIM_STYLE = {"2D": {"linestyle": "-"}, "3D": {"linestyle": "--"}}


def create_comm_visualizations(
    results_2d: str | Path,
    results_3d: str | Path,
    sections: list[Section],
    output_dir: str | Path,
    experiment_name: str,
    *,
    skip_warmup: int = TIMING_WARMUP_ITERATIONS,
) -> Path:
    """Build the 4 communication figures, merging a 2D and a 3D run."""
    configure_matplotlib()
    required = {
        col
        for s in sections
        for col in (s.cuda_col, s.compute_col, s.compute_max_col, s.transfer_col)
    }

    summaries = []
    stems = []
    for dim, results in (("2D", results_2d), ("3D", results_3d)):
        df, results_path = load_results(results, required=None)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(
                f"{results_path} is missing {', '.join(missing)}. This experiment "
                "needs a distributed run recorded with profiler_mode=torch."
            )
        summary = prepare_comm_summary(df, sections, skip_warmup)
        summary["dim"] = dim
        summaries.append(summary)
        stems.append(results_path.stem)

    summary = pd.concat(summaries, ignore_index=True)
    output_path = make_output_path(output_dir, experiment_name, Path(max(stems)))
    clear_png_outputs(output_path)

    _plot_comm_share(summary, sections, output_path)
    _plot_time_breakdown(summary, sections, output_path)
    _plot_wait_detail(summary, sections, output_path)
    _plot_transfer_bandwidth(summary, sections, output_path)
    return output_path


def prepare_comm_summary(
    df: pd.DataFrame, sections: list[Section], skip_warmup: int
) -> pd.DataFrame:
    """Average per-section compute/compute_max/transfer/wait over steady state.

    Groups by (GPU count, distributed, image size) so a size sweep and a
    tiled-vs-non-tiled 1-GPU baseline are never averaged together, then drops
    each group's first ``skip_warmup`` timed iterations (NCCL bootstrap +
    autotune inflate the first one or two).

    The skip is capped per group at ``count - 1`` so a short run (e.g. a
    comm-time sweep with only 4 profiled iterations) still keeps its last
    iteration instead of raising, at the cost of a warmup-contaminated mean
    for that group only.
    """
    timing_cols = [
        c
        for s in sections
        for c in (s.cuda_col, s.compute_col, s.compute_max_col, s.transfer_col)
    ]
    wait_cols = [s.wait_col for s in sections if s.wait_col in df.columns]
    df = df[df[timing_cols].notna().all(axis=1)].copy()
    if df.empty:
        raise ValueError("No rows carry per-section timings.")
    df["distributed"] = distributed_flag(df)

    keys = ["n_gpus", "distributed", "image_mpix"]
    group = df.groupby(keys, dropna=False)["stop_val"]
    ranked = group.rank(method="first")
    effective_skip = np.minimum(skip_warmup, group.transform("count") - 1)
    df = df[ranked > effective_skip]
    if df.empty:
        raise ValueError("No rows left after skipping warmup.")

    aggregations = {col: (col, "mean") for col in timing_cols + wait_cols}
    aggregations["n_nodes"] = ("n_nodes", "first")
    if "p_solver_n_iter" in df.columns:
        aggregations["p_solver_n_iter"] = ("p_solver_n_iter", "first")
    if "p_dataset_channels" in df.columns:
        aggregations["p_dataset_channels"] = ("p_dataset_channels", "first")

    summary = df.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    for s in sections:
        # NaN here means "no collective ran" (e.g. a non-distributed 1-GPU
        # baseline never enters the distributed op), which is 0 wait, not
        # missing data.
        if s.wait_col not in summary.columns:
            summary[s.wait_col] = 0.0
        else:
            summary[s.wait_col] = summary[s.wait_col].fillna(0.0)
        summary[s.transfer_col] = summary[s.transfer_col].fillna(0.0)
        summary[f"{s.name}_lb"] = np.where(
            summary[s.compute_max_col] > 0,
            summary[s.compute_col] / summary[s.compute_max_col],
            np.nan,
        )
        summary[f"{s.name}_wait_pct"] = np.where(
            summary[s.compute_col] > 0,
            100 * summary[s.wait_col] / summary[s.compute_col],
            np.nan,
        )

    # Sum of the sections' own cuda_sec, not benchopt's wall-clock
    # objective_total_time_sec -- keeps this self-consistent with the
    # compute/transfer/wait components above (each cuda_sec already equals
    # compute + transfer + wait, so no separate wait term needs adding back).
    summary["total_sec"] = summary[[s.cuda_col for s in sections]].sum(axis=1)
    summary["payload_bytes"] = payload_bytes(summary)
    return summary.sort_values(keys).reset_index(drop=True)


def collective_count(row: pd.Series, section_name: str) -> float | None:
    """Number of collectives summed into one section's transfer_sec, or None if
    that count can't be turned into a bandwidth number.

    Inference fires exactly one all_reduce per section (count 1) -- ``payload
    x count`` is exact. Training's ``forward`` fires 2 image-sized collectives
    per solver iteration (distribute + gather), also homogeneous, so ``2 *
    n_iter`` is exact too. Training's ``backward`` additionally fires a fixed
    64 *small* parameter all_reduces from ``DistributedParameterSync``
    (measured, not estimated -- see ``docs/comm-metrics-plan.md`` Phase 1b),
    mixed in with the ~``2 * n_iter`` image-sized ones -- a flat multiplier
    against ``payload_bytes()`` (one image's worth of bytes) would overstate
    backward's transferred bytes by a huge margin, so this returns None there
    and the bandwidth figure skips backward for training (transfer *time* is
    still exact and shown).
    """
    n_iter = row.get("p_solver_n_iter")
    if n_iter is None or pd.isna(n_iter):
        return 1.0
    if section_name == "forward":
        return 2.0 * n_iter
    return None


def _dim_groups(summary: pd.DataFrame) -> list[str]:
    return [d for d in ("2D", "3D") if d in summary["dim"].unique()]


def _plot_comm_share(
    summary: pd.DataFrame, sections: list[Section], output_path: Path
) -> str:
    """Stacked compute vs comm(=transfer+wait) cost, one paired bar (2D | 3D)
    per GPU count.

    Each dim is scaled to *its own* baseline (its smallest GPU count), so a
    bar's height is ``n_gpus * total_sec / (baseline_gpus * baseline_total_sec)``
    -- the baseline bar for each dim sits at 1.0, and a bar above 1.0 costs
    more GPU-seconds than ideal scaling would. 2D bars are solid, 3D bars are
    hatched, so both dims share one x-axis of GPU counts even though their
    absolute costs differ. Each segment is labelled with its share of that
    bar's own total (the segment percentages sum to 100% per bar).
    """
    dims = _dim_groups(summary)
    gpu_counts = sorted(summary["n_gpus"].unique())
    x = np.arange(len(gpu_counts))
    n_dims = len(dims)
    width = 0.72 / n_dims
    offsets = (np.arange(n_dims) - (n_dims - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(9.0, 1.8 * len(gpu_counts)), 6.6))

    for dim, offset in zip(dims, offsets, strict=True):
        rows = summary[summary["dim"] == dim].sort_values("n_gpus")
        if rows.empty:
            continue
        baseline = rows.iloc[0]
        ideal = float(baseline["n_gpus"]) * float(baseline["total_sec"])
        hatch = None if dim == "2D" else "///"

        for gi, gpus in enumerate(gpu_counts):
            row = rows[rows["n_gpus"] == gpus]
            if row.empty:
                continue
            row = row.iloc[0]
            scale = float(row["n_gpus"]) / ideal
            xi = x[gi] + offset
            bottom = 0.0
            bar_total = 0.0
            segments = []  # (value, color) in stacking order, for the label pass
            for i, s in enumerate(sections):
                dark, light = _SECTION_SHADES[i % len(_SECTION_SHADES)]
                segments.append((float(row[s.compute_col]) * scale, dark))
                segments.append(
                    (float(row[s.transfer_col] + row[s.wait_col]) * scale, light)
                )
            for value, color in segments:
                ax.bar(
                    xi,
                    value,
                    bottom=bottom,
                    width=width * 0.92,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    hatch=hatch,
                )
                bottom += value
                bar_total += value

            bottom = 0.0
            for value, _ in segments:
                pct = 100 * value / bar_total if bar_total > 0 else 0.0
                if pct >= 3:
                    ax.annotate(
                        f"{pct:.0f}%",
                        (xi, bottom + value / 2),
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="white",
                    )
                bottom += value

            ax.annotate(
                f"{bar_total:.2f}x",
                (xi, bar_total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#111827",
            )

    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(g)) for g in gpu_counts])
    style_axes(
        ax,
        "Compute vs comm (transfer+wait) cost per iteration (1.0 = ideal scaling)",
        "Number of GPUs",
        "GPU-seconds relative to each series' own baseline",
    )

    legend_handles = []
    for i, s in enumerate(sections):
        dark, light = _SECTION_SHADES[i % len(_SECTION_SHADES)]
        legend_handles += [
            mpatches.Patch(color=dark, label=f"{s.name} compute"),
            mpatches.Patch(color=light, label=f"{s.name} comm (transfer+wait)"),
        ]
    if "3D" in dims and "2D" in dims:
        legend_handles += [
            mpatches.Patch(facecolor="white", edgecolor="#111827", label="2D"),
            mpatches.Patch(
                facecolor="white", edgecolor="#111827", hatch="///", label="3D"
            ),
        ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncols=min(len(legend_handles), 4),
        fontsize=7.5,
        frameon=False,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "comm_share.png")


def _plot_time_breakdown(
    summary: pd.DataFrame, sections: list[Section], output_path: Path
) -> str:
    """Compute_max (critical path) / transfer / wait per section vs GPU count."""
    dims = _dim_groups(summary)
    fig, axes = plt.subplots(
        1, len(sections), figsize=(6.5 * len(sections), 5.6), squeeze=False
    )
    for i, s in enumerate(sections):
        ax = axes[0, i]
        for dim in dims:
            rows = summary[summary["dim"] == dim].sort_values("n_gpus")
            if rows.empty:
                continue
            gpus = rows["n_gpus"].to_numpy(dtype=float)
            style = _DIM_STYLE[dim]
            ax.plot(
                gpus,
                rows[s.compute_max_col],
                marker="o",
                color=_COMPUTE_COLOR,
                linewidth=2.2,
                label=f"compute_max ({dim})",
                **style,
            )
            ax.plot(
                gpus,
                rows[s.transfer_col],
                marker="o",
                color=_TRANSFER_COLOR,
                linewidth=2.2,
                label=f"transfer ({dim})",
                **style,
            )
            ax.plot(
                gpus,
                rows[s.wait_col],
                marker="o",
                color=_WAIT_COLOR,
                linewidth=1.8,
                label=f"wait ({dim})",
                **style,
            )
        ax.set_yscale("log")
        set_gpu_axis(ax, summary["n_gpus"])
        style_axes(
            ax,
            f"{s.name} time vs GPU count",
            "Number of GPUs",
            "Time per iteration (s, log scale)" if i == 0 else "",
        )
        mark_node_boundary(ax, summary)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return write_figure(fig, output_path, "time_breakdown.png")


def _plot_wait_detail(
    summary: pd.DataFrame, sections: list[Section], output_path: Path
) -> str:
    """LB and wait-as-%-of-compute per section vs GPU count."""
    dims = _dim_groups(summary)
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, len(sections), figsize=(6.0 * len(sections), 9.4), squeeze=False
    )
    row_ylabels = ["LB", "Wait / compute (%)"]
    for col, s in enumerate(sections):
        for row in range(n_rows):
            ylabel = row_ylabels[row]
            ax = axes[row, col]
            for dim in dims:
                rows = summary[summary["dim"] == dim].sort_values("n_gpus")
                if rows.empty:
                    continue
                gpus = rows["n_gpus"].to_numpy(dtype=float)
                style = _DIM_STYLE[dim]
                y = rows[f"{s.name}_lb"] if row == 0 else rows[f"{s.name}_wait_pct"]
                ax.plot(
                    gpus,
                    y,
                    marker="o",
                    linewidth=2.2,
                    color=_COMPUTE_COLOR if row == 0 else _WAIT_COLOR,
                    label=dim,
                    **style,
                )
            set_gpu_axis(ax, summary["n_gpus"])
            style_axes(
                ax,
                s.name if row == 0 else "",
                "Number of GPUs" if row == n_rows - 1 else "",
                ylabel if col == 0 else "",
            )
            mark_node_boundary(ax, summary)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Load balance (compute / compute_max) and wait as % of compute, "
        "per section vs GPU count",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return write_figure(fig, output_path, "wait_detail.png")


def _plot_transfer_bandwidth(
    summary: pd.DataFrame, sections: list[Section], output_path: Path
) -> str:
    """Transfer time vs GPU count, then the implied bandwidth of the same data."""
    dims = _dim_groups(summary)
    rows_all = summary[summary["n_gpus"] > 1]
    if rows_all.empty:
        return None
    fig, (ax_time, ax_bw) = plt.subplots(1, 2, figsize=(15.5, 6.2))

    for i, s in enumerate(sections):
        color = _SECTION_SHADES[i % len(_SECTION_SHADES)][0]
        for dim in dims:
            rows = rows_all[rows_all["dim"] == dim].sort_values("n_gpus")
            if rows.empty:
                continue
            style = _DIM_STYLE[dim]
            gpus = rows["n_gpus"].to_numpy(dtype=float)
            transfer = rows[s.transfer_col].to_numpy(dtype=float)
            ax_time.plot(
                gpus,
                transfer,
                marker="o",
                color=color,
                linewidth=2.2,
                label=f"{s.name} ({dim})",
                **style,
            )

            counts_raw = [collective_count(r, s.name) for _, r in rows.iterrows()]
            if any(c is None for c in counts_raw):
                # e.g. training's backward section: see collective_count's
                # docstring for why a flat multiplier would be meaningless here.
                continue
            counts = np.array(counts_raw, dtype=float)
            payload = rows["payload_bytes"].to_numpy(dtype=float)
            transfer_bytes = counts * 2 * payload * (gpus - 1) / gpus
            bandwidth = (
                np.where(
                    transfer > 0,
                    transfer_bytes / np.where(transfer > 0, transfer, 1.0),
                    np.nan,
                )
                / 1e9
            )
            ax_bw.plot(
                gpus,
                bandwidth,
                marker="o",
                color=color,
                linewidth=2.2,
                label=f"{s.name} ({dim})",
                **style,
            )

    ax_time.set_yscale("log")
    set_gpu_axis(ax_time, rows_all["n_gpus"])
    style_axes(
        ax_time,
        "Transfer time (pure data movement)",
        "Number of GPUs",
        "Transfer time per iteration (s, log scale)",
    )
    mark_node_boundary(ax_time, rows_all)
    ax_time.legend(loc="best", fontsize=8)

    ax_bw.set_yscale("log")
    set_gpu_axis(ax_bw, rows_all["n_gpus"])
    style_axes(
        ax_bw,
        "Implied bandwidth (transfer check)",
        "Number of GPUs",
        "Implied bandwidth (GB/s, log scale)",
    )
    mark_node_boundary(ax_bw, rows_all)
    ax_bw.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return write_figure(fig, output_path, "transfer_bandwidth.png")
