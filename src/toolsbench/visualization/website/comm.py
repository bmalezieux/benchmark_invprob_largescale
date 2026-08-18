"""Plot-ready website data for the cross-experiment communication finding.

Reuses ``prepare_comm_summary``/``Section``/``collective_count`` from
:mod:`toolsbench.visualization.comm` -- the same POP (compute/transfer/wait)
aggregation the PNG comm figures use -- so the website numbers and the PNGs
can never silently diverge.

Inference and training are exported separately (``communication-inference``
and ``communication-training``), each merging its own 2D and 3D result set
via a ``dim`` field, so one study can be regenerated without needing fresh
results for the other -- they are reran on independent schedules in
practice. Both files share a ``study`` field so a chart that compares them
can simply concatenate the two ``values`` arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toolsbench.visualization.comm import (
    Section,
    collective_count,
    prepare_comm_summary,
)
from toolsbench.visualization.common import (
    TIMING_WARMUP_ITERATIONS,
    collective_bytes,
    load_results,
)
from toolsbench.visualization.inference.comm_inference import (
    SECTIONS as INFERENCE_SECTIONS,
)
from toolsbench.visualization.training.comm_time import SECTIONS as TRAINING_SECTIONS
from toolsbench.visualization.website.common import (
    SCHEMA_VERSION,
    number,
    provenance as _provenance,
    write_json as _write_json,
)

FINDING_ID = "communication"

_METHODOLOGY = {
    "reference": "POP methodology (https://pop-coe.eu/node/69)",
    "computeSec": (
        "mean over ranks of (section CUDA time - this rank's raw collective "
        "time), additive across ranks"
    ),
    "computeMaxSec": "max over ranks of the same quantity -- the critical path",
    "transferSec": (
        "sum over collectives in the section of the per-collective min over "
        "ranks -- pure data movement"
    ),
    "waitSec": "mean(raw collective time) - transferSec -- average idle at collectives",
    "scaledComputeSec": (
        "computeSec * gpuCount / (baselineGpuCount * baselineTotalSec), where "
        "baseline is the smallest GPU count for this study/dim and "
        "baselineTotalSec sums cudaSec over every section at that baseline -- "
        "GPU-seconds relative to ideal linear scaling (1.0 at baseline), the "
        "same scaling the comm_share.png figure uses"
    ),
    "scaledCommSec": "(transferSec + waitSec) scaled the same way as scaledComputeSec",
    "lbPct": "100 * computeSec / computeMaxSec",
    "waitPct": "100 * waitSec / computeSec",
    "lbCombinedPct": (
        "100 * sum(computeSec over every section) / sum(computeMaxSec over "
        "every section), at this gpuCount -- POP load balance for the whole "
        "study/dim, not just one section"
    ),
    "commEPct": (
        "100 * sum(computeMaxSec over every section) / totalSec, where "
        "totalSec sums cudaSec over every section -- POP communication "
        "efficiency; lbCombinedPct * commEPct / 100 is parallel efficiency"
    ),
    "commEPctSection": (
        "100 * computeMaxSec / cudaSec -- the same communication efficiency "
        "restricted to this section alone; summing numerator and denominator "
        "over every section instead gives commEPct"
    ),
    "bandwidthGBs": (
        "implied bandwidth from transfer time and payload size; null for a "
        "single-GPU row (nothing to transfer) or where the collective count "
        "for that section cannot be derived exactly (training's backward "
        "section mixes image and parameter collectives)"
    ),
    "independentRepetitions": 1,
}


def create_communication_inference_website_data(
    *,
    results_2d: str | Path,
    results_3d: str | Path,
    output_dir: str | Path,
    skip_warmup: int = TIMING_WARMUP_ITERATIONS,
) -> list[Path]:
    """Create the compute/transfer/wait communication data for inference."""
    return _create_communication_website_data(
        study="inference",
        sections=INFERENCE_SECTIONS,
        results_2d=results_2d,
        results_3d=results_3d,
        output_dir=output_dir,
        skip_warmup=skip_warmup,
    )


def create_communication_training_website_data(
    *,
    results_2d: str | Path,
    results_3d: str | Path,
    output_dir: str | Path,
    skip_warmup: int = TIMING_WARMUP_ITERATIONS,
) -> list[Path]:
    """Create the compute/transfer/wait communication data for training."""
    return _create_communication_website_data(
        study="training",
        sections=TRAINING_SECTIONS,
        results_2d=results_2d,
        results_3d=results_3d,
        output_dir=output_dir,
        skip_warmup=skip_warmup,
    )


def _create_communication_website_data(
    *,
    study: str,
    sections: list[Section],
    results_2d: str | Path,
    results_3d: str | Path,
    output_dir: str | Path,
    skip_warmup: int,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    sources = []
    for dim, results in (("2D", results_2d), ("3D", results_3d)):
        df, path = load_results(results, required=None)
        summary = prepare_comm_summary(df, sections, skip_warmup).sort_values("n_gpus")
        # Same baseline-relative scale as comm.py's _plot_comm_share: the
        # smallest GPU count's total (summed over every section) is the
        # ideal-scaling reference, so its own bar comes out to 1.0.
        baseline = summary.iloc[0]
        ideal = float(baseline["n_gpus"]) * float(baseline["total_sec"])
        summary = summary.copy()
        summary["comm_share_scale"] = summary["n_gpus"] / ideal
        # POP load balance / communication efficiency, summed across every
        # section in this study/dim (not per-section) -- see docs/comm-metrics-plan.md.
        compute_total = sum(summary[s.compute_col] for s in sections)
        compute_max_total = sum(summary[s.compute_max_col] for s in sections)
        summary["lb_combined_pct"] = 100 * compute_total / compute_max_total
        summary["comm_e_pct"] = 100 * compute_max_total / summary["total_sec"]
        rows.extend(_melt_sections(summary, sections, study, dim))
        sources.append(_provenance(df, path, None))

    finding_dir = Path(output_dir) / FINDING_ID
    finding_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "id": f"communication-{study}",
        "provenance": {"sources": sources},
        "methodology": _METHODOLOGY,
        "values": rows,
    }
    out_path = finding_dir / f"communication-{study}.json"
    _write_json(out_path, payload)
    return [out_path]


def _melt_sections(
    summary: pd.DataFrame, sections: list[Section], study: str, dim: str
) -> list[dict[str, Any]]:
    """One row per (summary row, section): the wide POP columns, tidied."""
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        gpu_count = int(row["n_gpus"])
        scale = float(row["comm_share_scale"])
        for s in sections:
            compute = float(row[s.compute_col])
            compute_max = float(row[s.compute_max_col])
            transfer = float(row[s.transfer_col])
            wait = float(row[s.wait_col])
            cuda = float(row[s.cuda_col])
            rows.append(
                {
                    "study": study,
                    "dim": dim,
                    "section": s.name,
                    "gpuCount": gpu_count,
                    "nodeCount": int(row["n_nodes"]),
                    "computeSec": number(compute),
                    "computeMaxSec": number(compute_max),
                    "transferSec": number(transfer),
                    "waitSec": number(wait),
                    "cudaSec": number(cuda),
                    "scaledComputeSec": number(compute * scale),
                    "scaledCommSec": number((transfer + wait) * scale),
                    "lbPct": (
                        number(100 * compute / compute_max) if compute_max > 0 else None
                    ),
                    "waitPct": number(100 * wait / compute) if compute > 0 else None,
                    "lbCombinedPct": number(float(row["lb_combined_pct"])),
                    "commEPct": number(float(row["comm_e_pct"])),
                    "commEPctSection": (
                        number(100 * compute_max / cuda) if cuda > 0 else None
                    ),
                    "bandwidthGBs": _bandwidth_gbs(row, s.name, gpu_count, transfer),
                }
            )
    return rows


def _bandwidth_gbs(
    row: pd.Series, section: str, gpu_count: int, transfer: float
) -> float | None:
    """Implied bandwidth (GB/s), or None where it can't be computed exactly.

    Same byte accounting as ``comm.py``'s ``transfer_bandwidth.png``: each of
    ``count`` collectives moves ``collective_bytes()`` bytes, which follows the
    algorithm NCCL actually chose (ring, or tree at 2 nodes) instead of
    assuming a ring everywhere -- see that helper's docstring.
    """
    if gpu_count <= 1 or transfer <= 0:
        return None
    count = collective_count(row, section)
    if count is None:
        return None
    transfer_bytes = count * collective_bytes(
        float(row["payload_bytes"]), gpu_count, int(row.get("n_nodes", 1))
    )
    return number(transfer_bytes / transfer / 1e9)
