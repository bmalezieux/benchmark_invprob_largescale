"""Plot-ready website data for the distributed training finding.

Covers the three training experiments that share a single subject — the GPU
memory budget of training on large-scale data, where the activation footprint
is set by the size of the data rather than the size of the model — and the time
each lever costs:

* strong scaling and weak scaling (both derived from the strong-scaling run,
  weak-scaling ladders being the runs that share a per-GPU workload),
* activation checkpointing (``checkpoint_batches`` ``always`` vs ``never``),
* ``max_batch_size``, which sets the size of the unit checkpointing refuses
  to store.

Aggregation reuses :func:`toolsbench.visualization.common.load_training_summary`
and :func:`toolsbench.visualization.scaling.resolve_scaling_baseline`, the same
functions behind the diagnostic PNGs, so the website and the PNGs cannot
silently disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toolsbench.visualization.common import (
    format_image_size,
    load_training_summary,
)
from toolsbench.visualization.scaling import resolve_scaling_baseline
from toolsbench.visualization.website.common import (
    SCHEMA_VERSION,
    finite,
    number,
    provenance as _provenance,
    write_json as _write_json,
)

FINDING_ID = "training_scaling"

SCALING_CONFIG = "benchmark_training/configs/experiments/strong_scaling.yml"
CHECKPOINTING_CONFIG = "benchmark_training/configs/experiments/checkpointing.yml"
BATCH_SIZE_CONFIG = "benchmark_training/configs/experiments/batch_size.yml"

DISTRIBUTE_COLUMN = "p_solver_distribute_model"

#: Per-GPU memory of the cluster's V100 cards, in MiB.
DEVICE_MEMORY_MB = 32768

#: Runs that are absent from the checkpointing parquet because they do not fit
#: in :data:`DEVICE_MEMORY_MB`, not because they were never attempted. Recorded
#: explicitly so the figure can mark them instead of silently dropping a bar —
#: they carry no measured numbers, only the reason they are missing.
DECLARED_INFEASIBLE_CHECKPOINTING = [
    {"gpuCount": 1, "maxBatchSize": 1, "checkpointMode": "never"},
    {"gpuCount": 4, "maxBatchSize": 1, "checkpointMode": "never"},
]

#: Number of decimals used to bucket runs into weak-scaling ladders. Runs whose
#: per-GPU workload agrees to this precision are treated as the same ladder.
WEAK_SCALING_PRECISION = 2


def create_training_scaling_website_data(
    *,
    scaling_results: str | Path,
    checkpointing_results: str | Path,
    batch_size_results: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Create the plot-ready datasets used by the training scaling finding."""
    scaling, scaling_path = load_training_summary(scaling_results)
    scaling_prov = _provenance(scaling, scaling_path, SCALING_CONFIG)

    checkpointing, checkpointing_path = load_training_summary(checkpointing_results)
    checkpointing_prov = _provenance(
        checkpointing, checkpointing_path, CHECKPOINTING_CONFIG
    )

    batch_size, batch_size_path = load_training_summary(batch_size_results)
    batch_size_prov = _provenance(batch_size, batch_size_path, BATCH_SIZE_CONFIG)

    finding_dir = Path(output_dir) / FINDING_ID
    finding_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            finding_dir / "strong-scaling.json",
            _strong_scaling_payload(scaling, scaling_prov),
        ),
        (
            finding_dir / "weak-scaling.json",
            _weak_scaling_payload(scaling, scaling_prov),
        ),
        (
            finding_dir / "checkpointing.json",
            _checkpointing_payload(checkpointing, checkpointing_prov),
        ),
        (
            finding_dir / "batch-size.json",
            _batch_size_payload(batch_size, batch_size_prov),
        ),
    ]
    for path, payload in outputs:
        _write_json(path, payload)
    return [path for path, _ in outputs]


def _strong_scaling_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Per-image-size speedup and parallel efficiency against a fixed baseline."""
    distributed = summary[summary[DISTRIBUTE_COLUMN].astype(bool)]
    if distributed.empty:
        raise ValueError("No distributed runs found in the training scaling results.")

    values: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    for image_size, group in distributed.groupby("training_image_size", sort=True):
        rows = group.sort_values("n_gpus")
        baseline_time, baseline_gpus, baseline_label = resolve_scaling_baseline(
            group, rows, DISTRIBUTE_COLUMN
        )
        label = format_image_size(image_size)
        baselines[label] = {
            "gpuCount": baseline_gpus,
            "timeSec": number(baseline_time),
            "reference": baseline_label,
        }
        for _, row in rows.iterrows():
            gpu_count = int(row["n_gpus"])
            speedup = baseline_time / float(row["avg_total_time_sec"])
            values.append(
                {
                    "imageSize": label,
                    "gpuCount": gpu_count,
                    "timeSec": finite(row["avg_total_time_sec"]),
                    "forwardSec": finite(row["avg_forward_time_sec"]),
                    "backwardSec": finite(row["avg_backward_time_sec"]),
                    "peakMemoryMb": finite(row["max_gpu_mb"]),
                    "speedup": number(speedup),
                    "efficiencyPct": number(
                        100.0 * speedup * baseline_gpus / gpu_count
                    ),
                }
            )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "training-strong-scaling",
        "provenance": provenance,
        "methodology": {
            "stepTime": (
                "mean per-step forward + backward time; the benchopt wall-clock "
                "total also contains data loading and logging and is not used"
            ),
            "speedup": "baseline step time / step time",
            "efficiency": "100 x speedup x baselineGpuCount / gpuCount",
            "baselineByImageSize": baselines,
            "checkpointing": "always, for every run in this experiment",
            "maxBatchSize": 1,
            "deviceMemoryMb": DEVICE_MEMORY_MB,
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _weak_scaling_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Weak-scaling ladders: runs that hold megapixels-per-GPU constant.

    These are not a separate experiment. The strong-scaling grid sweeps three
    image sizes over overlapping GPU counts, so runs sharing a per-GPU workload
    (for example 4096 px on 4 GPUs and 8192 px on 16) already form weak-scaling
    ladders; this groups them out of the same parquet.
    """
    distributed = summary[summary[DISTRIBUTE_COLUMN].astype(bool)].copy()
    distributed["local_mpix"] = (
        distributed["training_mpix"] / distributed["n_gpus"]
    ).round(WEAK_SCALING_PRECISION)

    values: list[dict[str, Any]] = []
    for local_mpix, group in distributed.groupby("local_mpix", sort=True):
        rows = group.sort_values("n_gpus")
        if len(rows) < 2:
            # A single point cannot show whether the workload scales.
            continue
        baseline = rows.iloc[0]
        baseline_time = float(baseline["avg_total_time_sec"])
        baseline_gpus = int(baseline["n_gpus"])
        for _, row in rows.iterrows():
            values.append(
                {
                    "localWorkloadMpix": number(float(local_mpix)),
                    "imageSize": format_image_size(row["training_image_size"]),
                    "gpuCount": int(row["n_gpus"]),
                    "timeSec": finite(row["avg_total_time_sec"]),
                    "peakMemoryMb": finite(row["max_gpu_mb"]),
                    "peakMemoryGb": number(float(row["max_gpu_mb"]) / 1024.0),
                    "forwardPeakMemoryMb": finite(row["max_forward_gpu_mb"]),
                    "timeRatio": number(
                        float(row["avg_total_time_sec"]) / baseline_time
                    ),
                    "efficiencyPct": number(
                        100.0 * baseline_time / float(row["avg_total_time_sec"])
                    ),
                    "baselineGpuCount": baseline_gpus,
                }
            )

    if not values:
        raise ValueError("No weak-scaling ladder found: every workload has one point.")

    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "training-weak-scaling",
        "provenance": provenance,
        "methodology": {
            "source": (
                "the strong-scaling parquet; a ladder is the set of runs whose "
                "megapixels-per-GPU agree to "
                f"{WEAK_SCALING_PRECISION} decimals"
            ),
            "ladderMinimumPoints": 2,
            "baseline": "the smallest-GPU run of each ladder",
            "efficiency": "100 x baseline step time / step time",
            "memoryNote": (
                "peak memory is per GPU; it is not expected to be flat along a "
                "ladder, because per-rank tensors sized by the global image are "
                "replicated on every rank"
            ),
            "deviceMemoryMb": DEVICE_MEMORY_MB,
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _checkpointing_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Activation-checkpointing cost in time against what it saves in memory."""
    values: list[dict[str, Any]] = []
    for _, row in summary.sort_values(
        ["n_gpus", "p_solver_max_batch_size", "p_solver_checkpoint_batches"]
    ).iterrows():
        values.append(
            {
                "checkpointMode": str(row["p_solver_checkpoint_batches"]),
                "gpuCount": int(row["n_gpus"]),
                "maxBatchSize": int(row["p_solver_max_batch_size"]),
                "timeSec": finite(row["avg_total_time_sec"]),
                "forwardSec": finite(row["avg_forward_time_sec"]),
                "backwardSec": finite(row["avg_backward_time_sec"]),
                "peakMemoryMb": finite(row["max_gpu_mb"]),
                "forwardPeakMemoryMb": finite(row["max_forward_gpu_mb"]),
                "backwardPeakMemoryMb": finite(row["max_backward_gpu_mb"]),
                "status": "measured",
            }
        )

    measured = {(value["gpuCount"], value["checkpointMode"]) for value in values}
    for declared in DECLARED_INFEASIBLE_CHECKPOINTING:
        key = (declared["gpuCount"], declared["checkpointMode"])
        if key in measured:
            raise ValueError(
                f"Configuration {key} is declared infeasible but was measured; "
                "update DECLARED_INFEASIBLE_CHECKPOINTING."
            )
        values.append({**declared, "status": "did-not-fit"})

    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "training-checkpointing",
        "provenance": provenance,
        "methodology": {
            "checkpointModes": {
                "precondition": (
                    "gradients must be tracked; a no-grad pass checkpoints nothing "
                    "regardless of the mode"
                ),
                "always": (
                    "every patch-batch forward is wrapped in torch.utils.checkpoint, "
                    "so denoiser activations are discarded and recomputed in backward"
                ),
                "never": "all patch-batch activations are kept for backward",
                "auto": (
                    "wraps every batch when the rank ends up with more than one "
                    "batch, that is when ceil(patches / max_batch_size) > 1, and "
                    "behaves like never otherwise — so a max_batch_size large "
                    "enough to hold a rank's whole patch list disables it"
                ),
            },
            "checkpointedUnit": (
                "one patch-batch, so max_batch_size sets how much is stored even "
                "when checkpointing is on"
            ),
            "statusValues": {
                "measured": "run present in the parquet",
                "did-not-fit": (
                    "not run: the configuration exceeds the "
                    f"{DEVICE_MEMORY_MB} MiB per-GPU budget. No measured numbers "
                    "are attached to these entries"
                ),
            },
            "deviceMemoryMb": DEVICE_MEMORY_MB,
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _batch_size_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    """What a larger checkpoint unit costs in memory and returns in time.

    Efficiency uses a **single** baseline for every series — the largest batch
    size at the smallest GPU count — rather than one baseline per GPU count.
    Normalizing each GPU count against its own batch-1 run would force every
    curve to start at the same place and hide the absolute difference between
    batch sizes, which is the whole question here. This matches the baseline
    rule in :mod:`toolsbench.visualization.training.batch_size`.
    """
    largest_batch = summary["p_solver_max_batch_size"].max()
    candidates = summary[summary["p_solver_max_batch_size"] == largest_batch]
    baseline = candidates.loc[candidates["n_gpus"].idxmin()]
    baseline_time = float(baseline["avg_total_time_sec"])
    baseline_gpus = int(baseline["n_gpus"])

    values: list[dict[str, Any]] = []
    for _, row in summary.sort_values(["p_solver_max_batch_size", "n_gpus"]).iterrows():
        gpu_count = int(row["n_gpus"])
        values.append(
            {
                "gpuCount": gpu_count,
                "maxBatchSize": int(row["p_solver_max_batch_size"]),
                "timeSec": finite(row["avg_total_time_sec"]),
                "efficiencyPct": number(
                    100.0
                    * baseline_time
                    * baseline_gpus
                    / (gpu_count * float(row["avg_total_time_sec"]))
                ),
                "peakMemoryMb": finite(row["max_gpu_mb"]),
                "peakMemoryGb": number(float(row["max_gpu_mb"]) / 1024.0),
                "forwardPeakMemoryMb": finite(row["max_forward_gpu_mb"]),
            }
        )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "training-batch-size",
        "provenance": provenance,
        "methodology": {
            "maxBatchSize": (
                "how many of a rank's patches go through the denoiser together; "
                "the patch grid is unchanged, so a step does the same total work "
                "at every batch size — this is a utilization knob, not more data"
            ),
            "efficiency": (
                "100 x baseline step time x baselineGpuCount / "
                "(gpuCount x step time)"
            ),
            "baseline": {
                "maxBatchSize": int(largest_batch),
                "gpuCount": baseline_gpus,
                "timeSec": number(baseline_time),
                "rationale": (
                    "one baseline shared by every series, so the curves are "
                    "comparable in absolute terms rather than only within a "
                    "series; a per-GPU-count baseline would pin every curve to "
                    "the same start and hide the batch-size effect"
                ),
            },
            "checkpointing": "always, for every run in this experiment",
            "deviceMemoryMb": DEVICE_MEMORY_MB,
            "independentRepetitions": 1,
        },
        "values": values,
    }
