"""Communication figures for training communication-time experiments.

Figures are generic and shared with inference's comm_inference experiment --
see ``toolsbench.visualization.comm``.
"""

from __future__ import annotations

from pathlib import Path

from toolsbench.visualization.comm import Section, create_comm_visualizations
from toolsbench.visualization.common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    TIMING_WARMUP_ITERATIONS,
)

SECTIONS = [
    Section(
        "forward",
        "objective_forward_cuda_sec",
        "objective_forward_compute_sec",
        "objective_forward_compute_max_sec",
        "objective_forward_transfer_sec",
        "objective_forward_wait_sec",
    ),
    Section(
        "backward",
        "objective_backward_cuda_sec",
        "objective_backward_compute_sec",
        "objective_backward_compute_max_sec",
        "objective_backward_transfer_sec",
        "objective_backward_wait_sec",
    ),
]


def create_comm_time_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
    *,
    skip_warmup: int = TIMING_WARMUP_ITERATIONS,
) -> Path:
    """Create the communication figures for training comm-time runs.

    ``results`` is the parent directory containing a ``comm_2D`` and a
    ``comm_3D`` subfolder, each with its own benchopt parquet.
    """
    results_dir = Path(results)
    return create_comm_visualizations(
        results_dir / "comm_2D",
        results_dir / "comm_3D",
        SECTIONS,
        output_dir,
        "comm_time",
        skip_warmup=skip_warmup,
    )
