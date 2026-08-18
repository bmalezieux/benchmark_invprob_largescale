"""Communication figures for distributed PnP inference.

For the ``comm_inference_2D`` / ``comm_inference_3D`` experiments: a distributed
PnP run profiled with ``profiler_mode: torch``. Figures are generic and shared
with training's comm_time experiment -- see ``toolsbench.visualization.comm``.
"""

from __future__ import annotations

from pathlib import Path

from toolsbench.visualization.comm import Section, create_comm_visualizations
from toolsbench.visualization.common import DEFAULT_OUTPUT_DIR, TIMING_WARMUP_ITERATIONS

DEFAULT_SKIP_WARMUP = TIMING_WARMUP_ITERATIONS

SECTIONS = [
    Section(
        "denoiser",
        "objective_denoise_cuda_sec",
        "objective_denoise_compute_sec",
        "objective_denoise_compute_max_sec",
        "objective_denoise_transfer_sec",
        "objective_denoise_wait_sec",
    ),
    Section(
        "gradient",
        "objective_gradient_cuda_sec",
        "objective_gradient_compute_sec",
        "objective_gradient_compute_max_sec",
        "objective_gradient_transfer_sec",
        "objective_gradient_wait_sec",
    ),
]


def create_comm_inference_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    skip_warmup: int = DEFAULT_SKIP_WARMUP,
) -> Path:
    """Create the communication figures for distributed PnP inference runs.

    ``results`` is the parent directory containing a ``comm_2D`` and a
    ``comm_3D`` subfolder, each with its own benchopt parquet.
    """
    results_dir = Path(results)
    return create_comm_visualizations(
        results_dir / "comm_2D",
        results_dir / "comm_3D",
        SECTIONS,
        output_dir,
        "comm_inference",
        skip_warmup=skip_warmup,
    )
