"""Plot-ready website data for the torch.compile finding.

Reuses the same aggregation functions as the PNG visualizations
(:mod:`toolsbench.visualization.inference.compile_speedup`) so the website
numbers and the diagnostic plots can never silently diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toolsbench.visualization.common import load_results
from toolsbench.visualization.inference.compile_speedup import (
    GPU_RIDGE,
    compile_summary,
    denoiser_summary,
    load_denoiser_df,
)
from toolsbench.visualization.website.common import (
    SCHEMA_VERSION,
    finite,
    number,
    provenance as _provenance,
    write_json as _write_json,
)

FINDING_ID = "compile_speedup"
ROOFLINE_GPU = "h100"


def create_compile_speedup_website_data(
    *,
    denoiser_results: str | Path,
    pnp_results: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Create the plot-ready datasets used by the torch.compile finding."""
    denoiser_df, denoiser_path = load_denoiser_df(denoiser_results)
    denoiser_prov = _provenance(denoiser_df, denoiser_path, None)
    denoiser_data = denoiser_summary(denoiser_df)
    if denoiser_data.empty:
        raise ValueError("No timed denoiser iterations found for speedup comparison.")

    pnp_df, pnp_path = load_results(pnp_results)
    pnp_prov = _provenance(pnp_df, pnp_path, None)
    pnp_data = compile_summary(pnp_df)

    finding_dir = Path(output_dir) / FINDING_ID
    finding_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            finding_dir / "denoiser-speedup.json",
            _denoiser_speedup_payload(denoiser_data, denoiser_prov),
        ),
        (
            finding_dir / f"roofline-{ROOFLINE_GPU}.json",
            _roofline_payload(denoiser_data, denoiser_prov, gpu=ROOFLINE_GPU),
        ),
        (
            finding_dir / "pnp-compile-modes.json",
            _pnp_compile_modes_payload(pnp_data, pnp_prov),
        ),
    ]
    for path, payload in outputs:
        _write_json(path, payload)
    return [path for path, _ in outputs]


def _denoiser_speedup_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    values = []
    for _, row in summary.sort_values(["gpu", "dim", "area", "denoiser"]).iterrows():
        values.append(
            {
                "gpu": str(row["gpu"]) or "unknown",
                "denoiser": str(row["denoiser"]),
                "shape": str(row["shape"]),
                "dim": int(row["dim"]),
                "eagerSec": finite(row["eager_sec"]),
                "compiledSec": finite(row["compiled_sec"]),
                "speedup": number(float(row["speedup"])),
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "compile-denoiser-speedup",
        "provenance": provenance,
        "methodology": {
            "comparison": (
                "steady-state per-iteration denoiser time, compile=None (eager) "
                "vs compile=pre, mean over the last 5 timed iterations"
            ),
            "speedup": "eagerSec / compiledSec",
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _roofline_payload(
    summary: pd.DataFrame, provenance: dict[str, Any], *, gpu: str
) -> dict[str, Any]:
    panel = summary[summary["gpu"].str.lower() == gpu.lower()]
    panel = panel[panel["arith_intensity"].notna() & panel["speedup"].notna()]
    if panel.empty:
        raise ValueError(f"No {gpu} rows with arithmetic-intensity data found.")

    values = []
    for _, row in panel.sort_values(["denoiser", "dim", "area"]).iterrows():
        values.append(
            {
                "denoiser": str(row["denoiser"]),
                "shape": str(row["shape"]),
                "dim": int(row["dim"]),
                "arithIntensityFlopByte": number(float(row["arith_intensity"])),
                "speedup": number(float(row["speedup"])),
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": f"compile-denoiser-roofline-{gpu}",
        "provenance": provenance,
        "methodology": {
            "gpu": gpu,
            "ridgePointFlopByte": number(GPU_RIDGE[gpu.lower()]),
            "ridgeDefinition": (
                "peak TF32 tensor-core TFLOPS / peak memory bandwidth TB/s; "
                "points left of the ridge are memory-bound, right are compute-bound"
            ),
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _pnp_compile_modes_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    values = []
    for _, row in summary.sort_values(["n_gpus", "compile_order"]).iterrows():
        values.append(
            {
                "compileMode": str(row["compile_mode"]),
                "gpuCount": int(row["n_gpus"]),
                "firstIterSec": number(float(row["first_iter_mean"])),
                "firstIterStdSec": number(float(row["first_iter_std"])),
                "stableIterSec": number(float(row["stable_iter_mean"])),
                "stableIterStdSec": number(float(row["stable_iter_std"])),
                "speedup": number(float(row["speedup"])),
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "compile-pnp-modes",
        "provenance": provenance,
        "methodology": {
            "compileModes": {
                "pre": "torch.compile applied to the denoiser (and physics ops) "
                "before the distributed wrapper",
                "post": "torch.compile applied after the distributed wrapper",
                "fused": "torch.compile applied to the whole per-iteration step "
                "(gradient + prox + clamp) as one region",
            },
            "firstIteration": "includes one-time graph compilation cost",
            "stableIteration": "mean over the last 5 timed iterations",
            "speedup": "baseline 1-GPU compile=None stable time / stable time",
            "independentRepetitions": 1,
        },
        "values": values,
    }
