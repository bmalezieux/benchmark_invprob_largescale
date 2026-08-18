"""Shared helpers for website JSON data generators.

Every finding's website module (``inference_scaling.py``, ``compile_speedup.py``,
...) writes the same JSON envelope — ``schemaVersion``/``provenance``/
``methodology``/``values`` — so the provenance-hashing and serialization
helpers live here once instead of being duplicated per finding.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def provenance(df: pd.DataFrame, path: Path, config: str | None) -> dict[str, Any]:
    """Build the ``provenance`` block: source parquet path/hash + run metadata."""
    metadata = {
        "source": path.as_posix(),
        "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": config,
    }
    for source_column, output_key in [
        ("run_date", "runDate"),
        ("version-cuda", "accelerator"),
        ("platform", "platform"),
        ("benchmark-git-tag", "benchmarkGitTag"),
    ]:
        if source_column not in df.columns:
            continue
        values = df[source_column].dropna().unique()
        metadata[output_key] = json_scalar(values[0]) if len(values) == 1 else None
    return metadata


def json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def number(value: float) -> float:
    return round(float(value), 6)


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Website result data must be finite, got {value!r}")
    return number(result)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
