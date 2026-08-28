"""toolsbench package."""

from __future__ import annotations

import builtins
import sys
import typing

# SimAI-Bench may eagerly import Dragon symbols in worker contexts.
# Provide harmless fallbacks so non-Dragon runs keep working.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["vizinference"]:
        from toolsbench.visualization.cli import main as visualization_main

        return visualization_main("vizinference", argv[1:])
    if argv[:1] == ["viztraining"]:
        from toolsbench.visualization.cli import main as visualization_main

        return visualization_main("viztraining", argv[1:])
    if argv[:1] == ["vizwebsite"]:
        from toolsbench.visualization.cli import main as visualization_main

        return visualization_main("vizwebsite", argv[1:])
    if argv[:1] == ["autotune"]:
        from toolsbench.autotune.cli import main as autotune_main

        return autotune_main(argv[1:])
    if argv[:1] == ["prepareweights"]:
        from toolsbench.utils import download_denoiser_weights

        try:
            download_denoiser_weights(argv[1:] or None)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        return 0

    print(
        "toolsbench installs shared benchmark utilities. "
        "Run benchmarks with `benchopt run <benchmark_path>` or create "
        "visualizations with `toolsbench vizinference --help` or "
        "`toolsbench viztraining --help`. Generate website result data with "
        "`toolsbench vizwebsite --help`. Cache pretrained denoiser weights "
        "for offline compute nodes with `toolsbench prepareweights [name ...]`. "
        "Suggest tiling parameters for a distributed config with "
        "`toolsbench autotune --help`."
    )
    return 0
