"""Read a benchopt config into the shape of the problem, and nothing else.

Image size, denoiser, inference or unrolled training, unrolled iterations, and
`denoiser_sigma` -- which matters more than it looks, since the seam floor
tracks the denoising strength. Hardware never comes from here; it comes from
the command line.

The patch size and overlap the config pins are read too, but only to print
alongside the recommendation that replaces them.

The dataset name, its parameters and the solver's are carried through verbatim
for the probe, which rebuilds the benchmark's own objects from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Solvers that build a training graph. Anything else is treated as inference.
TRAINING_SOLVERS = {"unrolledpnp"}


@dataclass(frozen=True)
class BenchCase:
    """One configuration to advise on."""

    solver: str
    denoiser: str
    img_size: tuple[int, ...]
    train: bool
    n_iter: int | None  # a hint only; the call count is counted, not parsed
    config_patch: int | None
    config_overlap: int | None
    sigma: float | None = None  # denoiser_sigma; the seam floor tracks it

    # What the probe needs to rebuild the benchmark's own objects. Carried
    # verbatim rather than interpreted: the probe runs whatever the config
    # names, so nothing here is specific to a problem or a solver.
    dataset: str | None = None
    dataset_params: dict = field(default_factory=dict, compare=False, repr=False)
    solver_params: dict = field(default_factory=dict, compare=False, repr=False)
    benchmark_dir: Path | None = None

    @property
    def ndim(self) -> int:
        return len(self.img_size) - 2

    def describe(self) -> str:
        kind = "unrolled training" if self.train else "inference"
        spatial = "x".join(str(d) for d in self.img_size[2:])
        iters = f", n_iter={self.n_iter}" if self.train and self.n_iter else ""
        return (
            f"{self.solver} · {self.denoiser} {self.ndim}D · {spatial} · "
            f"{self.img_size[1]}ch · {kind}{iters}"
        )


def _entries(block) -> list[tuple[str, dict]]:
    """benchopt writes `- Name:` with params, or a bare `- Name`."""
    out = []
    for item in block or []:
        if isinstance(item, str):
            out.append((item, {}))
        elif isinstance(item, dict):
            for name, params in item.items():
                out.append((name, params if isinstance(params, dict) else {}))
    return out


def expand_params(params: dict) -> dict[str, list]:
    """Flatten benchopt's comma-joined tuple keys into one list per name.

    `"a, b": [[1, 2], [3, 4]]` means a in {1, 3} and b in {2, 4}, positionally
    paired. The pairing does not matter here -- the advisor only needs the set
    of values each name takes.
    """
    flat: dict[str, list] = {}
    for key, value in (params or {}).items():
        names = [n.strip() for n in str(key).split(",")]
        if len(names) == 1:
            flat.setdefault(names[0], []).extend(
                value if isinstance(value, list) else [value]
            )
            continue
        for row in value or []:
            row = row if isinstance(row, (list, tuple)) else [row]
            for name, item in zip(names, row):
                flat.setdefault(name, []).append(item)
    return flat


def _first(flat: dict, name: str, default=None):
    values = flat.get(name) or []
    return values[0] if values else default


def parse_config(
    path,
    *,
    ndim: int | None = None,
    channels: int | None = None,
    image_size: int | None = None,
) -> list[BenchCase]:
    """Every distinct case a config file describes.

    A config is usually a sweep, so this returns one case per distinct image
    size. GPU count is deliberately absent: it comes from the command line.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    benchmark_dir = _benchmark_root(path)
    datasets = _entries(raw.get("dataset"))
    solvers = _entries(raw.get("solver"))
    if not solvers:
        raise ValueError(f"{path}: no solver block to read")

    dataset_name = datasets[0][0] if datasets else None
    dataset_flat = expand_params(datasets[0][1]) if datasets else {}
    cases: list[BenchCase] = []
    for name, params in solvers:
        flat = expand_params(params)
        train = name.lower() in TRAINING_SOLVERS or "checkpoint_batches" in flat
        denoiser = _first(flat, "denoiser", "drunet")
        n_channels = channels or _first(dataset_flat, "channels", 3)

        shapes = (
            [(int(image_size),)]
            if image_size is not None
            else _shapes(flat, dataset_flat, path)
        )
        default_dim = ndim or (3 if _looks_3d(dataset_flat) else 2)
        for shape in shapes:
            spatial = shape * default_dim if len(shape) == 1 else shape
            if ndim and len(spatial) != ndim:
                spatial = (spatial[0],) * ndim
            cases.append(
                BenchCase(
                    solver=name,
                    denoiser=denoiser,
                    img_size=(1, int(n_channels), *(int(d) for d in spatial)),
                    train=train,
                    n_iter=_first(flat, "n_iter"),
                    sigma=_first(flat, "denoiser_sigma"),
                    config_patch=_first(flat, "patch_size"),
                    config_overlap=_first(flat, "overlap"),
                    dataset=dataset_name,
                    dataset_params=_singles(dataset_flat),
                    solver_params=_singles(flat),
                    benchmark_dir=benchmark_dir,
                )
            )
    return cases


def _benchmark_root(path) -> Path | None:
    """The benchmark a config belongs to: the nearest parent holding the
    ``datasets`` and ``solvers`` the probe has to build. Configs live at
    ``<benchmark>/configs/...``, and there are several benchmarks in this repo.
    """
    for parent in Path(path).resolve().parents:
        if (parent / "datasets").is_dir() and (parent / "solvers").is_dir():
            return parent
    return None


def _singles(flat: dict) -> dict:
    """One value per parameter, for handing back to a benchopt object.

    A config is a sweep; the probe runs one point of it. The first value of
    each parameter is that point, matching the case this row describes.
    """
    return {name: values[0] for name, values in flat.items() if values}


def _shapes(solver_flat: dict, dataset_flat: dict, path) -> list[tuple[int, ...]]:
    """Distinct spatial shapes, from wherever the config happens to put them.

    Both blocks are searched because the repo does it both ways: the inference
    configs pin `image_size` on the dataset, the training ones sweep it inside a
    solver tuple alongside the slurm settings. A scalar means a square or a cube
    and takes its dimensionality from the dataset; a list of two or three is
    already a shape, so `[[512, 512, 512]]` is one 3D volume rather than a sweep
    over three sizes.
    """
    found = []
    for flat in (solver_flat, dataset_flat):
        for key in ("image_size", "size"):
            for value in flat.get(key) or []:
                if isinstance(value, (list, tuple)):
                    if len(value) in (2, 3):
                        found.append(tuple(int(d) for d in value))
                    elif len(value) == 1 and isinstance(value[0], int):
                        found.append((int(value[0]),))
                elif isinstance(value, int):
                    found.append((value,))
    if not found:
        raise ValueError(
            f"{path}: no image_size found in the dataset or solver block; "
            f"pass --image explicitly"
        )
    return sorted(set(found))


def _looks_3d(dataset_flat: dict) -> bool:
    """A scalar size takes its dimensionality from what the dataset declares."""
    markers = [str(v).lower() for v in (dataset_flat.get("data") or [])]
    return any("3d" in m for m in markers)
