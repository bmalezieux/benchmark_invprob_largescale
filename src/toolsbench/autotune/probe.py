"""Run the real benchmark config with the denoiser replaced by an identity.

That single run gives everything the tiling does not touch -- the CUDA context,
the physics and its workspace, the measurements, the solver's own buffers, the
initialisation, the loss -- without any of it being modelled. Whatever the
config names is what gets built, so no problem or solver is special-cased here.

Two readings come back. The peak includes the physics burst; the resident
figure is what survives it. :func:`peak_mb` needs both, because in inference
the physics burst and the denoiser burst never coexist.

The physics is measured undistributed -- every operator on one rank -- so the
reading is an upper bound on any split of it, and does not depend on how many
GPUs the run is being planned for.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from benchopt.benchmark import Benchmark

import toolsbench.solver

from toolsbench.autotune.memory import poll_peak_mb

#: Distribution switches the probe forces off, for two different reasons.
#:
#: The denoiser ones -- solvers name it either way -- because a tiling wrapper
#: would put the padded copy, out_local and the tile concatenations into the
#: reading, and peak_mb already counts those as `tiler` and `kept`.
#:
#: `distribute_physics` because leaving it on measures only rank 0's share of
#: the operators, which under-counts any run with fewer ranks. Off, every
#: operator is measured on one rank: the largest the physics can be, an upper
#: bound on every split of it, and independent of the target GPU count.
DISABLED_FLAGS = (
    "distribute_denoiser",
    "distribute_model",
    "distribute_physics",
)


@dataclass(frozen=True)
class ProbeResult:
    peak_mb: float  # occupancy at the physics burst
    resident_mb: float  # what stays between bursts
    n_calls: int  # denoiser calls per step, counted not parsed


class _IdentityDenoiser(torch.nn.Module):
    """Stands in for the denoiser and counts how often it is reached.

    An identity rather than a removal, so the solver still runs exactly as
    configured -- including any call the loss makes, which is why the count is
    trustworthy in a way that reading ``n_iter`` is not.
    """

    def __init__(self, counter: list):
        super().__init__()
        self._counter = counter

    def forward(self, x, *args, **kwargs):
        self._counter[0] += 1
        return x


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(f"_probe_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find(root: Path, kind: str, name: str, attr: str):
    """The benchopt object a config names, by its declared `name`.

    Matched on the class attribute rather than the filename: benchopt names and
    module names diverge (`UnrolledPnP` lives in `unrolled_pnp.py`), and the
    config carries the benchopt name.
    """
    folder = root / kind
    wanted = name.lower().replace("-", "_").replace(" ", "_")
    candidates = sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
    for path in candidates:
        obj = getattr(_import(path), attr, None)
        declared = getattr(obj, "name", None)
        if (
            obj is not None
            and declared
            and declared.lower().replace("-", "_") == wanted
        ):
            return obj
    raise FileNotFoundError(
        f"no {attr} named '{name}' in {folder}; found "
        f"{[p.stem for p in candidates]}"
    )


@contextmanager
def _world_size(n: int):
    """Let the benchmark objects see the world size they are being planned for.

    WORLD_SIZE without RANK gives world_size=n, rank=0, use_dist=False, so the
    dataset and solver set themselves up as they would in production without
    needing n GPUs. The memory reading itself no longer depends on n, since
    :data:`DISABLED_FLAGS` keeps the physics whole.
    """
    previous = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(n)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous


@contextmanager
def _identity_denoiser(counter: list):
    """Replace the factory every solver builds its denoiser with.

    Patching the factory rather than the built object: solvers construct the
    denoiser inside their own setup, so there is nothing to swap afterwards.
    Each solver module holds its own binding of the imported name, so all of
    them are patched, not just the one this case uses.
    """
    package = toolsbench.solver
    owners = []
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        if hasattr(module, "create_drunet_denoiser"):
            owners.append((module, module.create_drunet_denoiser))
    for module, _ in owners:
        module.create_drunet_denoiser = lambda *a, **k: _IdentityDenoiser(counter)
    try:
        yield
    finally:
        for module, original in owners:
            module.create_drunet_denoiser = original


def identity_run(case, world_size=1, n_iter=2, device="cuda") -> ProbeResult:
    """Measure the config with its denoiser neutralised.

    Every distribution switch is forced off; see :data:`DISABLED_FLAGS` for
    why each. The physics is therefore measured whole, which is the safe
    direction and makes the reading independent of ``world_size``.
    """
    # Datasets reach their data through benchopt.config.get_data_path, which
    # reads the *running* benchmark -- a global that Benchmark.__init__ sets.
    # Nothing constructs one outside a benchopt run, so the probe does.
    Benchmark(case.benchmark_dir)

    root = case.benchmark_dir
    if root is None:
        raise ValueError(
            "no benchmark directory for this case; the probe builds the "
            "benchmark's own objects, so it needs a config inside one"
        )
    DatasetCls = _find(root, "datasets", case.dataset, "Dataset")
    SolverCls = _find(root, "solvers", case.solver, "Solver")

    counter = [0]
    with _world_size(world_size):
        dataset = DatasetCls.get_instance(**case.dataset_params)
        data = dataset.get_data()

        params = dict(case.solver_params)
        params.update(patch_size=0, overlap=0)
        params.update({flag: False for flag in DISABLED_FLAGS})
        params = {
            key: value for key, value in params.items() if key in SolverCls.parameters
        }
        solver = SolverCls.get_instance(**params)

        with _identity_denoiser(counter):
            solver.set_objective(**data)
            torch.cuda.empty_cache()

            # Solvers call the denoiser once during setup, before the first
            # callback -- an unrolled PGD runs a full inference there. Those
            # calls are not per-step work, so the counter is zeroed when the
            # first callback fires and only whole steps are counted. Dividing
            # the raw total instead made the answer drift with n_iter: on a
            # 3-iteration config it reported 9//2 = 4 at n_iter=2 and 15//4 = 3
            # at n_iter=4, for the same solver.
            steps = [0]

            def callback():
                if steps[0] == 0:
                    counter[0] = 0
                steps[0] += 1
                return steps[0] <= n_iter

            peak, resident = poll_peak_mb(lambda: solver.run(callback), device=device)

    return ProbeResult(
        peak_mb=peak,
        resident_mb=resident,
        n_calls=max(1, round(counter[0] / max(1, n_iter))),
    )
