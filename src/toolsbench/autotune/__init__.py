"""Tiling-parameter advisor for distributed inverse-problem benchmarks.

WHAT IT DECIDES
---------------
A distributed solver splits the denoiser across ranks by *space*: the image is
cut into tiles, each rank denoises the tiles it owns, and the results are
blended back together. Three numbers control that, and the configs here pin
them by hand::

    patch_size, overlap, max_batch_size

They are hard to guess and wrong in different ways. Too large a patch runs out
of memory. Too small a patch wastes compute, because every tile carries a halo
and the halo is pure duplicated work. Too small an overlap is worse than
either: it does not raise, it silently returns an image with seams along the
tile boundaries. This tool replaces the guessing with measurement.

Note that tiling is only one of two parallelisms. The physics and the data
fidelity are split by *operator* -- angle chunks in tomography, frames in
multiframe superresolution -- and applied to the whole image. Nothing here
changes that; its cost is measured and carried along.

TWO STAGES, RUN SEPARATELY
--------------------------
**Stage A -- seam** (:mod:`~toolsbench.autotune.seam`) measures how much halo a
tile needs before tiling reproduces what an untiled pass would have produced.
That is a property of the network's receptive field and its denoising sigma
only, so it is measured once on a small volume and reused at every image size.
It prints a table rather than picking a number: the floor is a trade between
agreement and compute, and the choice should be visible.

**Stage B -- plan** (:mod:`~toolsbench.autotune.select`) takes the overlap you
chose and reports which ``(patch_size, max_batch_size)`` pairs fit on the
hardware you name, cheapest first. It depends on image size, GPU count and GPU
memory, so it is re-run for each target machine.

They are separate because they answer different questions, change on different
schedules, and need different machines.

THE MEMORY MODEL
----------------
Everything that is not the denoiser is measured in one go, by running the real
config with the denoiser replaced by an identity
(:mod:`~toolsbench.autotune.probe`). That single run covers the CUDA context,
the physics and its workspace, the measurements, the solver's own buffers, the
initialisation and the loss -- none of it modelled, and none of it specific to
a problem or a solver, because whatever the config names is what gets built.

It returns two readings, because the physics burst and the denoiser burst do
not always coexist::

    probe_peak      occupancy at the physics burst
    probe_resident  what survives between bursts

The denoiser is then added to whichever base applies::

    denoiser = optimizer + tiler + kept + count x tile

    inference   peak = max(probe_peak, probe_resident + denoiser)
    training    peak = probe_peak + denoiser

``max`` for inference and ``+`` for training is physical: under
``no_grad`` the gradient step's temporaries are freed before the denoiser
allocates, while autograd retains the physics subgraph across the denoiser
call, so there they really do add.

+--------------+-------------------------------------------+---------------+
| term         | what it is                                | how obtained  |
+==============+===========================================+===============+
| ``probe_*``  | everything tiling never touches           | measured, one |
|              |                                           | identity run  |
+--------------+-------------------------------------------+---------------+
| ``optimizer``| the denoiser's gradients and optimizer    | measured      |
|              | state; zero for inference                 | (delta)       |
+--------------+-------------------------------------------+---------------+
| ``tile``     | what ONE tile costs at the exact window   | measured      |
|              | the candidate uses                        |               |
+--------------+-------------------------------------------+---------------+
| ``tiler``    | ``_apply_op``'s reflect-padded copy of    | arithmetic    |
|              | the signal, and the zeroed ``out_local``  |               |
|              | it reduces into                           |               |
+--------------+-------------------------------------------+---------------+
| ``kept``     | three copies of every tile, live at once: | arithmetic    |
|              | the concatenated batches, the forward     |               |
|              | outputs, and the per-patch tensors they   |               |
|              | are split back into                       |               |
+--------------+-------------------------------------------+---------------+

``count`` is how many tiles are alive simultaneously, which depends on
``checkpoint_batches``: with ``'always'`` one batch survives the backward's
recompute, with ``'never'`` every tile of every call stays in the graph.
``'never'`` is therefore the memory-hungry mode and the fast one, so it is
preferred wherever it fits and ``'always'`` appears only when it must. The
number of denoiser calls per step is *counted* by the identity stand-in rather
than read from ``n_iter`` -- a loss that calls the denoiser itself makes those
two differ.

Every element count includes the channel axis. Counting spatial elements only
under-predicts 2D by exactly the channel count and leaves 3D untouched.

UNITS
-----
Predictions are in ``allocated + CUDA context``, which is what actually runs
out: PyTorch releases its cached-but-idle blocks and retries ``cudaMalloc``
before raising, so reserve that is merely cached is elastic. The probe polls
``mem_get_info`` rather than reading ``max_memory_allocated`` because the
context and any non-PyTorch allocation -- ASTRA's workspace -- are invisible to
the PyTorch counter.


MODULES
-------
``seam``     stage A: the overlap floor, and a cache of past measurements
``probe``    the identity-denoiser run: everything that is not the denoiser
``memory``   the two denoiser measurements, and the arithmetic that sums them
``select``   stage B: enumerate candidates, keep what fits, rank by work
``config``   read a benchopt config into the problem shape
``cli``      the ``seam`` and ``plan`` subcommands

USAGE
-----
::

    # stage A -- once per denoiser and sigma; prints a table, picks nothing
    toolsbench autotune seam --config configs/experiments/multiframe_superres.yml

    # stage B -- once per target machine, with the overlap you chose
    toolsbench autotune plan --config configs/experiments/multiframe_superres.yml \\
        --overlap 32 --gpus 4 --gpu-mem 32

Hardware never comes from the config: how many GPUs and how much memory each
has are properties of where you are about to run, not of the experiment.
"""

from toolsbench.autotune.memory import (
    KEPT_TENSORS_PER_TILE,
    measure_optimizer_mb,
    measure_tile_mb,
    occupancy_mb,
    peak_mb,
    poll_peak_mb,
)
from toolsbench.autotune.probe import ProbeResult, identity_run
from toolsbench.autotune.seam import (
    OVERLAP_GRID,
    SEAM_LADDER,
    SeamResult,
    denoise_gain_db,
    largest_untiled,
    load_cache,
    save_cache,
    saturating_overlap,
    seam_floors,
    seam_input,
)
from toolsbench.autotune.select import (
    Candidate,
    default_patch_grid,
    geometry,
    rank,
    resolve_overlap,
    top3,
)

__all__ = [
    "Candidate",
    "KEPT_TENSORS_PER_TILE",
    "OVERLAP_GRID",
    "SEAM_LADDER",
    "SeamResult",
    "ProbeResult",
    "default_patch_grid",
    "denoise_gain_db",
    "identity_run",
    "geometry",
    "largest_untiled",
    "load_cache",
    "measure_optimizer_mb",
    "measure_tile_mb",
    "occupancy_mb",
    "poll_peak_mb",
    "peak_mb",
    "rank",
    "resolve_overlap",
    "saturating_overlap",
    "save_cache",
    "seam_floors",
    "seam_input",
    "top3",
]
