"""The denoiser's own memory, and the arithmetic that combines every term.

Two measurements live here -- the optimizer delta and one tile -- plus two
exact byte counts from the tiler. Everything that is not the denoiser comes
from :mod:`toolsbench.autotune.probe`. The package docstring has the model.
"""

from __future__ import annotations

import threading
import time

import torch

MB = 1024**2
BYTES_PER_FLOAT = 4

#: Copies of every tile that are live at the same time inside ``_apply_op``:
#: ``apply_batching``'s concatenated input, the forward outputs it collects,
#: and the per-patch tensors ``unpack_batched_results`` splits those back into.
#: A stage-by-stage reading of a real call showed each of the three adding
#: exactly ``k x window x C x 4B``.
KEPT_TENSORS_PER_TILE = 3


# --------------------------------------------------------------------------
# reading the device
# --------------------------------------------------------------------------


def occupancy_mb(device="cuda") -> float:
    """Everything the process holds, including what PyTorch cannot see.

    ``memory_allocated`` misses the CUDA context and any library that calls
    ``cudaMalloc`` itself -- ASTRA does. Right after ``cuda.init()`` this
    returns ~230 MB while ``memory_allocated`` returns 0.
    """
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / MB


def poll_peak_mb(fn, device="cuda", poll_s=0.02):
    """Peak occupancy while ``fn`` runs, and what stays resident afterwards.

    Polled rather than read from a counter, for the same reason as
    :func:`occupancy_mb`: the largest term may be allocated outside PyTorch.
    """
    # Seeded with the reading before the run and clamped with the one after,
    # so the peak can never come back below what is demonstrably resident. A
    # thread alone can miss a late allocation: Adam creates its moment buffers
    # inside the first step(), at the very end of a training iteration, and a
    # short run can finish between two polls -- which returned a peak 20 MB
    # BELOW the resident figure for 3D training.
    peak = occupancy_mb(device)
    stop = threading.Event()

    def watch():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, occupancy_mb(device))
            time.sleep(poll_s)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        fn()
    finally:
        stop.set()
        thread.join()
    peak = max(peak, occupancy_mb(device))
    torch.cuda.empty_cache()
    return peak, occupancy_mb(device)


# --------------------------------------------------------------------------
# the denoiser's own two measurements
# --------------------------------------------------------------------------


def measure_optimizer_mb(model, ndim, C, train, device="cuda", call_fn=None):
    """The denoiser's gradients and optimizer state, as a delta.

    Excludes the weights: the probe runs the real solver, so they are already
    resident and counted there. What the probe does *not* contain is gradients
    and optimizer state, because its denoiser is an identity, so no gradient
    reaches the parameters and Adam skips them for want of one.

    Must run before :func:`measure_tile_mb`, which is why :func:`rank` calls it
    first. It leaves ``.grad`` and the Adam moments resident, so the tile
    measurement's baseline already holds them and does not charge them twice.
    """
    call_fn = call_fn or (lambda m, x: m(x, 0.05))
    torch.cuda.empty_cache()
    weights_only = torch.cuda.memory_allocated(device)
    if not train:
        return 0.0

    # The tiny call is also the warm-up that puts cuDNN workspace in place
    # before any tile is measured.
    x = torch.randn(1, C, *(32,) * ndim, device=device, requires_grad=True)
    call_fn(model, x).sum().backward()

    # The optimizer has to outlive its own step(). Adam allocates its two moment
    # buffers lazily inside step(), so a temporary `Adam(...).step()` is
    # collected on the next line and the moments are freed before the reading --
    # which returned 1x the weights instead of 3x, under-counting training by
    # 249 MB in 2D and 772 MB in 3D.
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.step()

    del x
    torch.cuda.empty_cache()
    measured = (torch.cuda.memory_allocated(device) - weights_only) / MB
    del optimizer
    torch.cuda.empty_cache()
    return measured


def measure_tile_mb(model, window_spatial, C, train, device="cuda", call_fn=None):
    """What ONE tile costs, at the window the candidate really uses.

    Measured at the exact window, never extrapolated from a smaller one.
    Returns None when a single tile does not fit, which is that candidate's
    answer and costs about a second to obtain.
    """
    call_fn = call_fn or (lambda m, x: m(x, 0.05))
    base = torch.cuda.memory_allocated(device)
    x = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(1, C, *window_spatial, device=device, requires_grad=train)
        if train:
            call_fn(model, x).sum().backward()
        else:
            with torch.no_grad():
                call_fn(model, x)
        return (torch.cuda.max_memory_allocated(device) - base) / MB
    except torch.OutOfMemoryError:
        return None
    finally:
        del x
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# the sum
# --------------------------------------------------------------------------


def peak_mb(
    probe_peak,
    probe_resident,
    optimizer_mb,
    tile_mb,
    k,
    window_px,
    b,
    ckpt,
    n_calls,
    train,
    signal_px=0,
    padded_px=0,
):
    """What the run will peak at, from the probe's two readings and the tiles.

    ``probe_peak`` and ``probe_resident`` come from one run of the real config
    with the denoiser replaced by an identity: the peak includes the physics
    burst, the resident figure is what survives it.

    ``signal_px``, ``padded_px`` and ``window_px`` count elements *including*
    the channel axis. Counting spatial elements only under-predicted every 2D
    case by exactly the channel count and left 3D untouched, which is how the
    error stayed hidden until a 3-channel case was decomposed.
    """
    # How many tiles are alive at once. 'always' checkpoints each batch, so
    # exactly one batch survives the backward's recompute; 'never' keeps every
    # tile of every call in the graph. n_calls is how many times the tiled
    # denoiser runs in one step, counted by the probe's identity denoiser
    # rather than inferred -- the calls may come from the loss rather than from
    # the model's forward. 
    count = (n_calls * k) if (train and ckpt == "never") else b

    # The tile copies that sit outside the graph in every regime. This
    # double-counts the batch in flight, which is the safe direction.
    kept = (
        KEPT_TENSORS_PER_TILE
        * (n_calls if train else 1)
        * k
        * window_px
        * BYTES_PER_FLOAT
        / MB
    )

    # _apply_op's own two full-size tensors: the reflect-padded copy of the
    # signal, and the zeroed out_local it reduces into. The probe never sees
    # them, because its denoiser is an identity and so never tiles or pads.
    tiler = (signal_px + padded_px) * BYTES_PER_FLOAT / MB

    denoiser_side = optimizer_mb + tiler + kept + count * tile_mb

    # Inference frees the gradient step's temporaries before the denoiser
    # allocates, so the two bursts never coexist and the peak is whichever is
    # larger. Training retains the physics subgraph across the denoiser call,
    # so there they really do add.
    if train:
        return probe_peak + denoiser_side
    return max(probe_peak, probe_resident + denoiser_side)
