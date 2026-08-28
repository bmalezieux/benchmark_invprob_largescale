"""Enumerate tiling candidates, keep the ones that fit, order them by work.

Every megabyte figure comes from :mod:`toolsbench.autotune.memory`; nothing
here is estimated. Tile counts and windows come from the real tiler, so the
geometry matches what the solver will actually build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deepinv.distributed.strategies.utils import tiling_splitting_strategy

from toolsbench.autotune.memory import (
    measure_optimizer_mb,
    measure_tile_mb,
    peak_mb,
)

PATCH_GRID = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


# --------------------------------------------------------------------------
# geometry — exact, no GPU
# --------------------------------------------------------------------------


def geometry(img_size, p, r):
    """Tile count and window shape, from the real tiler.

    Every spatial axis is tiled, matching what the solvers hardcode.
    """
    # imported lazily: keeps the module importable without deepinv, and avoids
    # an import-order cycle in deepinv's package __init__.

    ndim = len(img_size) - 2
    dims = tuple(range(-ndim, 0))
    slices, _ = tiling_splitting_strategy(
        img_size, patch_size=p, overlap=r, tiling_dims=dims
    )
    window = list(img_size)
    for d in dims:
        window[d] = p + 2 * r
    return len(slices), tuple(window)


def default_patch_grid(img_size):
    return [p for p in PATCH_GRID if p < min(img_size[2:])]


def resolve_overlap(p: int, overlap=None, seam=None) -> tuple[int, bool]:
    """The overlap to use at patch size ``p``, and whether it is extrapolated.

    There is no hardcoded default. Overlap is a property of the network, so it
    comes either from the caller or from a seam measurement of that network. A
    default keyed on ``(p, ndim)`` alone would hand every architecture DRUNet's
    floor, and a wrong overlap does not raise -- it returns a seamed image.
    """
    if overlap is not None:
        return overlap, False
    if seam is None:
        raise ValueError(
            "no overlap given and no seam measurement to take one from. Pass "
            "overlap= explicitly, or run seam_floors() for this denoiser: a "
            "wrong overlap does not raise, it returns a seamed image."
        )
    return seam.floor_for(p)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    patch_size: int
    overlap: int
    max_batch_size: int
    checkpoint_batches: str | None
    n_tiles: int
    k: int
    window: tuple[int, ...]
    redundancy: float
    work: int
    tile_mb: float
    peak_mb: float

    def as_yaml_row(self) -> str:
        ckpt = (
            "" if self.checkpoint_batches is None else f", '{self.checkpoint_batches}'"
        )
        return (
            f"[{self.patch_size:>5}, {self.overlap:>3}, {self.max_batch_size:>3}"
            f"{ckpt}],  # N={self.n_tiles} k={self.k} "
            f"redundancy={self.redundancy:.2f} mem={self.peak_mb / 1024:.1f}GB"
        )


def rank(
    img_size,
    world_size,
    gpu_mem_gb,
    overlap=None,
    *,
    train=False,
    n_calls=1,
    seam=None,
    model_obj=None,
    call_fn=None,
    optimizer_mb=None,
    probe_peak=0.0,
    probe_resident=0.0,
    margin=0.90,
    device="cuda",
    tile_mb_table=None,
) -> list[Candidate]:
    """Candidates that fit, cheapest per-rank work first.

    ``probe_peak`` and ``probe_resident`` come from one identity-denoiser run of
    the real config, so the physics, the loss, the CUDA context and the
    solver's own buffers are all measured rather than modelled. See
    :func:`toolsbench.autotune.probe.identity_run`.

    ``checkpoint_batches`` is chosen by preferring ``'never'`` wherever it fits,
    since checkpointing buys memory by recomputing. That is settled within a
    patch size, not across: comparing modes at different patch sizes would need
    a forward/backward time ratio, which depends on the GPU.

    ``margin`` covers only what a single emulated rank cannot see: NCCL buffers
    and allocator fragmentation.

    ``tile_mb_table`` maps a window side to a measured MB, for reproducing a
    run without a GPU; when absent the tiles are measured on ``device``.
    """
    ndim = len(img_size) - 2
    C = img_size[1]

    # Step 1: the optimizer delta, once for the whole sweep. It must precede
    # every tile
    # measurement, because it leaves .grad and the Adam moments resident and
    # so keeps them out of the tile readings.
    if optimizer_mb is None:
        optimizer_mb = measure_optimizer_mb(
            model_obj, ndim, C, train, device=device, call_fn=call_fn
        )
    budget = gpu_mem_gb * 1024.0 * margin
    signal_px = math.prod(img_size[1:])

    # Step 2: geometry for every patch size, ordered by per-rank work. Work is
    # exact arithmetic, so the cheapest candidates get their tile measured
    # first and the ranking never depends on a memory reading.
    plan = []
    skipped = []
    for p in default_patch_grid(img_size):
        try:
            r, _ = resolve_overlap(p, overlap, seam)
        except ValueError:
            # No seam measurement reaches this patch size. Drop the candidate
            # rather than the whole run: a guessed overlap does not raise, it
            # returns a seamed image.
            skipped.append(p)
            continue
        n_tiles, window = geometry(img_size, p, r)
        if n_tiles < world_size:
            continue  # some ranks would sit idle
        k = math.ceil(n_tiles / world_size)  # round-robin: the busiest rank
        px = math.prod(window[1:])
        padded_px = C * math.prod([d + 2 * r for d in img_size[2:]])
        plan.append((k * px, p, r, n_tiles, k, window, px, padded_px))
    plan.sort()

    # Step 3: measure one tile per patch size, then search (ckpt, b) for the
    # largest batch that still fits. A larger batch means fewer kernel
    # launches, so the search runs downwards and stops at the first fit.
    out: list[Candidate] = []
    for work, p, r, n_tiles, k, window, px, padded_px in plan:
        if tile_mb_table is not None:
            tile = tile_mb_table.get(window[-1])
        else:
            tile = measure_tile_mb(
                model_obj, window[2:], C, train, device=device, call_fn=call_fn
            )
        if tile is None:
            continue  # one tile does not fit: dead
        for ckpt in (("always", "never") if train else (None,)):
            for b in range(k, 0, -1):
                m = peak_mb(
                    probe_peak,
                    probe_resident,
                    optimizer_mb,
                    tile,
                    k,
                    px,
                    b,
                    ckpt,
                    n_calls,
                    train,
                    signal_px=signal_px,
                    padded_px=padded_px,
                )
                if m > budget:
                    if train and ckpt == "never":
                        break  # b does not help here
                    continue
                out.append(
                    Candidate(
                        patch_size=p,
                        overlap=r,
                        max_batch_size=b,
                        checkpoint_batches=ckpt,
                        n_tiles=n_tiles,
                        k=k,
                        window=window,
                        # from the tiler, so it counts the tile overlap that
                        # (w/p)**ndim misses whenever p does not divide the image
                        redundancy=n_tiles * px / signal_px,
                        work=work,
                        tile_mb=tile,
                        peak_mb=m,
                    )
                )
                break

    if skipped and not out:
        raise ValueError(
            f"no candidate has a known overlap floor: patch sizes {skipped} are "
            f"all below the smallest seam-tested patch. Seam-test a smaller "
            f"patch, or pass overlap= explicitly."
        )
    # 'always' does everything 'never' does plus one extra forward per tile --
    # that is what torch.utils.checkpoint recomputes -- so 'never' is strictly
    # fewer passes and leads wherever it fits. False sorts before True. Only the
    # ratio between the modes is device-specific, and this never needs it.
    out.sort(key=lambda c: (c.checkpoint_batches == "always", c.work, c.peak_mb))
    return out


def top3(cands: list[Candidate]) -> list[Candidate]:
    """The three cheapest candidates with distinct patch sizes.

    No hedging: the memory figures are measured, so there is no error bar to
    spread over.
    """
    picks, seen = [], set()
    for c in cands:
        if c.patch_size in seen:
            continue
        picks.append(c)
        seen.add(c.patch_size)
        if len(picks) == 3:
            break
    return picks
