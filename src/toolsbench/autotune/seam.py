"""Stage A: how much halo a tile needs before tiling matches an untiled pass.

Runs on a small volume -- the largest that fits *untiled*, since the untiled
output is the reference. The answer transfers to any image size, because it is
a property of the network's receptive field and its denoising sigma.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from deepinv.distributed import DistributedContext, distribute

from toolsbench.data.base import DataConfig
from toolsbench.data.synthetic import SyntheticData

#: Candidate test volumes, largest first. The reference must fit untiled, and
#: `largest_untiled` walks down until one does -- so a rung too big for the
#: calibration GPU costs one failed attempt, not a wrong answer.
#:
#: The top rung matters beyond headroom: `test_patch_sizes` halves the test
#: volume, so it sets the largest patch size that gets measured at all. Topping
#: out at 2048 left PATCH_GRID's 1536 and 2048 borrowing 1024's floor, on the
#: very machines big enough to want them.
SEAM_LADDER = {
    2: (4096, 3072, 2048, 1536, 1024, 768, 512, 256),
    3: (256, 192, 160, 128, 96),
}
#: Overlaps to try. 0 is the control -- it shows what unmitigated seams cost.
OVERLAP_GRID = (0, 4, 8, 16, 32, 64)
#: An overlap is "enough" once it is within this of the best value in its row.
#: Tight, because the whole-image metric compresses the useful range: its curve
#: flattens onto a ceiling set by the image border rather than by the seams.
SATURATION_DB = 0.1
#: Sigma the seam test denoises at, when a config does not name one. The floor
#: tracks this and not the noise in the image, so it matters: measured at
#: 2048^2, sigma 0.05 gave floors of 64 at every patch size and sigma 0.005 gave
#: 32/16, a factor of four.
SEAM_SIGMA = 0.01
#: Tile budget for the seam sweep. 2D tiles are cheap even in quantity; 3D ones
#: are not, and the count grows as 8^k rather than 4^k.
MAX_TEST_TILES = {2: 1024, 3: 512}

CACHE_PATH = (
    Path(os.environ.get("TOOLSBENCH_CACHE", Path.home() / ".cache" / "toolsbench"))
    / "seam_floors.json"
)


@dataclass(frozen=True)
class SeamResult:
    """What phase A produces, and everything needed to judge it."""

    arch: str
    ndim: int
    channels: int
    test_size: int
    sigma: float
    floors: dict[int, int]  # patch size -> measured floor
    curve: dict[int, dict[int, float]] = field(default_factory=dict)
    reference_gain_db: float = 0.0  # denoise gain, measured separately
    provisional: bool = False  # set when the denoiser degrades its input

    @property
    def key(self) -> str:
        return f"{self.arch}-{self.ndim}d-c{self.channels}-s{self.sigma:g}"

    def floor_for(self, p: int) -> tuple[int, bool]:
        """The floor to use at patch size `p`, and whether it is extrapolated.

        Test patches come from halving the test volume, so they do not line up
        with the production grid. Between measured points, take the floor of the
        largest tested patch at or below `p`: seam density falls as 1/p, so a
        smaller patch's floor is an upper bound for a larger one. Above the
        largest measured, reuse it and say so. Below the smallest, refuse --
        that extrapolates in the direction that costs correctness.
        """
        if p in self.floors:
            return self.floors[p], False
        measured = sorted(self.floors)
        below = [q for q in measured if q < p]
        if below:
            return self.floors[below[-1]], p > measured[-1]
        raise ValueError(
            f"patch size {p} is below the smallest seam-tested patch "
            f"({measured[0]}); smaller patches need more halo, not less, so "
            f"this cannot be extrapolated safely. Pass overlap= explicitly."
        )

    def suggestions(self, thresholds=(0.1, 0.5, 1.5)) -> dict[float, dict[int, int]]:
        """Floors at several thresholds, so the choice is visible rather than made.

        The seam test does not have one right answer. 0.1 dB is the baseline --
        tight, because the whole-image metric flattens onto a ceiling set by the
        image border and compresses the useful range. Looser thresholds trade a
        little agreement for a lot of compute: halving the overlap at p=512
        saves 19% of the work in 2D and 27% in 3D.
        """
        return {
            db: {p: saturating_overlap(row, db) for p, row in self.curve.items()}
            for db in thresholds
        }

    def table(self, thresholds=(0.1, 0.5, 1.5)) -> str:
        """The measured curve and what it suggests, so a reader can choose."""
        overlaps = sorted({r for row in self.curve.values() for r in row})
        header = "  ".join(f"r={r:<4}" for r in overlaps)
        lines = [
            f"  seam test: {self.arch} {self.ndim}D, {self.channels}ch, "
            f"sigma={self.sigma:g}, on {self.test_size}^{self.ndim}",
            "  PSNR(tiled, untiled) over the whole volume",
            "",
            f"{'patch':>7}  {header}",
        ]
        for patch in sorted(self.curve):
            row = self.curve[patch]
            cells = "  ".join(
                f"{row[r]:6.2f}" if r in row else "   n/a" for r in overlaps
            )
            lines.append(f"{patch:>7}  {cells}")
        lines.append("")
        for db, floors in self.suggestions(thresholds).items():
            picks = ", ".join(f"p={p}: {r}" for p, r in sorted(floors.items()))
            lines.append(f"  within {db:g} dB of the plateau -> {picks}")
        lines.append(
            "  r=0 is shown as the control and excluded from the suggestion "
            "(no padding, so its borders match the reference unfairly)"
        )
        gain = self.reference_gain_db
        lines.append(
            f"  denoiser health: {gain:+.1f} dB on synthetic content"
            + ("   PROVISIONAL: it degrades its input" if self.provisional else "")
        )
        return "\n".join(lines)


def psnr(a, b) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def seam_input(size: int, ndim: int, channels: int, device="cuda", seed=0):
    """Uniform random content: broadband at every resolution."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(1, channels, *([size] * ndim), generator=generator).to(device)


def largest_untiled(
    model, ndim, channels, device="cuda", sigma=SEAM_SIGMA, ladder=None, call_fn=None
):
    """Biggest test volume whose *untiled* denoise fits, with its reference.

    Walks the ladder down and keeps the first rung that runs. This is the one
    hard limit on the seam test: the reference has to be computed in one pass,
    so the test volume is bounded by the calibration GPU even though the answer
    is not.
    """
    call_fn = call_fn or (lambda m, x: m(x, sigma))
    for size in ladder or SEAM_LADDER[ndim]:
        signal = reference = None
        try:
            signal = seam_input(size, ndim, channels, device)
            with torch.no_grad():
                reference = call_fn(model, signal).float()
            return size, signal, reference
        except torch.OutOfMemoryError:
            del signal, reference
            torch.cuda.empty_cache()
    raise RuntimeError(
        "no test volume fits untiled on this device; the seam test needs the "
        "whole reference in one pass"
    )


def test_patch_sizes(
    size: int, ndim: int, max_k: int = 5, min_patch: int = 32, max_tiles=None
) -> list[int]:
    """Patch sizes to seam-test on a volume of side `size`: size / 2^k.

    Halving keeps every test tiling exact -- the patch always divides the
    volume, so no tile overlaps another and the measured error is seam error
    rather than the redundancy a ragged fit would add. It also fixes the
    boundary positions, so the scored band is the same at every overlap. And it
    spans seam densities geometrically: 2^k tiles per axis.

    Two guards. Patches below `min_patch` stop being informative, since the halo
    then dwarfs the patch. And the tile count explodes in 3D -- on 160^3, k=5
    would mean 32768 tiles of a 69^3 window, some 2600x the volume in compute for
    a meaningless measurement. In practice this yields five patch sizes in 2D and
    two in 3D.
    """
    budget = max_tiles or MAX_TEST_TILES.get(ndim, 512)
    out = []
    for k in range(1, max_k + 1):
        patch = size // (2**k)
        if patch < min_patch or (2**k) ** ndim > budget:
            break
        out.append(patch)
    return sorted(out)


def saturating_overlap(row: dict[int, float], saturation_db=SATURATION_DB) -> int:
    """Smallest overlap within `saturation_db` of the best value in the row.

    Overlap 0 is excluded from the candidates, though it stays in the table as
    the control that shows what unmitigated seams cost. It is not comparable
    with the others: at zero overlap the tiler applies no global padding at all,
    so its image borders match the untiled reference exactly and it collects a
    bonus no other overlap can earn. Left in, it won the largest-patch row in
    all six noise settings measured and drove the suggestion to 0.

    The best value is the row's maximum, not its last entry: the band curve is
    monotone in practice, but a shallow tail plus measurement noise can still
    put the maximum one step short of the end.
    """
    candidates = {r: v for r, v in row.items() if r > 0}
    if not candidates:
        raise ValueError("no overlap above zero was measured")
    best = max(candidates.values())
    return min(r for r, value in candidates.items() if value >= best - saturation_db)


def denoise_gain_db(
    model, ndim, channels, device="cuda", sigma=SEAM_SIGMA, call_fn=None, size=None
) -> float:
    """Does this denoiser actually denoise? Reported beside every seam result.

    Measured on the benchmark's own synthetic signal, at a small size, because
    denoising quality cannot be judged on the random content the seam sweep
    uses -- there is nothing there to recover. A value at or below zero means
    the network degrades its input, which makes the seam floor provisional: the
    3D DRUNet in this benchmark scores about -12 dB, its weights being 2D
    kernels inflated to 3D.
    """
    call_fn = call_fn or (lambda m, x: m(x, sigma))
    size = size or (256 if ndim == 2 else 96)
    clean = SyntheticData().get_data(
        DataConfig(
            size=tuple([size] * ndim),
            batch_size=1,
            channels=channels,
            data_type=torch.float32,
            device=device,
        )
    )["data"]
    noisy = (clean + sigma * torch.randn_like(clean)).clamp(0, 1)
    with torch.no_grad():
        out = call_fn(model, noisy).float()
    gain = psnr(out.cpu(), clean.cpu()) - psnr(noisy.cpu(), clean.cpu())
    del clean, noisy, out
    torch.cuda.empty_cache()
    return gain


def seam_floors(
    model,
    arch,
    ndim,
    channels,
    device="cuda",
    sigma=SEAM_SIGMA,
    saturation_db=SATURATION_DB,
    call_fn=None,
) -> SeamResult:
    """Measure the overlap floor: phase A, start to finish.

    Compares the tiled output against the untiled one -- *not* against the clean
    signal. This is a test of tiling fidelity, not of denoising quality, which
    is why it stays meaningful even when the denoiser is poor. It does check the
    denoiser separately and marks the result provisional if it degrades its own
    input.
    """
    call_fn = call_fn or (lambda m, x: m(x, sigma))

    # Step 1: the reference. The largest volume that still denoises untiled,
    # since the untiled output is what everything is compared against.
    size, signal, reference = largest_untiled(
        model, ndim, channels, device, sigma, call_fn=call_fn
    )

    # Step 2: check the denoiser itself, separately, so a network that
    # degrades its input is flagged rather than silently producing a floor.
    gain = denoise_gain_db(model, ndim, channels, device, sigma, call_fn)

    # Step 3: patch sizes to sweep -- halvings of the test volume, stopping
    # when the tiles get too small or too numerous.
    patches = test_patch_sizes(size, ndim)
    if not patches:
        raise RuntimeError(
            f"test volume {size}^{ndim} admits no usable patch size; the seam "
            f"test needs at least two tiles per axis"
        )

    # Step 4: for every (patch, overlap) pair, tile the same signal and score
    # the result against the untiled reference. A pair that will not fit is
    # dropped rather than fatal.
    curve: dict[int, dict[int, float]] = {}
    with DistributedContext(device_mode="gpu") as ctx:
        for p in patches:
            row: dict[int, float] = {}
            for r in OVERLAP_GRID:
                torch.cuda.empty_cache()
                try:
                    tiled = distribute(
                        model,
                        ctx,
                        patch_size=p,
                        overlap=r,
                        tiling_dims=tuple(range(-ndim, 0)),
                        max_batch_size=1,
                        type_object="denoiser",
                    )
                    with torch.no_grad():
                        out = call_fn(tiled, signal).float()
                    row[r] = psnr(out, reference)
                    del tiled, out
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if row:
                curve[p] = row

    # Step 5: per patch size, the smallest overlap within saturation_db of the
    # best value in its row.
    floors = {p: saturating_overlap(row, saturation_db) for p, row in curve.items()}
    return SeamResult(
        arch=arch,
        ndim=ndim,
        channels=channels,
        test_size=size,
        sigma=sigma,
        floors=floors,
        curve=curve,
        reference_gain_db=gain,
        provisional=gain <= 0.0,
    )


def load_cache(path: Path = CACHE_PATH) -> dict[str, SeamResult]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for key, entry in raw.items():
        entry["floors"] = {int(k): v for k, v in entry["floors"].items()}
        entry["curve"] = {
            int(k): {int(rr): vv for rr, vv in v.items()}
            for k, v in entry.get("curve", {}).items()
        }
        out[key] = SeamResult(**entry)
    return out


def save_cache(result: SeamResult, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(path.read_text()) if path.exists() else {}
    raw[result.key] = {
        "arch": result.arch,
        "ndim": result.ndim,
        "channels": result.channels,
        "test_size": result.test_size,
        "sigma": result.sigma,
        "floors": result.floors,
        "curve": result.curve,
        "reference_gain_db": result.reference_gain_db,
        "provisional": result.provisional,
    }
    path.write_text(json.dumps(raw, indent=2, sort_keys=True))
