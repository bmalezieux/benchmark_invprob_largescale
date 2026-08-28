"""CPU-only tests for the tiling-parameter advisor.

Memory is injected through ``tile_mb_table`` and seam results are constructed
directly, so nothing here needs a GPU -- but the advisor takes its tile geometry
from the real tiler, so it needs a deepinv with the distributed tiling API. CI
pins an older one, hence the skip.
"""

import math
import textwrap

import pytest

pytest.importorskip(
    "deepinv.distributed.strategies.utils",
    reason="needs a deepinv with the distributed tiling API",
)

from toolsbench.autotune import (  # noqa: E402
    Candidate,
    SeamResult,
    default_patch_grid,
    geometry,
    peak_mb,
    rank,
    resolve_overlap,
    saturating_overlap,
    top3,
)
from toolsbench.autotune import seam as seam_mod  # noqa: E402
from toolsbench.autotune.config import (  # noqa: E402
    expand_params,
    parse_config,
)

IMG_2D = (1, 3, 4096, 4096)
IMG_3D = (1, 1, 512, 512, 512)

TILE_2D_INFER = {
    96: 21.3,
    112: 27.0,
    128: 32.1,
    192: 73.0,
    256: 128.1,
    320: 203.2,
    416: 338.2,
    544: 579.0,
    800: 1250.1,
    1056: 2179.6,
    1568: 4802.6,
}


def fit(**kwargs):
    """rank() with the arguments every call needs, so tests state only the point."""
    base = dict(
        img_size=IMG_2D,
        world_size=8,
        gpu_mem_gb=40,
        overlap=16,
        optimizer_mb=0.0,
        tile_mb_table=TILE_2D_INFER,
    )
    return rank(**{**base, **kwargs})


# --- geometry: exact, and it is the tiler's answer, not a formula ------------


def test_geometry_matches_the_real_tiler():
    assert geometry(IMG_2D, 1024, 16)[0] == 16
    assert geometry(IMG_2D, 512, 16)[0] == 64
    assert geometry(IMG_2D, 384, 16)[0] == 121
    n_tiles, window = geometry(IMG_3D, 128, 16)
    assert n_tiles == 64
    assert window == (1, 1, 160, 160, 160)


def test_patch_grid_stays_below_the_image():
    # the tiler raises for patch_size >= dimension
    assert all(p < 512 for p in default_patch_grid(IMG_3D))
    assert 512 not in default_patch_grid(IMG_3D)


@pytest.mark.parametrize("p, expected", [(64, 3.375), (128, 1.953), (256, 1.424)])
def test_redundancy_matches_halo_when_patch_divides_the_image(p, expected):
    n_tiles, window = geometry(IMG_3D, p, 16)
    redundancy = n_tiles * math.prod(window[2:]) / 512**3
    assert redundancy == pytest.approx(expected, rel=1e-3)
    assert redundancy == pytest.approx(((p + 32) / p) ** 3, rel=1e-3)


@pytest.mark.parametrize(
    "p, halo_only, real", [(192, 1.587, 2.262), (384, 1.271, 4.291)]
)
def test_redundancy_exceeds_halo_when_patch_does_not_divide(p, halo_only, real):
    """The tiler pins the last tile to D - p, so tiles overlap and compute twice."""
    n_tiles, window = geometry(IMG_3D, p, 16)
    redundancy = n_tiles * math.prod(window[2:]) / 512**3
    assert ((p + 32) / p) ** 3 == pytest.approx(halo_only, rel=1e-3)
    assert redundancy == pytest.approx(real, rel=1e-3)
    assert redundancy > ((p + 32) / p) ** 3


# --- stage A: the seam floor ------------------------------------------------

SEAM = SeamResult(
    arch="drunet",
    ndim=2,
    channels=3,
    test_size=2048,
    sigma=0.05,
    floors={64: 64, 128: 64, 256: 32, 512: 32},
)


def test_overlap_must_be_given_or_measured():
    """No hardcoded default: the old one was blind to the architecture."""
    assert resolve_overlap(512, overlap=32) == (32, False)
    with pytest.raises(ValueError, match="no overlap given"):
        resolve_overlap(512)


def test_floor_lookup_between_and_above_measured_patches():
    assert SEAM.floor_for(256) == (32, False)  # measured exactly
    assert SEAM.floor_for(384) == (32, False)  # largest tested below
    assert SEAM.floor_for(2048) == (32, True)  # above all, flagged
    with pytest.raises(ValueError, match="below the smallest"):
        SEAM.floor_for(32)  # would under-halo


def test_saturation_excludes_zero_overlap():
    """r=0 pads nothing, so its borders match the reference unfairly."""
    row = {0: 53.20, 4: 51.01, 8: 53.63, 16: 53.68, 32: 55.28, 64: 56.53}
    assert saturating_overlap(row, 0.5) == 64
    assert saturating_overlap(row, 1.5) == 32


def test_cache_key_includes_sigma():
    """The floor tracks the denoising strength, not the noise in the image."""
    other = SeamResult(
        arch="drunet", ndim=2, channels=3, test_size=2048, sigma=0.005, floors={64: 8}
    )
    assert SEAM.key != other.key
    assert "s0.05" in SEAM.key and "s0.005" in other.key


def test_seam_patches_halve_the_test_volume():
    assert seam_mod.test_patch_sizes(2048, 2) == [64, 128, 256, 512, 1024]
    assert seam_mod.test_patch_sizes(160, 3) == [40, 80]  # 3D stops on tile budget
    assert all(2048 % p == 0 for p in seam_mod.test_patch_sizes(2048, 2))


# --- stage B: counting ------------------------------------------------------


def test_count_is_b_under_always_and_calls_times_k_under_never():
    kw = dict(
        optimizer_mb=0.0,
        probe_peak=0.0,
        probe_resident=0.0,
        tile_mb=100.0,
        k=16,
        window_px=0,
        n_calls=5,
        train=True,
    )
    assert peak_mb(b=2, ckpt="always", **kw) == pytest.approx(200.0)
    assert peak_mb(b=2, ckpt="never", **kw) == pytest.approx(5 * 16 * 100.0)


def test_inference_ignores_the_call_count():
    kw = dict(
        optimizer_mb=0.0,
        probe_peak=0.0,
        probe_resident=0.0,
        tile_mb=100.0,
        k=16,
        window_px=0,
        b=2,
        ckpt=None,
        train=False,
    )
    assert peak_mb(n_calls=1, **kw) == peak_mb(n_calls=5, **kw)


def test_tiler_buffers_are_counted():
    """_apply_op's padded copy and out_local: no measurement sees them, and
    omitting them under-predicted 3D inference by 8.1%."""
    kw = dict(
        optimizer_mb=0.0,
        probe_peak=0.0,
        probe_resident=0.0,
        tile_mb=0.0,
        k=1,
        window_px=0,
        b=1,
        ckpt=None,
        n_calls=1,
        train=False,
    )
    assert peak_mb(**kw) == 0.0
    assert peak_mb(signal_px=384**3, padded_px=416**3, **kw) == pytest.approx(
        490.6, rel=1e-3
    )


def test_inference_takes_the_larger_burst_and_training_adds_them():
    """The physics burst and the denoiser burst coexist only under autograd.

    Inference frees the gradient step's temporaries before prox allocates, so
    the peak is whichever is larger; training retains the physics subgraph
    across the denoiser call, so there they add.
    """
    kw = dict(
        optimizer_mb=0.0,
        tile_mb=100.0,
        k=1,
        window_px=0,
        b=1,
        n_calls=1,
        probe_peak=1000.0,
        probe_resident=200.0,
    )
    # denoiser side is 100; the probe's own peak dominates
    assert peak_mb(ckpt=None, train=False, **kw) == pytest.approx(1000.0)
    # ... until the denoiser side clears it
    assert peak_mb(ckpt=None, train=False, **{**kw, "tile_mb": 900.0}) == (
        pytest.approx(1100.0)
    )
    # training adds, from the probe's peak rather than its resident figure
    assert peak_mb(ckpt="always", train=True, **kw) == pytest.approx(1100.0)


# --- stage B: ranking -------------------------------------------------------


def test_rank_rejects_more_ranks_than_tiles():
    # p=1536 on 4096 gives 9 tiles, so it drops out at world_size=16
    cands = fit(world_size=16)
    assert all(c.n_tiles >= 16 for c in cands)
    assert 1536 not in {c.patch_size for c in cands}


def test_rank_skips_patch_sizes_whose_tile_does_not_fit():
    partial = {k: v for k, v in TILE_2D_INFER.items() if k != 1056}
    assert 1024 not in {c.patch_size for c in fit(tile_mb_table=partial)}


def test_rank_honours_the_budget_and_orders_by_work():
    cands = fit(optimizer_mb=124.5, probe_peak=1536.0, probe_resident=1536.0)
    assert cands, "expected feasible candidates"
    assert all(c.peak_mb <= 40 * 1024 * 0.90 for c in cands)
    assert [c.work for c in cands] == sorted(c.work for c in cands)
    assert cands[0].patch_size == 1024


def test_tight_budget_lowers_the_batch_rather_than_dropping_candidates():
    roomy = fit()
    tight = fit(probe_peak=28000.0, probe_resident=28000.0)
    assert {c.patch_size for c in tight} <= {c.patch_size for c in roomy}
    by_p = {c.patch_size: c.max_batch_size for c in roomy}
    assert any(c.max_batch_size < by_p[c.patch_size] for c in tight)


def test_rank_takes_the_overlap_from_a_seam_result():
    cands = fit(overlap=None, seam=SEAM)
    assert {c.overlap for c in cands} <= {32, 64}


def test_top3_returns_distinct_patch_sizes():
    picks = top3(fit(optimizer_mb=124.5, probe_peak=1536.0, probe_resident=1536.0))
    assert len(picks) == 3
    assert len({c.patch_size for c in picks}) == 3


def test_yaml_row_drops_checkpoint_for_inference():
    c = Candidate(
        patch_size=512,
        overlap=16,
        max_batch_size=8,
        checkpoint_batches=None,
        n_tiles=64,
        k=8,
        window=(1, 3, 544, 544),
        redundancy=1.13,
        work=1,
        tile_mb=579.0,
        peak_mb=6000.0,
    )
    assert c.as_yaml_row().startswith("[  512,  16,   8],")
    assert "'" not in c.as_yaml_row().split("#")[0]


# --- reading configs --------------------------------------------------------

INFERENCE_2D = "benchmark_inference/configs/experiments/comm_inference_2D.yml"
INFERENCE_3D = "benchmark_inference/configs/experiments/comm_inference_3D.yml"
TRAINING = "benchmark_training/configs/experiments/strong_scaling.yml"


def test_comma_joined_keys_are_split_positionally():
    """benchopt writes tuple sweeps as one comma-joined key."""
    flat = expand_params({"a, b": [[1, 2], [3, 4]], "c": [5]})
    assert flat == {"a": [1, 3], "b": [2, 4], "c": [5]}


def test_inference_config_gives_one_case():
    (case,) = parse_config(INFERENCE_2D)
    assert case.solver == "PnP"
    assert case.denoiser == "drunet"
    assert case.img_size == (1, 3, 8192, 8192)
    assert case.train is False
    assert (case.config_patch, case.config_overlap) == (512, 32)


def test_three_dimensional_shape_is_not_a_sweep():
    """`image_size: [[512,512,512]]` is one volume, not three sizes."""
    (case,) = parse_config(INFERENCE_3D)
    assert case.ndim == 3
    assert case.img_size == (1, 1, 512, 512, 512)


def test_training_config_sweeps_image_size_inside_the_solver_block():
    cases = parse_config(TRAINING)
    assert {c.img_size[-1] for c in cases} == {1024, 4096, 8192}
    assert all(c.train for c in cases)
    assert all(c.n_iter == 5 for c in cases)


def test_gpu_count_is_never_read_from_the_config():
    """Hardware is a property of where you run, not of the experiment."""
    for case in parse_config(TRAINING):
        assert not hasattr(case, "world_size")
        assert not hasattr(case, "gpu_mem_gb")


def test_missing_image_size_says_what_to_do(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text(textwrap.dedent("""
        dataset:
          - walnut:
              channels: 1
        solver:
          - PnP:
              denoiser: drunet
    """))
    with pytest.raises(ValueError, match="pass --image explicitly"):
        parse_config(path)


def test_overrides_win_over_the_config():
    (case,) = parse_config(INFERENCE_2D, channels=1, image_size=2048)
    assert case.img_size == (1, 1, 2048, 2048)
