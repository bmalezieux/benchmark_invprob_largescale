"""Physics, losses, model and metric for the synthetic cryo-ET (``tomo_ei``) case.

The problem itself is tested in ``test_invprob.py`` (``TestCryoEIInvProb``) and
the solver in ``test_solver.py`` (``TestTomoEISolverMixedPrecision``), with the
rest of their kind.

Everything here runs on CPU with the pure-torch tomography backend: astra ships
CUDA kernels, so the torch operator is what makes this path testable at all.
Nothing calls ``distribute`` and nothing runs a training step — both need a
live process group to mean anything, and the distributed paths are covered by
the benchmark runs instead.
"""

import pytest
import torch
from deepinv.distributed import DistributedContext

from toolsbench.invprob import CryoEIInvProb
from toolsbench.invprob.base import InvProbConfig
from toolsbench.utils.cryo import (
    GpuFSC,
    build_cryo_pair,
    build_unet3d,
    fsc_shell,
    obs_loss,
    resolve_num_operators,
)
from toolsbench.utils.cryo.physics.sharded import projection_splits, split_sinogram

VOLUME_SIZE = (8, 4, 8)
NUM_ANGLES = 7


def _invprob(device=torch.device("cpu"), **params):
    defaults = dict(
        num_angles=NUM_ANGLES,
        noise_level=0.1,
        seed=0,
        tomography_backend="torch",
    )
    defaults.update(params)
    return CryoEIInvProb().get_invprob(
        InvProbConfig(
            size=VOLUME_SIZE,
            batch_size=1,
            channels=1,
            device=device,
            params=defaults,
        )
    )


def test_num_operators_resolution():
    assert resolve_num_operators(None, 4, 41) is None
    assert resolve_num_operators("auto", 4, 41) == 4
    assert resolve_num_operators(8, 4, 41) == 8
    # More shards than angles would build zero-angle operators.
    assert resolve_num_operators("auto", 64, 41) == 41
    assert resolve_num_operators(100, 1, 41) == 41
    with pytest.raises(ValueError):
        resolve_num_operators("all", 4, 41)


@pytest.mark.parametrize("num_operators", [1, 2, 3])
def test_split_sinogram_roundtrip(num_operators):
    sinogram = torch.randn(1, 1, 5, 6, 5)
    shards = split_sinogram(sinogram, num_operators)

    assert len(shards) == num_operators
    assert torch.equal(torch.cat(list(shards), dim=3), sinogram)

    splits = projection_splits(6, num_operators)
    assert splits[0][0] == 0 and splits[-1][1] == 6
    assert all(end == splits[i + 1][0] for i, (_, end) in enumerate(splits[:-1]))


@pytest.mark.parametrize("num_operators", [1, 2])
def test_obs_loss_invariant_to_sharding(num_operators):
    """A sharded stack must score exactly like the unsharded operator.

    This is what ``as_sinogram`` exists for: with the physics sharded, ``A``
    returns one measurement per shard instead of one sinogram, and the loss
    must not notice. Single-rank context — the shards are all local, which is
    enough to exercise the reassembly.
    """
    problem = _invprob()
    torch.manual_seed(0)
    x_net = torch.randn(1, 1, *VOLUME_SIZE)
    y_net = torch.randn(1, 1, *VOLUME_SIZE)

    losses = []
    with DistributedContext(device_mode="cpu", cleanup=False) as ctx:
        for shards in (None, num_operators):
            pair = build_cryo_pair(
                problem.physics,
                problem.measurements,
                ctx.device,
                ctx=ctx,
                num_operators=shards,
                backend="torch",
            )
            y_evn, y_odd = problem.measurements
            if pair.num_operators is not None:
                y_evn = split_sinogram(y_evn, pair.num_operators)
                y_odd = split_sinogram(y_odd, pair.num_operators)
            losses.append(obs_loss(pair, x_net, y_net, y_evn, y_odd).item())

    assert losses[0] == pytest.approx(losses[1], rel=1e-4)


def test_projector_stays_fp32_under_autocast():
    """The projector computes in fp32 whatever precision the denoiser runs in.

    ``grid_sample`` is autocast-fallthrough — it returns whatever dtype it is
    handed — so under autocast the line integrals would silently drop to bf16.
    Both guards are exercised here: ``custom_fwd(cast_inputs=torch.float32)``
    on the autograd wrappers, and ``_work_dtype`` inside them.
    """
    from toolsbench.utils.cryo.physics import TOMOGRAPHY_BACKENDS

    operator = TOMOGRAPHY_BACKENDS["torch"](
        volume_shape=(4, 4, 4), angles_deg=[0.0, 30.0], device="cpu"
    )
    volume = torch.randn(1, 1, 4, 4, 4)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        sinogram = operator.A(volume)
        assert sinogram.dtype == torch.float32
        assert operator.A_adjoint(sinogram).dtype == torch.float32
        assert operator.fbp(sinogram).dtype == torch.float32


@pytest.mark.parametrize("backend,mode", [("torch", "fast"), ("torch_exact", "exact")])
def test_torch_backend_adjoint_mode(backend, mode):
    """``torch`` reproduces astra's approximate back-projector, so flipping
    ``auto`` from astra to torch leaves the gradient unchanged; ``torch_exact``
    opts into the true transpose."""
    from toolsbench.utils.cryo.physics import TOMOGRAPHY_BACKENDS, TomographyEMTorch

    assert set(TOMOGRAPHY_BACKENDS) == {"astra", "torch", "torch_exact"}
    operator = TOMOGRAPHY_BACKENDS[backend](
        volume_shape=(4, 4, 4), angles_deg=[0.0], device="cpu"
    )
    assert isinstance(operator, TomographyEMTorch)
    assert operator.adjoint_mode == mode


def test_unet_wrapper_contract():
    from deepinv.models.base import Denoiser

    model, info = build_unet3d(torch.device("cpu"), f_maps=4, num_levels=2)
    volume = torch.randn(1, 1, *VOLUME_SIZE)

    # A Denoiser instance accepting the physics slot: both are preconditions
    # for distribute(..., type_object="denoiser") to engage at all.
    assert isinstance(model, Denoiser)
    assert model(volume, None).shape == volume.shape
    assert "unet3d" in info


def test_fsc_self_correlation():
    volume = torch.randn(1, 1, *VOLUME_SIZE)
    curve = GpuFSC(device="cpu")(volume, volume)

    assert curve.min() == pytest.approx(1.0, abs=1e-4)
    # A curve that never crosses the threshold reports the last shell.
    assert fsc_shell(curve, 0.143) == len(curve) - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_astra_torch_parity():
    """The two backends must agree — they are interchangeable by config."""
    pytest.importorskip("astra")
    from toolsbench.utils.cryo.physics import build_one_operator

    problem = _invprob()
    device = torch.device("cuda")
    volume = problem.ground_truth.to(device)
    angles = problem.physics.angles_evn.to(device)

    projections = [
        build_one_operator(problem.physics, angles, device, None, backend=backend).A(
            volume
        )
        for backend in ("astra", "torch")
    ]
    relative = (projections[0] - projections[1]).norm() / projections[0].norm()
    assert relative < 5e-2
