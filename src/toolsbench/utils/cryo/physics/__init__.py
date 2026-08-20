"""Synthetic cryo-ET tomography physics for the ``tomo_ei`` benchmark case.

The synthetic counterpart of ``toolcryo.physics.build_tomography_physics``:
the operators, the angle sharding and the normalisation are the same, but the
tilt angles come from a config instead of a ``.tlt`` file and the FBP init is
computed from the simulated sinogram instead of being read from an MRC volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sharded import (
    TOMOGRAPHY_BACKENDS,
    ShardedTomography,
    normalize_sharded,
    projection_splits,
    resolve_tomography_backend,
    split_sinogram,
)
from .tomography import TomographyEM
from .tomography_torch import TomographyEMTorch

__all__ = [
    "TOMOGRAPHY_BACKENDS",
    "CryoEISpec",
    "CryoPair",
    "ShardedTomography",
    "TomographyEM",
    "TomographyEMTorch",
    "build_cryo_pair",
    "build_one_operator",
    "normalize_sharded",
    "projection_splits",
    "resolve_num_operators",
    "resolve_tomography_backend",
    "split_sinogram",
]

#: Calibrated for the EMPIAR-11830 acquisition convention in demo_cyo — the
#: tilt sign matches IMOD's convention negated. Kept so a synthetic problem has
#: the same geometry as the real one.
TOMO_ANGLE_SIGN = -1.0


@dataclass
class CryoEISpec:
    """Everything needed to rebuild the operators, and nothing else.

    Carried through ``InvProb.physics`` as inert data: the objective never
    calls it, and the solver builds the real operators from it, because
    sharding needs a ``DistributedContext`` that only exists solver-side.
    """

    volume_shape: tuple[int, int, int]
    angles_evn: torch.Tensor
    angles_odd: torch.Tensor
    angle_sign: float = TOMO_ANGLE_SIGN
    detector_shape: tuple[int, int] | None = None

    @property
    def num_angles(self) -> int:
        """Angles of a single half — the cap that applies to a half's shards."""
        return int(self.angles_evn.numel())


@dataclass
class CryoPair:
    """A volume's two half-set operators and their FBP-init volumes.

    The synthetic stand-in for ``toolcryo.physics.TomographyEMPair``, minus the
    lazy per-tomogram rebuild (one volume here, so nothing to switch to).

    With ``num_operators=None`` each rank runs the full operator locally and
    only the denoiser is tiled; otherwise the angles are sharded and every
    ``A_adjoint``/``fbp`` costs a collective.
    """

    physics_evn: object
    physics_odd: object
    init_evn: torch.Tensor
    init_odd: torch.Tensor
    num_operators: int | None = None
    backend: str = "astra"
    volume_shape: tuple[int, int, int] = field(default=())


def resolve_num_operators(
    num_operators: int | str | None, world_size: int, num_angles: int
) -> int | None:
    """Resolve ``num_operators`` the way demo_cyo does.

    ``None`` leaves the physics unsharded — one full operator held locally by
    every rank, no physics collective. ``"auto"`` is one operator per rank. An
    int is taken as given. Either way the count is capped at the tilt-angle
    count: more shards than angles would build zero-angle operators, which the
    operator constructors reject. On 64 ranks over 41 angles this builds 41
    shards; the spare ranks hold no physics (deepinv supports empty ranks) but
    still carry their denoiser tiles.
    """
    if isinstance(num_operators, str):
        # A YAML/CLI config hands strings through: "null"/"none" is the unsharded
        # case spelled out, "auto" is one operator per rank.
        key = num_operators.strip().lower()
        if key in ("none", "null"):
            return None
        if key != "auto":
            raise ValueError(
                f"num_operators must be None, 'auto' or an int, got {num_operators!r}."
            )
        num_operators = int(world_size)
    if num_operators is None:
        return None
    return max(1, min(int(num_operators), int(num_angles)))


def build_one_operator(
    spec: CryoEISpec,
    angles: torch.Tensor,
    device,
    num_operators: int | None,
    ctx=None,
    backend: str = "astra",
):
    """One half's operator: a plain one, or ``num_operators`` angle shards.

    Shards are built with ``normalize=False`` — a shard's own spectral norm is
    not the assembled operator's — and are rescaled afterwards by
    ``normalize_sharded``, so sharded and unsharded physics stay identical.
    """
    operator_cls = TOMOGRAPHY_BACKENDS[backend]
    common = dict(
        volume_shape=spec.volume_shape,
        detector_shape=spec.detector_shape,
        angle_sign=spec.angle_sign,
        device=str(device),
    )

    if num_operators is None:
        return operator_cls(angles_deg=angles, normalize=True, **common)

    splits = projection_splits(int(angles.numel()), int(num_operators))

    def _factory(index: int, dev, shared=None):
        start, end = splits[index]
        return operator_cls(
            angles_deg=angles[start:end],
            normalize=False,
            **{**common, "device": str(dev)},
        )

    physics = ShardedTomography(ctx, int(num_operators), _factory)
    # The shards each hold a slice of the angles; the container carries the
    # global range and count, which ``fbp`` needs for its A_i/A reweighting.
    physics._tilt_min = float(angles.min())
    physics._tilt_max = float(angles.max())
    physics.n_angles_total = int(angles.numel())
    physics.volume_shape = spec.volume_shape
    return physics


def build_cryo_pair(
    spec: CryoEISpec,
    measurements,
    device,
    ctx=None,
    num_operators: int | str | None = None,
    backend: str = "auto",
) -> CryoPair:
    """Build both half-set operators and their FBP inits from simulated sinograms.

    ``measurements`` is the ``(y_evn, y_odd)`` pair produced by the dataset,
    each of shape ``(B, C, V, A, N)``. The inits are ``fbp(y)`` — the same
    z-normalised starting point ``load_fbp_init`` produces from an MRC volume in
    demo_cyo, computed here instead of read from disk.
    """
    world_size = int(getattr(ctx, "world_size", 1) or 1)
    n_ops = resolve_num_operators(num_operators, world_size, spec.num_angles)
    backend = resolve_tomography_backend(backend, device)

    physics_evn = build_one_operator(
        spec, spec.angles_evn.to(device), device, n_ops, ctx, backend
    )
    physics_odd = build_one_operator(
        spec, spec.angles_odd.to(device), device, n_ops, ctx, backend
    )

    y_evn, y_odd = (m.to(device) for m in measurements)
    if n_ops is not None:
        y_evn, y_odd = split_sinogram(y_evn, n_ops), split_sinogram(y_odd, n_ops)

    with torch.no_grad():
        init_evn = physics_evn.fbp(y_evn)
        init_odd = physics_odd.fbp(y_odd)
        if n_ops is not None:
            # A shard's norm is not the operator's, so the shards are built
            # unnormalised and rescaled here by the measured global norm.
            normalize_sharded(physics_evn, init_evn)
            normalize_sharded(physics_odd, init_odd)
            init_evn = physics_evn.fbp(y_evn)
            init_odd = physics_odd.fbp(y_odd)
        init_evn = _znorm(init_evn)
        init_odd = _znorm(init_odd)

    return CryoPair(
        physics_evn=physics_evn,
        physics_odd=physics_odd,
        init_evn=init_evn,
        init_odd=init_odd,
        num_operators=n_ops,
        backend=backend,
        volume_shape=spec.volume_shape,
    )


def _znorm(volume: torch.Tensor) -> torch.Tensor:
    """Centre and scale to unit std, as demo_cyo's ``load_fbp_init`` does.

    Centring matters as much as scaling: a volume with a non-zero mean projects
    to a constant offset in ``A(x)`` that can never match a centred sinogram.
    """
    return (volume - volume.mean()) / (volume.std() + 1e-8)
