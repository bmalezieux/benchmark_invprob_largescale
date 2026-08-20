"""Cryo-ET (``tomo_ei``) helpers, vendored and adapted from demo_cyo.

Physics (both the astra and the pure-torch operator, plus angle sharding), the
icecream UNet3D, the two self-supervised loss terms, the rotation used by the
equivariance term, and the GPU FSC metric. The benchmark-facing pieces live in
``toolsbench.invprob.cryo_ei`` and ``toolsbench.solver.tomo_ei``.
"""

from .fsc import GpuFSC, fsc_resolution, fsc_shell
from .losses import as_sinogram, eq_loss, obs_loss
from .models import build_unet3d
from .physics import (
    CryoEISpec,
    CryoPair,
    build_cryo_pair,
    resolve_num_operators,
    resolve_tomography_backend,
    split_sinogram,
)
from .transform import Rotate3D

__all__ = [
    "CryoEISpec",
    "CryoPair",
    "GpuFSC",
    "Rotate3D",
    "as_sinogram",
    "build_cryo_pair",
    "build_unet3d",
    "eq_loss",
    "fsc_resolution",
    "fsc_shell",
    "obs_loss",
    "resolve_num_operators",
    "resolve_tomography_backend",
    "split_sinogram",
]
