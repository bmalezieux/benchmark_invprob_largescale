"""The two ``tomo_ei`` loss terms.

Adapted from demo_cyo: toolcryo/losses/losses_unrolled.py (``ObsLoss``) and
toolcryo/losses/losses_equivariant_tomo.py (``EqLoss``). Written as functions
rather than ``deepinv.loss.Loss`` subclasses — no ``Trainer`` consumes them
here, the solver calls them directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["as_sinogram", "eq_loss", "obs_loss"]


def as_sinogram(projection) -> torch.Tensor:
    """Reassemble a sharded ``A(x)`` into one ``(B, C, V, A, N)`` sinogram.

    With the physics sharded, ``A`` returns one measurement per shard (a
    ``TensorList``) rather than a tensor. The shards are contiguous and in
    ascending angle order, so concatenating on the angle axis rebuilds exactly
    the sinogram the unsharded operator would have produced — which is what
    keeps the loss numerically identical whatever ``num_operators`` is.
    """
    return (
        projection
        if torch.is_tensor(projection)
        else torch.cat(list(projection), dim=3)
    )


def obs_loss(
    pair, x_net: torch.Tensor, y_net: torch.Tensor, y_evn, y_odd
) -> torch.Tensor:
    """Cross half-set data fidelity in the measurement domain.

    ``MSE(A_odd(f(evn)), y_odd) + MSE(A_evn(f(odd)), y_evn)``: each half's
    reconstruction is scored against the *other* half's measurements, which is
    what makes the loss self-supervised.

    Sharded or not: ``as_sinogram`` reassembles a sharded ``A(x)`` on the angle
    axis, so the value is identical whatever ``num_operators`` is, and the
    backward flows through the gather into each shard's own operator.
    """
    return F.mse_loss(
        as_sinogram(pair.physics_odd.A(x_net)), as_sinogram(y_odd)
    ) + F.mse_loss(as_sinogram(pair.physics_evn.A(y_net)), as_sinogram(y_evn))


def _eq_term(model, physics, transform, x_net: torch.Tensor) -> torch.Tensor:
    params = transform.get_params(x_net)
    x_rot = transform.transform(x_net, **params)
    return F.mse_loss(model(physics.fbp(physics.A(x_rot))), x_rot)


def eq_loss(
    pair, model, transform, x_net: torch.Tensor, y_net: torch.Tensor
) -> torch.Tensor:
    """Equivariance under a random shape-preserving rotation.

    The rotated volume is re-projected through the real geometry and
    reconstructed by ``fbp``, rather than having a wedge mask rotated in
    frequency space: ``A`` is volume->sinogram here, so the missing-angle
    pattern can only be imprinted by going through the acquisition geometry.
    """
    return _eq_term(model, pair.physics_evn, transform, x_net) + _eq_term(
        model, pair.physics_odd, transform, y_net
    )
