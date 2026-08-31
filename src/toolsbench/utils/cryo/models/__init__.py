"""The 3D denoiser used by the ``tomo_ei`` case.

icecream's ``UNet3D`` (vendored in ``unet3d.py`` / ``blocks3d.py`` /
``se3d.py``) plus the wrapper demo_cyo puts around it. This is the 3D network
this case needs; other benchmark cases use deepinv's 2D denoisers through
``toolsbench.utils.create_denoiser``, and the two are not interchangeable.
"""

from __future__ import annotations

import torch
from deepinv.models.base import Denoiser

from .unet3d import UNet3D

__all__ = ["IceCreamUNetWrapper", "UNet3D", "build_unet3d"]

#: demo_cyo's defaults (toolcryo/models.py::build_ei_model).
UNET_F_MAPS = 64
UNET_NUM_LEVELS = 4


class IceCreamUNetWrapper(Denoiser):
    """Wraps ``UNet3D`` so ``model(x, physics)`` works.

    deepinv calls ``model(y, physics)``; the physics object would otherwise
    land on ``UNet3D``'s ``pos_enc`` argument. Subclassing
    ``deepinv.models.base.Denoiser`` is not cosmetic: deepinv's tiling only
    engages when the target of ``distribute(..., type_object="denoiser")`` is a
    ``Denoiser`` instance.
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.unet(x)


def build_unet3d(
    device,
    f_maps: int = UNET_F_MAPS,
    num_levels: int = UNET_NUM_LEVELS,
    dropout: float = 0.0,
) -> tuple[torch.nn.Module, str]:
    """Build the wrapped UNet3D on ``device``, with demo_cyo's settings.

    ``UNet3D`` only inserts Dropout when ``layer_order`` asks for it ('d');
    ``dropout_prob`` alone is inert. 'd' not 'D': Dropout2d is deprecated on
    5-D input.
    """
    layer_order = "crd" if dropout > 0 else "cr"
    unet = UNet3D(
        in_channels=1,
        out_channels=1,
        f_maps=f_maps,
        num_levels=num_levels,
        layer_order=layer_order,
        use_bias=False,
        dropout_prob=dropout,
    ).to(device)
    info = f"unet3d f_maps={f_maps} num_levels={num_levels} layer_order={layer_order}"
    return IceCreamUNetWrapper(unet).to(device), info
