"""Benchmark utilities for inverse problems.

This module provides shared utilities for datasets, solvers, and objectives,
including visualization and data loading helpers.
"""

try:
    from .solver_utils import (
        build_solver_name as build_solver_name,
        distributed_callback_iter as distributed_callback_iter,
        get_device_from_context as get_device_from_context,
        initialize_reconstruction as initialize_reconstruction,
        setup_distributed_env as setup_distributed_env,
        sync_and_barrier as sync_and_barrier,
    )
except ImportError:
    pass

import inspect
import math
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import deepinv.models
    from deepinv.utils.demo import download_example, load_image
    from deepinv.models import DRUNet, DnCNN, UNet
except ImportError:
    torch = None
    deepinv = None
    download_example = None
    load_image = None
    DRUNet = None
    DnCNN = None
    UNet = None


@dataclass(frozen=True)
class DenoiserSpec:
    """How to build one denoiser architecture and where its weights come from.

    ``pretrained_2d`` / ``pretrained_3d`` hold the value passed to the model's
    ``pretrained`` argument, or ``None`` when the architecture has no weights
    to download in this benchmark. Architectures whose constructor takes no
    ``pretrained`` argument at all leave both as ``None``.
    """

    cls: type
    pretrained_2d: str | None = None
    pretrained_3d: str | None = None
    spatial_dim_arg: bool = False

    @property
    def has_weights(self):
        """Whether this architecture has pretrained weights to fetch."""
        return self.pretrained_2d is not None or self.pretrained_3d is not None


#: Registry of available denoisers. Adding an entry here makes the
#: architecture available to :func:`create_denoiser` and, when it declares
#: pretrained weights, to ``toolsbench prepareweights`` at the same time.
DENOISERS = {
    "drunet": DenoiserSpec(DRUNet, "download", "download_2d", spatial_dim_arg=True),
    "unet": DenoiserSpec(UNet, spatial_dim_arg=True),
    "dncnn": DenoiserSpec(DnCNN, spatial_dim_arg=True),
}


def _resolve_spec(arch):
    """Registry first, then any deepinv.models denoiser with a matching name.

    Falling back to deepinv means an architecture does not have to be listed in
    :data:`DENOISERS` to be usable; the registry only exists to override what
    deepinv would do by default, as it does for DnCNN.
    """
    key = str(arch).lower()
    if key in DENOISERS:
        return DENOISERS[key]

    cls = next(
        (getattr(deepinv.models, n) for n in dir(deepinv.models) if n.lower() == key),
        None,
    )
    if cls is None:
        raise ValueError(
            f"Unknown denoiser: {arch}. Choose from {sorted(DENOISERS)} "
            "or the name of a deepinv.models denoiser."
        )
    return DenoiserSpec(cls, "download", "download")


def tensor_to_numpy(tensor, clip=True):
    """Convert tensor to numpy array suitable for visualization.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (B, C, D, H, W), (B, C, H, W), (C, H, W), or (H, W).

    Returns
    -------
    np.ndarray
        Numpy array suitable for matplotlib imshow, with values in [0, 1].
    """
    img = tensor.detach().cpu()

    # Handle 5D tensor (B, C, D, H, W) -> take middle slice
    if img.ndim == 5:
        mid_slice = img.shape[2] // 2
        img = img[:, :, mid_slice, :, :]

    # Remove batch dimension if present
    if img.ndim == 4:
        img = img.squeeze(0)

    # Transpose from (C, H, W) to (H, W, C) for color images
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)

    # Remove channel dimension for grayscale
    if img.shape[-1] == 1:
        img = img.squeeze(-1)

    # Clip to valid range
    if clip:
        img = torch.clamp(img, 0, 1)

    return img.numpy()


def save_measurements_figure(
    ground_truth,
    measurement,
    output_dir="debug_output",
    filename="measurements_debug.png",
):
    """Save a figure showing ground truth and all measurements.

    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth image tensor of shape (C, H, W) or (B, C, H, W).
    measurement : list of torch.Tensor or TensorList
        List of measurement tensors, each of shape (C, H, W) or (B, C, H, W).
    output_dir : str or Path, optional
        Directory to save the debug figure. Default: "debug_output".
    filename : str, optional
        Name of the output file. Default: "measurements_debug.png".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_measurements = len(measurement)
    total_images = num_measurements + 1  # +1 for ground truth

    # Create grid layout (max 4 columns)
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    # Plot ground truth
    gt_img = tensor_to_numpy(ground_truth, clip=False)
    axes_flat[0].imshow(gt_img, cmap="gray" if gt_img.ndim == 2 else None)
    axes_flat[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes_flat[0].axis("off")

    # Plot measurements
    for i, meas in enumerate(measurement):
        meas_img = tensor_to_numpy(meas)
        axes_flat[i + 1].imshow(meas_img, cmap="gray" if meas_img.ndim == 2 else None)
        axes_flat[i + 1].set_title(f"Measurement {i + 1}", fontsize=12)
        axes_flat[i + 1].axis("off")

    # Hide unused subplots
    for i in range(total_images, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    output_file = output_path / filename
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Debug figure saved to {output_file}")


def save_comparison_figure(
    ground_truth,
    reconstruction,
    metrics,
    output_dir="evaluation_output",
    filename="comparison.png",
    evaluation_count=None,
    vmin=None,
    vmax=None,
):
    """Save a comparison figure showing ground truth and reconstruction side by side.

    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth tensor of shape (C, H, W) or (B, C, H, W).
    reconstruction : torch.Tensor
        Reconstruction tensor of same shape as ground_truth.
    metrics : dict
        Dictionary with 'psnr' and 'ssim' keys.
    output_dir : str or Path, optional
        Directory to save the comparison figure. Default: "evaluation_output".
    filename : str, optional
        Name of the output file. Default: "comparison.png".
    evaluation_count : int, optional
        Evaluation number to display in title.
    vmin : float, optional
        Lower bound used for both displays.
    vmax : float, optional
        Upper bound used for both displays.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    gt_img = tensor_to_numpy(ground_truth, clip=False)
    recon_img = tensor_to_numpy(reconstruction, clip=False)
    psnr = metrics.get("psnr", 0)
    ssim = metrics.get("ssim", 0)
    asinh_psnr = metrics.get("asinh_psnr", None)

    # --- Row 0: linear scale ---
    axes[0, 0].imshow(
        gt_img,
        cmap="gray" if gt_img.ndim == 2 else None,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title("Ground Truth (linear)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    recon_title = f"Reconstruction (linear)\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}"
    if asinh_psnr is not None:
        recon_title += f"\nasinh-PSNR: {asinh_psnr:.2f} dB"
    axes[0, 1].imshow(
        recon_img,
        cmap="gray" if recon_img.ndim == 2 else None,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title(recon_title, fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # --- Row 1: asinh scale ---
    beta_display = max(gt_img.max() * 1e-2, 1e-10)
    asinh_gt_img = np.arcsinh(gt_img / beta_display)
    asinh_recon_img = np.arcsinh(recon_img / beta_display)
    asinh_vmin = np.arcsinh(gt_img.min() / beta_display)
    asinh_vmax = np.arcsinh(gt_img.max() / beta_display)

    axes[1, 0].imshow(
        asinh_gt_img,
        cmap="gray" if asinh_gt_img.ndim == 2 else None,
        vmin=asinh_vmin,
        vmax=asinh_vmax,
    )
    axes[1, 0].set_title("Ground Truth (asinh)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        asinh_recon_img,
        cmap="gray" if asinh_recon_img.ndim == 2 else None,
        vmin=asinh_vmin,
        vmax=asinh_vmax,
    )
    axes[1, 1].set_title("Reconstruction (asinh)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    # Add overall title if evaluation count provided
    if evaluation_count is not None:
        fig.suptitle(f"Evaluation #{evaluation_count}", fontsize=16, fontweight="bold")

    plt.tight_layout()
    output_file = output_path / filename
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_cached_example(name, cache_dir=None, **kwargs):
    """Load example image with local caching.

    Downloads the image from HuggingFace if not already cached locally.
    This allows benchmarks to run on cluster nodes without internet access.

    Parameters
    ----------
    name : str
        Filename of the image from the HuggingFace dataset.
    cache_dir : str or Path, optional
        Directory to cache downloaded images.
        Defaults to 'data' folder in benchmark root.
    **kwargs
        Additional arguments passed to load_image.

    Returns
    -------
    torch.Tensor
        The loaded image tensor.
    """
    if cache_dir is None:
        # Get benchmark root from src/toolsbench/utils/__init__.py.
        benchmark_root = Path(__file__).resolve().parents[3]
        cache_dir = benchmark_root / "data"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / name

    # Download if not cached
    if not cached_file.exists():
        print(f"Downloading {name} to {cache_dir}...")
        download_example(name, cache_dir)
        print(f"Downloaded {name} successfully.")
    else:
        print(f"Using cached {name} from {cache_dir}")

    # Load from local file
    if name.endswith(".pt"):
        device = kwargs.get("device", "cpu")
        return torch.load(cached_file, weights_only=True, map_location=device)
    else:
        return load_image(str(cached_file), **kwargs)


def create_denoiser(arch, ground_truth_shape, device="cpu", dtype=None):
    """Create a denoiser appropriate for the given ground truth shape.

    Parameters
    ----------
    arch : str
        Architecture name: 'drunet', 'unet' or 'dncnn' (case-insensitive).
    ground_truth_shape : tuple
        Shape of the ground truth tensor: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.
    device : str or torch.device, optional
        Device to load the model on. Default: 'cpu'.
    dtype : torch.dtype, optional
        Data type for the model. Default: torch.float32.

    Returns
    -------
    torch.nn.Module
        Configured denoiser model.
    """
    if torch is None:
        raise ImportError("Torch is required to create a denoiser.")

    if dtype is None:
        dtype = torch.float32

    spec = _resolve_spec(arch)
    model_cls = spec.cls

    ndim = len(ground_truth_shape)
    if ndim == 4:
        dim, num_channels = 2, ground_truth_shape[1]
    elif ndim == 5:
        dim, num_channels = 3, ground_truth_shape[1]
    else:
        raise ValueError(
            f"Unsupported ground truth shape: {ground_truth_shape}. "
            "Expected 4D (B,C,H,W) or 5D (B,C,D,H,W)."
        )

    if num_channels not in (1, 3):
        raise ValueError(
            f"Unsupported number of channels: {num_channels}. Expected 1 (grayscale) or 3 (color)."
        )

    # Architectures accept different subsets of these arguments, so only pass
    # the ones this one actually declares. ``dim`` is only forwarded for
    # registered architectures: elsewhere in deepinv the same name means a
    # layer width (SCUNet defaults to 64, Restormer to 48), so passing a
    # spatial dimension there would silently build the wrong network.
    params = inspect.signature(model_cls.__init__).parameters
    if dim == 3 and not spec.spatial_dim_arg:
        raise ValueError(f"Denoiser {arch} does not support 3D inputs.")

    kwargs = {
        key: value
        for key, value in dict(
            in_channels=num_channels, out_channels=num_channels
        ).items()
        if key in params
    }
    if spec.spatial_dim_arg and "dim" in params:
        kwargs["dim"] = dim
    if "pretrained" in params:
        # For 3D, deepinv initialises 3D convolutions from 2D pretrained weights.
        kwargs["pretrained"] = spec.pretrained_3d if dim == 3 else spec.pretrained_2d

    try:
        model = model_cls(**kwargs)
    except (TypeError, ValueError):
        # The model rejects this ``pretrained`` value; fall back to its own default.
        kwargs.pop("pretrained", None)
        model = model_cls(**kwargs)

    return model.to(dtype).to(device).eval()


def create_drunet_denoiser(ground_truth_shape, device="cpu", dtype=None):
    """Create a DRUNet denoiser appropriate for the given ground truth shape."""
    return create_denoiser("drunet", ground_truth_shape, device=device, dtype=dtype)


def download_denoiser_weights(names=None):
    """Cache pretrained denoiser weights in the local torch hub directory.

    Builds and discards one model per channel count: the point is the side
    effect of ``torch.hub`` writing the checkpoint to disk, so that a later
    run on a compute node without internet access finds it already cached.
    A 3D network initialises from the same 2D checkpoints, so the channel
    count is the only axis that matters here.

    Parameters
    ----------
    names : iterable of str, optional
        Architectures to fetch. Default: every entry in :data:`DENOISERS`
        declaring pretrained weights. Names without weights are skipped with
        a message; unknown names raise :class:`ValueError`.
    """
    if names is None:
        names = [name for name, spec in DENOISERS.items() if spec.has_weights]

    for name in names:
        spec = _resolve_spec(name)
        if not spec.has_weights:
            print(f"{name}: no pretrained weights in this benchmark, skipping")
            continue
        for num_channels in (1, 3):
            create_denoiser(name, (1, num_channels, 64, 64), device="cpu")
        print(f"{name}: cached (1 channel, 3 channels)")


def compute_psnr(reconstruction, reference, max_pixel=1.0):
    """Compute PSNR in dB."""
    reconstruction = reconstruction.to(reference.device)
    mse = torch.mean((reconstruction - reference) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_pixel**2) / mse)
