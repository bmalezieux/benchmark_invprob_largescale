import inspect
from unittest.mock import patch

import deepinv.models
import pytest

import toolsbench
from toolsbench.utils import (
    DENOISERS,
    DenoiserSpec,
    _resolve_spec,
    create_denoiser,
)

# Behaviour pinned before the registry refactor: the `pretrained` value each
# architecture receives, per spatial dimension. "absent" means the constructor
# takes no `pretrained` argument, so the kwarg must not be passed at all.
ABSENT = "absent"
EXPECTED_PRETRAINED = {
    ("drunet", 2): "download",
    ("drunet", 3): "download_2d",
    ("dncnn", 2): None,
    ("dncnn", 3): None,
    ("unet", 2): ABSENT,
    ("unet", 3): ABSENT,
}


def _recording_cls(real_cls, recorded):
    """Stand-in carrying `real_cls`'s __init__ signature, recording its kwargs.

    `create_denoiser` inspects the signature to decide whether `pretrained` is
    accepted, so the stand-in has to keep it. Nothing is constructed, so no
    weights are downloaded.
    """

    class Fake:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    Fake.__init__.__signature__ = inspect.signature(real_cls.__init__)
    return Fake


@pytest.mark.parametrize("arch,dim", sorted(EXPECTED_PRETRAINED))
@pytest.mark.parametrize("channels", [1, 3])
def test_create_denoiser_pretrained_kwarg_unchanged(arch, dim, channels):
    """The registry must reproduce the pre-refactor `pretrained` policy exactly."""
    shape = (1, channels, 64, 64) if dim == 2 else (1, channels, 64, 64, 64)
    recorded = {}
    spec = DENOISERS[arch]
    stand_in = DenoiserSpec(
        _recording_cls(spec.cls, recorded),
        spec.pretrained_2d,
        spec.pretrained_3d,
        spec.spatial_dim_arg,
    )
    with patch.dict(DENOISERS, {arch: stand_in}):
        create_denoiser(arch, shape, device="cpu")

    expected = EXPECTED_PRETRAINED[(arch, dim)]
    if expected is ABSENT:
        assert "pretrained" not in recorded
    else:
        assert recorded["pretrained"] == expected
    assert recorded["in_channels"] == channels
    assert recorded["dim"] == dim


def test_prepareweights_fetches_both_channel_counts():
    """`toolsbench prepareweights` caches the grayscale and color checkpoints."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        assert toolsbench.main(["prepareweights"]) == 0

    archs = [call.args[0] for call in mock_create.call_args_list]
    channels = [call.args[1][1] for call in mock_create.call_args_list]
    assert archs == ["drunet", "drunet"]
    assert channels == [1, 3]


def test_prepareweights_builds_on_cpu():
    """Weights are only cached to disk, so the throw-away models stay on CPU."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        toolsbench.main(["prepareweights"])

    assert all(call.kwargs["device"] == "cpu" for call in mock_create.call_args_list)


def test_prepareweights_accepts_an_explicit_name():
    """A named architecture is fetched instead of the default set."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        assert toolsbench.main(["prepareweights", "drunet"]) == 0

    assert [call.args[0] for call in mock_create.call_args_list] == ["drunet"] * 2


def test_prepareweights_skips_architectures_without_weights(capsys):
    """unet has no pretrained weights here: skip it rather than failing."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        assert toolsbench.main(["prepareweights", "unet"]) == 0

    assert mock_create.call_args_list == []
    assert "no pretrained weights" in capsys.readouterr().out


def test_prepareweights_rejects_an_unknown_name(capsys):
    """A name in neither the registry nor deepinv is a user error, not a no-op."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        assert toolsbench.main(["prepareweights", "definitely_not_a_denoiser"]) == 1

    assert mock_create.call_args_list == []
    assert "Unknown denoiser" in capsys.readouterr().out


def test_prepareweights_accepts_an_unlisted_deepinv_model():
    """An architecture absent from the registry is fetched from deepinv."""
    with patch("toolsbench.utils.create_denoiser") as mock_create:
        assert toolsbench.main(["prepareweights", "scunet"]) == 0

    assert [call.args[0] for call in mock_create.call_args_list] == ["scunet"] * 2


def test_default_set_is_derived_from_the_registry():
    """Adding a weighted architecture must extend the default set on its own."""
    default = [name for name, spec in DENOISERS.items() if spec.has_weights]
    assert default == ["drunet"]


# ---------------------------------------------------------------------------
# Fallback to deepinv.models for architectures not in the registry
# ---------------------------------------------------------------------------


def test_unlisted_deepinv_model_resolves():
    """A denoiser absent from the registry is looked up in deepinv.models."""
    spec = _resolve_spec("scunet")
    assert spec.cls is deepinv.models.SCUNet
    assert spec.has_weights


def test_resolution_is_case_insensitive():
    """Users type `swinir`, deepinv spells it `SwinIR`."""
    assert _resolve_spec("swinir").cls is deepinv.models.SwinIR


def test_registry_wins_over_deepinv():
    """dncnn is registered as weightless here, overriding deepinv's default."""
    assert _resolve_spec("dncnn") is DENOISERS["dncnn"]
    assert not _resolve_spec("dncnn").has_weights


def test_unknown_everywhere_still_raises():
    """A name in neither place is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="Unknown denoiser"):
        _resolve_spec("definitely_not_a_denoiser")


def test_dim_is_not_forwarded_to_unregistered_models():
    """`dim` is a layer width in SCUNet (64) and Restormer (48), not 2D-vs-3D.

    Forwarding a spatial dimension there builds a 2-channel-wide network whose
    pretrained checkpoint no longer fits, so it must not be passed.
    """
    recorded = {}
    spec = _resolve_spec("scunet")
    stand_in = DenoiserSpec(_recording_cls(spec.cls, recorded), "download", "download")
    with patch.dict(DENOISERS, {"scunet": stand_in}):
        create_denoiser("scunet", (1, 3, 64, 64), device="cpu")

    assert "dim" not in recorded
    assert "in_channels" not in recorded  # SCUNet does not declare it either


def test_falls_back_when_model_rejects_download():
    """Restormer wants `pretrained='denoising'`; retry without the argument."""
    attempts = []

    class Picky:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            if "pretrained" in kwargs:
                raise ValueError("unsupported pretrained value")

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    Picky.__init__.__signature__ = inspect.signature(DENOISERS["drunet"].cls.__init__)
    with patch.dict(DENOISERS, {"picky": DenoiserSpec(Picky, "download", "download")}):
        create_denoiser("picky", (1, 3, 64, 64), device="cpu")

    assert len(attempts) == 2
    assert attempts[0]["pretrained"] == "download"
    assert "pretrained" not in attempts[1]


def test_3d_request_on_a_2d_only_model_raises():
    """Silently building a 2D network for a 3D problem would be a wrong answer."""
    with pytest.raises(ValueError, match="does not support 3D"):
        create_denoiser("swinir", (1, 3, 32, 32, 32), device="cpu")
