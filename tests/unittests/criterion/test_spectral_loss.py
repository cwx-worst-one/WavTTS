import pytest
import torch
from packaging import version

from samantha.criterion.spectral_loss import (
    MagnitudeSTFTLoss,
    MultiScaleSTFTLoss,
    STFTLoss,
)
from samantha.transforms.audio import Spectrogram


def test_mag_stft_loss():
    batch_size = 1
    n_fft = 1024
    win_length = 1024
    hop_length = 256

    x = torch.randn(batch_size, 1, 24000)
    y = x

    transform = Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,  # magnitude spectrogram
    )

    mag_x = transform(x)
    mag_y = transform(y)

    criterion = MagnitudeSTFTLoss()
    loss = criterion(mag_x, mag_y)
    assert loss == 0


def test_stft_loss():
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    y = x
    criterion = STFTLoss(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        scale=None,
    )
    loss = criterion(x, y)
    assert loss == 0

@pytest.mark.parametrize("scale", [None, "mel"])
def test_multistft_loss(scale):
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    y = x
    criterion = MultiScaleSTFTLoss(
        n_ffts=[1024, 2048, 4096],
        win_lengths=[1024, 2048, 4096],
        hop_lengths=[256, 512, 1024],
        scale=scale,
        n_mels=[64, 128, 128],
        sample_rate=48000,
    )
    loss, losses = criterion(x, y, return_all=True)
    assert loss == 0
    assert len(losses) == 3


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.4.0"),
    reason="`torch.compile` is not compatible with triton 3.0.0 in our training image.",
)
@pytest.mark.parametrize("scale", [None, "mel"])
def test_compile_loss(scale):

    mag_criterion = MagnitudeSTFTLoss()
    stft_criterion = STFTLoss(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        scale=None,
    )
    criterion = MultiScaleSTFTLoss(
        n_ffts=[1024, 2048, 4096],
        win_lengths=[1024, 2048, 4096],
        hop_lengths=[256, 512, 1024],
        scale=scale,
        n_mels=[64, 128, 128],
        sample_rate=48000,
    )

    torch.compile(mag_criterion)
    torch.compile(stft_criterion)
    torch.compile(criterion)
