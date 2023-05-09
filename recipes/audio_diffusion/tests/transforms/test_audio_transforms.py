import pytest
import torch

from recipes.audio_diffusion.modules.transforms.audio import (
    Pad,
    RandomPad,
    RandomResizedCrop,
    ResizedCrop,
)


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("start_idx", [5, 100])
@pytest.mark.parametrize("n_samples", [1, 200])
def test_crop(n_channels, start_idx, n_samples):
    total_samples = 24000
    audio = torch.ones(n_channels, total_samples)
    transform = ResizedCrop(start_idx, n_samples)

    t_audio = transform(audio)
    assert t_audio.shape[0] == n_channels
    assert t_audio.shape[1] == n_samples


def test_crop_index_bounds():
    max_samples = 24000
    audio = torch.ones(1, max_samples)
    transform = ResizedCrop(23999, 400)

    with pytest.raises(IndexError):
        transform(audio)

    transform = ResizedCrop(max_samples - 400, 400)
    t_audio = transform(audio)
    assert t_audio.shape[1] == 400


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("n_samples", [1, 200])
def test_random_crop(n_channels, n_samples):
    max_samples = 24000
    audio = torch.ones(n_channels, max_samples)
    transform = RandomResizedCrop(n_samples)
    t_audio = transform(audio)
    assert t_audio.shape[0] == n_channels
    assert t_audio.shape[1] == n_samples


@pytest.mark.parametrize("n_channels", [1, 2])
def test_pad(n_channels):
    n_samples = 20
    audio = torch.ones(n_channels, n_samples)
    transform = Pad(n_samples)

    t_audio = transform(audio)
    assert torch.equal(t_audio, audio)
    assert t_audio.shape[0] == n_channels
    assert t_audio.shape[1] == n_samples

    audio_sample_len = n_samples - 10
    audio = torch.ones(n_channels, audio_sample_len)
    t_audio = transform(audio)
    assert t_audio.shape[0] == n_channels
    assert t_audio.shape[1] == n_samples
    assert t_audio[n_channels - 1, audio_sample_len - 1] == 1
    assert t_audio[n_channels - 1, audio_sample_len] == 0


@pytest.mark.parametrize("n_channels", [1, 2])
def test_random_pad(mocker, n_channels):
    rand_pad_idx = 5
    mocker.patch("random.randint", return_value=rand_pad_idx)

    n_samples = 20
    audio = torch.ones(n_channels, n_samples)
    transform = RandomPad(n_samples)
    t_audio = transform(audio)
    assert torch.equal(t_audio, audio)

    audio_sample_len = n_samples - 10
    audio = torch.ones(n_channels, audio_sample_len)
    t_audio = transform(audio)
    assert int(torch.count_nonzero(t_audio)) == n_channels * audio_sample_len

    assert t_audio[n_channels - 1, rand_pad_idx] == 1
    assert t_audio[n_channels - 1, rand_pad_idx + audio_sample_len - 1] == 1
