import pytest
import torch

from recipes.musiclm.transforms.audio import Pad, RandomPad


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
