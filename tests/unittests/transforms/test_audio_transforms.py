from packaging import version
import matplotlib.pyplot as plt
import pytest
import torch

from samantha.transforms.audio import MelSpectrogram, Pad, RandomPad, Spectrogram


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


def test_spectrogram():
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    transform = Spectrogram(n_fft=1024, win_length=1024, hop_length=256)

    y = transform(x)

    fig, ax = plt.subplots(1, 1)
    # transform.plot(x, ax=ax)
    # plt.savefig("spectrogram.png")


def test_mel_spectrogram():
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    transform = MelSpectrogram(
        sample_rate=24000, n_mels=128, n_fft=1024, win_length=1024, hop_length=256
    )
    y = transform(x)
    fig, ax = plt.subplots(1, 1)
    # transform.plot(x, ax=ax)
    # plt.savefig("melspectrogram.png")

@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.4.0"),
    reason="`torch.compile` is not compatible with triton 3.0.0 in our training image.",
)
def test_compile_transforms():
    transform = Spectrogram(n_fft=1024, win_length=1024, hop_length=256)

    mel_transform = MelSpectrogram(
        sample_rate=24000, n_mels=128, n_fft=1024, win_length=1024, hop_length=256
    )

    torch.compile(transform)
    torch.compile(mel_transform)
