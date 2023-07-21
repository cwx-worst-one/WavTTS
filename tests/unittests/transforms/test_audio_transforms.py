import torch
from samantha.transforms.audio import Spectrogram, MelSpectrogram
import matplotlib.pyplot as plt


def test_spectrogram():
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    transform = Spectrogram(n_fft=1024, win_length=1024, hop_length=256)

    y = transform(x)

    fig, ax = plt.subplots(1, 1)
    transform.plot(x, ax=ax)
    plt.savefig("spectrogram.png")


def test_mel_spectrogram():
    batch_size = 1
    x = torch.randn(batch_size, 1, 24000)
    transform = MelSpectrogram(
        sample_rate=24000, n_mels=128, n_fft=1024, win_length=1024, hop_length=256
    )
    y = transform(x)
    fig, ax = plt.subplots(1, 1)
    transform.plot(x, ax=ax)
    plt.savefig("melspectrogram.png")

def test_compile_transforms():
    transform = Spectrogram(n_fft=1024, win_length=1024, hop_length=256)

    mel_transform = MelSpectrogram(
        sample_rate=24000, n_mels=128, n_fft=1024, win_length=1024, hop_length=256
    )
    
    torch.compile(transform)
    torch.compile(mel_transform)