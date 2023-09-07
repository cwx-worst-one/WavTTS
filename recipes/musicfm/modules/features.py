import numpy as np
import torch
import torchaudio
from einops import einsum, rearrange
from librosa import filters, hz_to_note, midi_to_hz, note_to_midi, util
from torch import nn


def get_chromatic_f0(sample_rate):
    # Chromatic f0
    low_midi = note_to_midi("A3")
    high_midi = note_to_midi(hz_to_note(sample_rate // 2))
    level = high_midi - low_midi
    midi = np.linspace(low_midi, high_midi, level + 1)
    return midi_to_hz(midi)


def get_chromatic_filterbank(sample_rate, n_fft):
    fft_bins = np.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    chromatic_bins = get_chromatic_f0(sample_rate)

    f_diff = np.diff(chromatic_bins)
    ramps = np.subtract.outer(chromatic_bins, fft_bins)

    weights = np.zeros((len(chromatic_bins) - 2, int(1 + n_fft // 2)))
    for i in range(len(chromatic_bins) - 2):
        lower = -ramps[i] / f_diff[i]
        upper = ramps[i + 2] / f_diff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return torch.tensor(weights.astype("float32"))


def get_chroma_filterbank(sample_rate, n_fft):
    return torch.tensor(filters.chroma(sr=sample_rate, n_fft=n_fft).astype("float32"))


class STFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=240):
        super(STFT, self).__init__()

        # spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # STFT
        spec = self.spec(waveform)

        # amplitude to db
        log_spec = self.amplitude_to_db(spec)

        return log_spec


class MelSTFT(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128):
        super(MelSTFT, self).__init__()

        # spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # MelSTFT
        melspec = self.melspec(waveform)

        # amplitude to db
        log_melspec = self.amplitude_to_db(melspec)

        return log_melspec


class MFCC(nn.Module):
    def __init__(
        self, sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128, n_mfcc=12
    ):
        super(MFCC, self).__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
        )

    def forward(self, waveform):
        return self.mfcc(waveform)[:, 1:, :]


class ChromaticSTFT(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240):
        super(ChromaticSTFT, self).__init__()

        # spectrogram
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # get filterbank
        self.register_buffer(
            "chromatic_fb", get_chromatic_filterbank(sample_rate, n_fft)
        )

    def forward(self, waveform):
        # STFT
        spec = self.stft(waveform)

        # chromatic filterbank
        chromatic_spec = torch.matmul(
            spec.transpose(1, 2), self.chromatic_fb.T
        ).transpose(1, 2)

        # amplitude to db
        log_chromatic_spec = self.amplitude_to_db(chromatic_spec)

        return log_chromatic_spec


class Chromagram(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240):
        super(Chromagram, self).__init__()

        # spectrogram
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length
        )

        # get filterbank
        self.register_buffer("chroma_fb", get_chroma_filterbank(sample_rate, n_fft))

    def forward(self, waveform):
        # STFT
        spec = self.stft(waveform)

        chromagram = torch.matmul(spec.transpose(1, 2), self.chroma_fb.T).transpose(
            1, 2
        )

        return chromagram
