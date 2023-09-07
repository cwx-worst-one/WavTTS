import librosa
import numpy as np
import torch
from einops import rearrange
from torch import nn


class SubbandProjection(nn.Module):
    def __init__(self, bandwidth, out_dim):
        super(SubbandProjection, self).__init__()
        self.layer_norm = nn.LayerNorm(bandwidth)
        self.fc = nn.Linear(bandwidth, out_dim)

    def forward(self, x):
        """
        Input:
            x (torch.Tensor): a subband spectrogram (batch, frequency, time)
        Output:
            x (torch.Tensor): a projected subband embedding (batch, emb, time)
        """
        x = rearrange(x, "b f t -> b t f")
        x = self.layer_norm(x)
        x = self.fc(x)
        x = rearrange(x, "b t f -> b f t")
        return x


class NeuralFilterbank(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, n_filterbank=128, out_dim=64):
        super(NeuralFilterbank, self).__init__()

        self.bandwidth_indices = self.get_bandwidth_indices(
            sample_rate, n_fft, n_filterbank
        )
        self.projection_modules = self.get_projection_layers(out_dim)

    def get_bandwidth_indices(self, sample_rate, n_fft, n_filterbank):
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_filterbank
        )
        bandwidth_indices = [np.where(row > 0)[0] for row in mel_basis]
        return bandwidth_indices

    def get_projection_layers(self, out_dim):
        projection_modules = nn.ModuleList([])
        for indices in self.bandwidth_indices:
            projection_modules.append(SubbandProjection(len(indices), out_dim))
        return projection_modules

    def forward(self, spec):
        """
        Input:
            spec (torch.Tensor): input spectrogram (batch, frequency, time)
        Output:
            emb (torch.Tensor): projected embedding (batch, channel, frequency, time)
        """
        emb = []
        for indices, layer in zip(self.bandwidth_indices, self.projection_modules):
            emb.append(layer(spec[:, indices, :]))
        return rearrange(torch.stack(emb), "f b c t -> b c f t")
