import math

import scipy.signal as sig
import torch
import torch.nn as nn


class EnvelopeDetector(nn.Module):
    def __init__(self, sample_rate, response_time_ms=50, lowpass_kernel_length=100):
        super().__init__()
        cutoff = 1.0 / response_time_ms
        kernel = torch.tensor(
            sig.firwin(numtaps=lowpass_kernel_length, cutoff=cutoff, fs=sample_rate),
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel)

    def forward(self, x, *args, **kwargs):
        kernel = self.kernel.expand([1, x.shape[1], self.kernel.shape[0]])
        envelope = nn.functional.conv1d(x.abs(), kernel, padding="same")
        return envelope.mean(dim=1, keepdim=True)


def calculateEnvelope(shape, peak_time):
    t = torch.arange(shape[2]) / shape[2]
    envelope = math.exp(1) / peak_time * t * torch.exp(-t / peak_time)
    return envelope.expand(shape)
