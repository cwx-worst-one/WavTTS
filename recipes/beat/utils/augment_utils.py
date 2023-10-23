import numpy as np
import torch
import random
from torchaudio_augmentations import (
    RandomApply,
    Gain,
    HighLowPass,
    Compose,
)

def time_augmentation(hop_length):
    return int(np.round(np.random.normal(hop_length, int(hop_length * 0.05), 1))[0])


class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.001, max_snr=1.0):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        noise = torch.distributions.normal.Normal(loc=0.0, scale=noise_std).sample(audio.shape).to(audio.device)

        return audio + noise


def get_augmentations(sample_rate):
    transforms = [
        RandomApply([Noise(min_snr=0.1, max_snr=0.1)], p=0.3),
        RandomApply([Gain()], p=0.2),
        RandomApply([HighLowPass(sample_rate=sample_rate)], p=0.3),
    ]
    transform = Compose(transforms=transforms)

    return transform
