import math
from typing import Optional

import torch
from dac import DAC
from dac.utils import download
from pytorch_lightning import LightningModule


class DACModel(LightningModule):

    n_quantizers = 9
    frame_rate = 87  # TODO
    codebook_size = 1024

    def __init__(self, src_sample_rate: int, target_sample_rate: int) -> None:
        super().__init__()
        self.src_sample_rate = src_sample_rate
        self.sample_rate = target_sample_rate

        if target_sample_rate == 16000:
            model_path = download(model_type="16khz")
        elif target_sample_rate == 24000:
            model_path = download(model_type="22khz")
        elif target_sample_rate == 44100:
            model_path = download(model_type="44khz")
        else:
            raise NotImplementedError(
                "This sample rate is not supported for DAC (16kHz/24kHz/44.1kHz)"
            )

        self.model = DAC.load(model_path).eval()

    def n_frames(self, n_seconds: int):
        length = n_seconds * self.sample_rate
        padded_samples = (
            math.ceil(length / self.model.hop_length) * self.model.hop_length - length
        )
        n_padded_seconds = (length + padded_samples) / self.sample_rate
        return math.ceil(n_padded_seconds * self.frame_rate)

    def preprocess(self, audio):
        return self.model.preprocess(audio, self.src_sample_rate)

    @torch.no_grad()
    def encode(
        self, audio: torch.Tensor, n_quantizers: Optional[int] = None
    ) -> torch.Tensor:
        self.model.eval()

        if audio.ndim == 3 and audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)

        audio = self.preprocess(audio)
        z, codes, latents, commitment_loss, codebook_loss = self.model.encode(
            audio, n_quantizers=n_quantizers
        )
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        z, _, _ = self.model.quantizer.from_codes(codes)
        return self.model.decode(z)

    def forward(
        self, audio: torch.Tensor, n_quantizers: Optional[int] = None
    ) -> torch.Tensor:
        return self.encode(audio, n_quantizers=n_quantizers)
