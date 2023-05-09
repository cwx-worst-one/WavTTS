from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange

from recipes.audio_lm.requires.mulan.mulan_infer import LitMuLanModule, mulan_rvq_indexs

MULAN_247_MODEL_PATHS = (
    "/mnt/bn/audio-diffusion/pretrained_models/mulan/v247/mulan_247.pt",
    "/mnt/bn/audio-diffusion/pretrained_models/mulan/v247/kmeans_minibatch_codebook-mulan249-1024x12.npy",  # noqa
)


class MulanModel(pl.LightningModule):
    def __init__(
        self,
        mulan_model_fp: str = MULAN_247_MODEL_PATHS[0],
        sample_rate: int = 24000,
        music_train_window_seconds: int = 10,
        shift_music_seconds: int = 5,
    ):
        super().__init__()
        self.mulan_model = LitMuLanModule.load_from_checkpoint(mulan_model_fp)
        self.sample_rate = sample_rate
        self.music_train_window_seconds = music_train_window_seconds
        self.shift_music_seconds = shift_music_seconds

    def post_init(self, device) -> None:
        # TODO: replace these with register_buffer so they are automatically cast
        for k, v in self.mulan_model.music_encoder.feat_extract.items():
            self.mulan_model.music_encoder.feat_extract[k] = v.to(device)

    def text_to_token_ids(self, text: str, device: str):
        return self.mulan_model.tokenizer(text, return_tensors="pt").to(device)

    def embed(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], data_type: str
    ) -> torch.Tensor:
        if data_type == "music":
            with torch.autocast(device_type="cuda", enabled=False):
                if x.ndim == 3:
                    x = x.squeeze(dim=1)
                x = x.float()

                max_model_samples = self.sample_rate * self.music_train_window_seconds
                if x.shape[1] > max_model_samples:
                    b = x.shape[0]
                    x = x.unfold(
                        1,
                        max_model_samples,
                        self.sample_rate * self.shift_music_seconds,
                    )
                    x = rearrange(x, "b n t -> (b n) t", b=b)
                    x = self.mulan_model.music_encoder(x)
                    x = rearrange(x, "(b n) t -> b n t", b=b)
                    x = x.mean(dim=1)
                    return F.normalize(x, p=2, dim=1)
                else:
                    return self.mulan_model.music_encoder(x)

        elif data_type == "text":
            return self.mulan_model.text_encoder(**x)
        else:
            raise NotImplementedError("Mulan only supports 'music' and 'text'")

    def forward(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], data_type: str
    ) -> torch.Tensor:
        return self.embed(x, data_type=data_type)


class QuantizedMulanModel(MulanModel):
    def __init__(
        self,
        mulan_model_fp: str = MULAN_247_MODEL_PATHS[0],
        mulan_centoids_fp: str = MULAN_247_MODEL_PATHS[1],
        n_codebooks: int = 12,
        codebook_size: int = 1024,
    ):
        super().__init__(mulan_model_fp)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        self.mulan_emb_dim = self.mulan_model.music_encoder.music_linear.out_features

        mulan_centers = np.load(mulan_centoids_fp)
        self.register_buffer("mulan_centers", torch.from_numpy(mulan_centers).float())

    def quantize(self, latents: torch.Tensor) -> torch.Tensor:
        mulan_tokens, _ = mulan_rvq_indexs(latents.squeeze(1), self.mulan_centers)
        return mulan_tokens

    def forward(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], data_type: str
    ) -> torch.Tensor:
        mulan_latents = self.embed(x, data_type=data_type)
        return self.quantize(mulan_latents)
