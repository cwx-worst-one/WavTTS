import math
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, reduce, pack, unpack
from torch import Tensor, int32, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from recipes.umm.models.vocoder import BigVGAN
from recipes.umm.models.vq import EMAVectorQuantizer
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform
from recipes.umm.models.umm_mkii import (Stage2, ClusteredVectorQuantizer, 
            FiniteScalarQuantizer, LookupFreeQuantizer, EMAVectorQuantizerEntropy,
            Transpose, WNConv1d)


class Stage3(Stage2):
    def __init__(self, config, use_weighted_sum=True):
        super().__init__(config)
        if config.get("vq_type", None) == "CVQ":
            self.vq = ClusteredVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                distance=config.get("vq_distance", "cos"),
            )
        elif config.get("vq_type", None) == "FSQ":
            self.vq = FiniteScalarQuantizer(
                codebook_size=config.vq_codebook_size,
            )
        elif config.get("vq_type", None) == "LFQ":
            self.vq = LookupFreeQuantizer(
                codebook_size=config.vq_codebook_size,
            )
        elif config.get("vq_type", None) == "EMAEntropy":
            self.vq = EMAVectorQuantizerEntropy(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        else:
            self.vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        if config.get("vq_proj_norm", None) == "bn":
            self.vq_proj_in = nn.Sequential(
                Transpose(),
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False) if config.hidden_size != config.vq_codebook_dim else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False) if config.vq_codebook_dim != config.hidden_size else nn.Identity(),
            )
        else:
            self.vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        self.use_weighted_sum = use_weighted_sum
        self.layer_num = self.config.vq_layer_idx
        if self.use_weighted_sum:
            self.weights = nn.Parameter(torch.zeros(self.config.vq_layer_idx))

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature)
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(self, input_dict):
        feature = input_dict["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        hidden_states_list = []
        for i, layer in enumerate(self.encoder_layers):
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            if i <= self.config.vq_layer_idx-1:
                hidden_states_list.append(hidden_states)

            if i == self.config.vq_layer_idx-1:
                # weighted sum
                if self.use_weighted_sum:
                    hidden_states = self._weighted_sum(hidden_states_list)
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0)
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                hidden_states = self.vq_proj_out(vq_embs)

        mel_out = self.mel_head(hidden_states)
        ctc_out = self.ctc_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "ctc_out": ctc_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
        }
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        return output_dict

