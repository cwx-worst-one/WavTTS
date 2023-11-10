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

from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform
from recipes.umm.models.umm_mk3 import (Conv2dUpsampling, Transpose, WNConv1d, 
    EMAVectorQuantizerEntropy, EMAVectorQuantizer,
    FiniteScalarQuantizer, LookupFreeQuantizer, ClusteredVectorQuantizer)

from samantha.utils.hparams import DotDict

from mariana.models.audio.usm_encoder import UsmEncoder
from mariana.data.audio.transforms import KaldiFbank, CMVN
from mariana.models.audio.infer_utils import PantherInfer, replace_with_panther_conformer_layer
from mariana.models.audio.misc import rnnt_transpose



class USMStage2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_transform = SpeechTransform(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=0,
            f_max=config.sample_rate // 2,
        )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)
        self.fbank_fn = KaldiFbank(dither=0.0, out_numpy=False, device='cuda')
        self.cmvn_fn = CMVN(key="fbank",
                            cmvn_mean=np.load('recipes/datasets/mcc/usm_mean.npy'),
                            cmvn_var=np.load('recipes/datasets/mcc/usm_var.npy'))
        # model define
        usm_config = DotDict(config.usm_config)
        self.audio_encoder = UsmEncoder(usm_config.network)
        self.mel_head = Conv2dUpsampling(config.hidden_size, config.n_mels)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2)

    def forward(self, input_dict, return_hidden_states=False):
        feature = input_dict["fbank"]
        src_mask = input_dict["src_mask"]
        ### one forward ###
        # hidden_states = self.audio_encoder(feature, src_mask, is_training=self.training)
        ### Let's do it step by step ###
        # copy from `mariana.models.audio.usm_encoder.py`
        # copy from `mariana.models.audio.conformer.py`
        front_end_out, backbone_mask, frontend_shape = self.audio_encoder.frontend(feature, src_mask)
        conformers = self.audio_encoder.acoustic_backbone_module
        all_hidden_states = []
        conformer_input = conformers.pos_enc(front_end_out)
        if backbone_mask is None:
            conformer_mask = None
        else:
            conformer_mask = backbone_mask.unsqueeze(1)
        attn_weights = None
        for i, layer in enumerate(conformers.encoders):
            conformer_input, conformer_mask = layer(
                [conformer_input, conformer_mask], is_training=True
            )
            if return_hidden_states:
                if i != len(conformers.encoders) - 1 or not conformers.normalize_before:
                    all_hidden_states.append(conformer_input)
        if isinstance(conformer_input, (tuple, list)):
            conformer_input = conformer_input[0]
        if conformers.normalize_before:
            conformer_input = conformers.after_norm(conformer_input)
            if return_hidden_states:
                all_hidden_states.append(conformer_input)
        # return conformer_input
        with torch.cuda.amp.autocast(enabled=False):
            hidden_states = conformer_input
            mel_out = self.mel_head(hidden_states)
            ctc_out = self.ctc_head(hidden_states)
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict = {"mel_out": mel_out, "ctc_out": ctc_out,
                     "f0_out": f0_vuv_out[:,:,0:1], "vuv_out": f0_vuv_out[:,:,1:]}
            
            if return_hidden_states:
                all_hidden_states = [h[0] if isinstance(h, (list, tuple)) else h for h in all_hidden_states]
                output_dict["hidden_states"] = all_hidden_states
        return output_dict

    @torch.no_grad()
    def extract_features(self, wavs, dtype=torch.float32):
        is_amp = (dtype in [torch.float16, torch.bfloat16])
        input_dict = self.preprocessing(wavs)
        with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
            out_dict = self.forward(input_dict, return_hidden_states=True)
        return out_dict["hidden_states"]

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        assert x.dtype == torch.float32
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        # kaldif fbank setting, 400 for 25ms, 160 for 10ms
        x_pad = F.pad(x, ((400 - 160) // 2, (400 - 160) // 2), mode='reflect')
        fbank = self.fbank_fn({"waveform": x_pad.unsqueeze(1) * 32768.0})["fbank"]
        input_dict["fbank"] = fbank
        input_dict["src_mask"] = torch.ones_like(fbank[:, :, 0])
        input_dict = self.cmvn_fn(input_dict)
        
        return input_dict



class WSUSMStage3_V1_6_1(USMStage2):
    def __init__(self, config):
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

        if config.fix_layers:
            self.layer_to_fix = self.fix_parameters_and_bn()
        else:
            self.layer_to_fix = None

        self.weights = nn.Parameter(torch.zeros(self.config.vq_layer_idx))

    def fix_parameters_and_bn(self):
        conformers = self.audio_encoder.acoustic_backbone_module
        layer_to_fix = []
        for i, layer in enumerate(conformers.encoders):
            if i == self.config.vq_layer_idx:
                break
            else:
                for p in layer.parameters():
                    p.requires_grad = False
                layer_to_fix.append(layer)
        return layer_to_fix

    def _weighted_sum(self, feature):
        assert self.config.vq_layer_idx == len(feature)
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.config.vq_layer_idx, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(self, input_dict, return_hidden_states=False, return_vq_ids=False):
        feature = input_dict["fbank"]
        src_mask = input_dict["src_mask"]
        ### one forward ###
        # hidden_states = self.audio_encoder(feature, src_mask, is_training=self.training)
        ### Let's do it step by step ###
        # copy from `mariana.models.audio.usm_encoder.py`
        # copy from `mariana.models.audio.conformer.py`
        front_end_out, backbone_mask, frontend_shape = self.audio_encoder.frontend(feature, src_mask)
        conformers = self.audio_encoder.acoustic_backbone_module
        all_hidden_states = []
        conformer_input = conformers.pos_enc(front_end_out)
        if backbone_mask is None:
            conformer_mask = None
        else:
            conformer_mask = backbone_mask.unsqueeze(1)
        attn_weights = None
        hidden_states_list = []
        for i, layer in enumerate(conformers.encoders):
            if i == self.config.vq_layer_idx:
                with torch.cuda.amp.autocast(enabled=False):
                    vq_inputs = self._weighted_sum(hidden_states_list)
                    vq_inputs = self.vq_proj_in(vq_inputs)
                    if self.config.get("vq_proj_noise", 0) > 0:
                        noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                            0
                        ) / self.config.vq_proj_noise
                        vq_inputs = (
                            vq_inputs + torch.randn_like(vq_inputs) * noise_scale
                        )
                        self.cnt.add_(1)
                    if self.config.get("vq_type", None) == "FSQ":
                        vq_embs, vq_ids = self.vq(vq_inputs)
                        vq_loss = None
                    elif self.config.get("vq_type", None) == "EMAEntropy":
                        vq_embs, vq_ids, vq_loss = self.vq(vq_inputs, e_scale=1.0 if self.cnt < 30_000 else 0.0)
                    else:
                        vq_embs, vq_ids, vq_loss = self.vq(vq_inputs)
                    if return_vq_ids:
                        return vq_ids
                    vq_inputs = self.vq_proj_out(vq_embs)
                    conformer_input = (vq_inputs,) + conformer_input[1:] if isinstance(conformer_input, (list, tuple)) else vq_inputs
                    conformer_input, conformer_mask = layer(
                        [conformer_input, conformer_mask], is_training=True
                    )
            elif i < self.config.vq_layer_idx:
                conformer_input, conformer_mask = layer(
                    [conformer_input, conformer_mask], is_training=True
                )
                hidden_states_list.append(conformer_input[0] if isinstance(conformer_input, (list, tuple)) else conformer_input)
            else:
                conformer_input, conformer_mask = layer(
                    [conformer_input, conformer_mask], is_training=True
                )
            if return_hidden_states:
                if i != len(conformers.encoders) - 1 or not conformers.normalize_before:
                    all_hidden_states.append(conformer_input)
        if isinstance(conformer_input, (list, tuple)):
            conformer_input = conformer_input[0]
        if conformers.normalize_before:
            conformer_input = conformers.after_norm(conformer_input)
            if return_hidden_states:
                all_hidden_states.append(conformer_input)
        # return conformer_input
        hidden_states = conformer_input
        with torch.cuda.amp.autocast(enabled=False):
            mel_out = self.mel_head(hidden_states)
            ctc_out = self.ctc_head(hidden_states)
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict = {"mel_out": mel_out, "ctc_out": ctc_out,
                     "f0_out": f0_vuv_out[:,:,0:1], "vuv_out": f0_vuv_out[:,:,1:],
                    "vq_ids": vq_ids, "vq_loss": vq_loss,}

            
            if self.config.get("vq_proj_noise", False):
                output_dict.update(noise_scale=noise_scale)
            
        if return_hidden_states:
            all_hidden_states = [h[0] if isinstance(h, (list, tuple)) else h for h in all_hidden_states]
            output_dict["hidden_states"] = all_hidden_states
        return output_dict

    @torch.no_grad()
    def wav2token(self, wav, dtype=torch.float32):
        is_amp = (dtype in [torch.float16, torch.bfloat16])
        input_dict = self.preprocessing(wav)
        with torch.cuda.amp.autocast(enabled=is_amp, dtype=dtype):
            vq_ids = self.forward(input_dict, return_vq_ids=True)
        return vq_ids
