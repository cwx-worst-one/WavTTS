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
from recipes.umm.models.umm_mk3 import Conv2dUpsampling
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