# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/expert.py ]
#   Synopsis     [ the baseline wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import yaml
import torch
from torch import nn 
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .extracter import get_extracter
from .preprocessor import get_preprocessor

SAMPLE_RATE = 16000


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi or torchaudio preprocessor
    Support: spectrogram, fbank, mfcc, mel, linear
    """

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        assert "offline" in self.config

        if "offline" in self.config:
            if self.config['offline']['use_lookup']:
                self.lookup_emb = nn.Embedding(
                    self.config['offline']['vocab_size'], self.config['offline']['input_dim'])
            self.downsample_rate = self.config['offline']['downsample_rate']
            self.input_dim = self.config['offline']['input_dim']
    

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))
        return feats

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def get_input_dim(self) -> int:
        return self.input_dim

    def _preprocessor_forward(self, wavs):
        wav_lengths = [len(wav) for wav in wavs]

        feats = pad_sequence(wavs, batch_first=True)
        feats = feats.unsqueeze(
            1
        )  # (batch_size, audio_len) -> (batch_size, 1, audio_len)
        feats = self.extracter(feats)[0]

        ratio = len(feats[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        feats = [f[:l] for f, l in zip(feats, feat_lengths)]
        return feats

    def forward(self, x):
        assert "offline" in self.config
        if "offline" in self.config:  
            if self.config['offline']['use_lookup']:
                x = [xi.to(dtype=torch.int)+1 for xi in x]  # shift 1 (0 as padding value)
                feats_len = [xi.shape[0] for xi in x]
                padded_x = pad_sequence(x, batch_first=True)
                padded_feats = self.lookup_emb(padded_x)
                return [f[:l] for f, l in zip(padded_feats, feats_len)]
                # feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
            else:
                feats = [xi for xi in x]
                return feats
                
                feats_len = [feat.shape[0] for feat in feats]
                padded_feats = pad_sequence(feats, batch_first=True)
                return [f[:l] for f, l in zip(padded_feats, feats_len)]
        
        
        # return {
        #     "last_hidden_state": padded_feats,
        #     "hidden_states": [padded_feats],
        # }
