import json

import librosa
import numpy as np
import torch
from einops import rearrange
from sami_ai_models.recipes.musicfm.modules.neural_filterbank import TrainableFilterbank
from torch import nn

from recipes.musicfm.modules.conv import Conv2dSubsampling
from recipes.musicfm.modules.features import STFT, MelSTFT
from recipes.musicfm.modules.freq_attention import FreqAttention
from recipes.musicfm.modules.random_quantizer import RandomProjectionQuantizer


class CausalMusicFM(nn.Module):
    """
    A Foundation Model for Music trained with Classic Features

    Input: 128-band trainable filterbank
    Frontend: frequency attention layer + 2-layer Residual convolution
    Backend: 24-layer Conformer
    Quantizer: multiple codebooks for subband spectrograms
    """

    def __init__(
        self,
        codebook_dim=16,
        codebook_size_sub=8192,
        codebook_size_mel=8192,
        sample_rate=24000,
        hop_length=240,
        n_fft=2047,
        n_filterbank=64,
        n_subband=16,
        freq_attn_dim=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=24,
        is_flash=True,
        stat_path=None,
        model_path=None,
    ):
        super(CausalMusicFM, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.codebook_size_sub = codebook_size_sub
        self.n_filterbank = n_filterbank
        self.n_subband = n_subband
        self.features = ["spec", "melspec"]

        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # subband indices
        self.subband_indices = self.get_subband_indices(sample_rate, n_fft, n_subband)

        # multiple random quantizers
        self.quantizer_melspec = RandomProjectionQuantizer(
            128 * 4, codebook_dim, codebook_size_mel
        )  # mel spec
        for i, subband in enumerate(self.subband_indices):
            setattr(
                self,
                "quantizer_subband_%d" % i,
                RandomProjectionQuantizer(
                    len(subband) * 4, codebook_dim, codebook_size_sub
                ),
            )

        # feature extractor
        self.preprocessor_spec = STFT(n_fft=n_fft)
        self.preprocessor_melspec = MelSTFT(n_fft=n_fft, n_mels=128)

        # trainable filterbank
        self.fb = TrainableFilterbank(
            n_fft=n_fft, n_filterbank=n_filterbank, out_dim=freq_attn_dim
        )

        # frequency attention
        self.freq_attention = FreqAttention(dim=freq_attn_dim)

        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            freq_attn_dim, conv_dim, encoder_dim, n_filterbank
        )

        # Conformer
        if is_flash:
            from recipes.musicfm.modules.flash_conformer import (
                Wav2Vec2ConformerConfig,
                Wav2Vec2ConformerEncoder,
            )
        else:
            from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
                Wav2Vec2ConformerConfig,
                Wav2Vec2ConformerEncoder,
            )
        config = Wav2Vec2ConformerConfig.from_pretrained(
            "facebook/wav2vec2-conformer-rope-large-960h-ft"
        )
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        self.conformer = Wav2Vec2ConformerEncoder(config, is_causal=True)

        # projection
        self.linear = nn.Linear(
            encoder_dim, codebook_size_sub * n_subband + codebook_size_mel
        )

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            self.load_state_dict(SS, strict=False)

    def get_subband_indices(self, sample_rate, n_fft, n_subband):
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_subband)
        bandwidth_indices = [np.where(row > 0)[0] for row in mel_basis]
        return bandwidth_indices

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())
            if precision == 16:
                out[key] = out[key].half()
        return out

    def encoder(self, x):
        """trainable filterbank + frequency attention + 2-layer conv + w2v-conformer"""
        # trainable filterbank
        x = self.fb(x)

        # frequency attention
        x = self.freq_attention(x)

        # CNN
        x = self.conv(x)[:, :-1, :]  # causal

        # W2V conformer
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits_dict = {
            "subband_%d"
            % i: logits[
                :, :, i * self.codebook_size_sub : (i + 1) * self.codebook_size_sub
            ]
            for i in range(self.n_subband)
        }
        logits_dict["melspec"] = logits[:, :, self.n_subband * self.codebook_size_sub :]
        return logits_dict, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key]
        return x

    @torch.no_grad()
    def group_subband(self, x):
        """group subband frequency bins to form subband features"""
        for i, indices in enumerate(self.subband_indices):
            x["subband_%d" % i] = x["spec"][:, indices, :]
        del x["spec"]
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])[:, 1:]  # causal
        return out

    def get_targets(self, x):
        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.group_subband(x)
        x = self.rearrange(x)
        target_tokens = self.tokenize(x)
        return target_tokens

    def get_predictions(self, x):
        # preprocessing
        x = self.preprocessing(x, features=["spec"])

        # normalization
        x = self.normalize(x)

        # encoding
        logits, hidden_emb = self.encoder(x["spec"])

        return logits, hidden_emb

    def get_latent(self, x, layer_ix=12):
        _, hidden_emb = self.get_predictions(x)
        emb = hidden_emb[layer_ix]
        return emb

    def get_loss(self, logits, target_tokens):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            _logits = rearrange(logits[key], "b t c -> (b t) c")
            _targets = rearrange(target_tokens[key], "b t -> (b t)")
            losses[key] = self.loss(_logits, _targets)
            accuracies[key] = (
                torch.sum(_logits.argmax(-1) == _targets) / _targets.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens)
        print(losses["melspec"], accuracies["melspec"])

        return logits, hidden_emb, losses, accuracies
