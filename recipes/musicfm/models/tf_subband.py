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


class SubbandMusicFM(nn.Module):
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
        codebook_size=512,
        codebook_size_mel=8192,
        sample_rate=24000,
        hop_length=240,
        n_fft=2047,
        n_filterbank=64,
        n_mels=128,
        n_subband=8,
        filterbank_dim=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=24,
        mask_hop=0.4,
        mask_prob=0.6,
        is_flash=True,
        stat_path=None,
        model_path=None,
    ):
        super(SubbandMusicFM, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.codebook_size = codebook_size
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
            n_mels * 4, codebook_dim, codebook_size_mel
        )  # mel spec
        for i, subband in enumerate(self.subband_indices):
            setattr(
                self,
                "quantizer_subband_%d" % i,
                RandomProjectionQuantizer(
                    len(subband) * 4, codebook_dim, codebook_size
                ),
            )

        # feature extractor
        self.preprocessor_spec = STFT(n_fft=n_fft)
        self.preprocessor_melspec = MelSTFT(n_fft=n_fft, n_mels=128)

        # trainable filterbank
        self.fb = TrainableFilterbank(
            n_fft=n_fft, n_filterbank=n_filterbank, out_dim=filterbank_dim
        )

        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            filterbank_dim, conv_dim, encoder_dim, n_filterbank
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

        self.conformer = Wav2Vec2ConformerEncoder(config)

        # projection
        self.linear = nn.Linear(
            encoder_dim, codebook_size * n_subband + codebook_size_mel
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

    def masking(self, x):
        """random masking of 400ms with given probability"""
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop)

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices

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

        # CNN
        x = self.conv(x)

        # W2V conformer
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits_dict = {
            "subband_%d"
            % i: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i in range(self.n_subband)
        }
        logits_dict["melspec"] = logits[:, :, self.n_subband * self.codebook_size :]
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
            out[key] = layer(x[key])
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

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # masking
        x, masked_indices = self.masking(x)

        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)
        print(losses["melspec"], accuracies["melspec"])

        return logits, hidden_emb, losses, accuracies
