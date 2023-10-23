from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from recipes.beat.models.networks import (
    Attention,
    Classifier,
    FeedForward,
    PreNorm,
    Preprocess,
)
from samantha.core import BaseStage


# Transformer frontend
class Res_2d_mp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, stride=None):
        super(Res_2d_mp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling, stride=stride)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        """
        :param x: (n_batch, n_channel, n_freq, n_frame)
        :return: (n_batch, n_channel, n_freq, n_frame)
        """
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))

        return out


class ResFrontEnd(nn.Module):
    def __init__(self, conv_ndim, nharmonics, pool_sizes=[(1, 2), (1, 1), (1, 1)]):
        super(ResFrontEnd, self).__init__()
        layers = nn.ModuleList([])
        layers.append(Res_2d_mp(nharmonics, conv_ndim, pooling=pool_sizes[0]))
        for i in range(1, len(pool_sizes)):
            layers.append(Res_2d_mp(conv_ndim, conv_ndim, pooling=pool_sizes[i]))
        self.layers = layers

    def forward(self, x):
        """
        :param x: (n_batch, n_channel, n_freq, n_frame)
        :return: (n_batch, n_channel, n_freq, n_frame)
        """
        # CNN
        for layer in self.layers:
            x = layer(x)
        return x


# main class
class SpecTNTModelStage(BaseStage):
    def __init__(
        self,
        sample_rate: int,
        sample_len: int,
        hop_len: int,
        n_layers: int,
        spec_dim: int,
        temporal_dim: int,
        temporal_heads: int,
        spec_heads: int,
        resnet_pools: int,
        n_fft: int,
        semitone_scale: int,
        freq_pool_size: int,
        time_pool_size: int,
        input_feature: str = "hcqt",
        ff_dropout=0.3,
        attn_dropout=0.3,
        n_harmonic=6,
        learn_bw="only_Q",
        takes=["audio", "aug_hop_size"],
        provides=["emb"],
        serialize_opts=None,
    ):
        super().__init__(takes, provides, serialize_opts)

        dim_spec_head = spec_dim // spec_heads
        dim_temporal_head = temporal_dim // temporal_heads
        semitone_scale = semitone_scale

        self.sample_rate = sample_rate
        self.sample_len = sample_len
        self.hop_len = hop_len
        self.total_length = int(sample_rate / hop_len * sample_len)
        input_channel = n_harmonic

        # input
        self.preprocess = Preprocess(
            sample_rate,
            n_fft,
            n_harmonic,
            semitone_scale,
            learn_bw,
            hop_len,
            input_feature,
        )
        if input_feature == "hcqt":
            n_freq = 64 * semitone_scale
        elif input_feature == "mel":
            n_freq = 128

        # front end
        self.emb_params = int(n_freq / freq_pool_size)
        self.input_layer = ResFrontEnd(spec_dim, input_channel, resnet_pools)

        self.n_timesteps = int(int(sample_rate * sample_len / hop_len) / time_pool_size)

        self.num_fct = 1

        self.freq_pos_emb = nn.Parameter(
            torch.randn(self.num_fct + self.emb_params, spec_dim)
        )
        self.temporal_emb = nn.Parameter(torch.randn(self.n_timesteps, temporal_dim))

        self.to_spectral_embedding = nn.Sequential(Rearrange("b c f t -> (b t) f c"))

        layers = nn.ModuleList([])
        for _ in range(n_layers):
            spec_to_temp = nn.Sequential(
                nn.LayerNorm(spec_dim),
                Rearrange("... n d -> ... (n d)"),
                nn.Linear(self.num_fct * spec_dim, temporal_dim),
            )
            get_fcts = nn.Sequential(
                nn.LayerNorm(temporal_dim),
                nn.Linear(temporal_dim, spec_dim * self.num_fct),
                Rearrange("b t (n d) -> (b t) n d", n=self.num_fct),
            )

            layers.append(
                nn.ModuleList(
                    [
                        get_fcts,
                        PreNorm(
                            spec_dim,
                            Attention(
                                dim=spec_dim,
                                heads=spec_heads,
                                dim_head=dim_spec_head,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(
                            spec_dim, FeedForward(dim=spec_dim, dropout=ff_dropout)
                        ),
                        spec_to_temp,
                        PreNorm(
                            temporal_dim,
                            Attention(
                                dim=temporal_dim,
                                heads=temporal_heads,
                                dim_head=dim_temporal_head,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(
                            temporal_dim,
                            FeedForward(dim=temporal_dim, dropout=ff_dropout),
                        ),
                    ]
                )
            )

        self.layers = layers
        self.dropout = nn.Dropout2d(p=ff_dropout)


    def forward(self, data):
        oup = {}

        inp = data["audio"]
        aug_hop_size = data["aug_hop_size"]
        inp = self.preprocess(inp, aug_hop_size)

        if inp.shape[-1] > self.total_length:
            inp = inp[..., : self.total_length]
        elif inp.shape[-1] < self.total_length:
            inp = torch.nn.functional.pad(
                inp, (0, self.total_length - inp.shape[-1]), "constant", 0
            )

        # (batch, channel, freq, time)
        x = self.input_layer(inp)[..., : self.n_timesteps]

        x = self.dropout(x)

        # batch, conv_ndim, time, freq
        b, c, f, t = x.shape
        spec_emb = self.to_spectral_embedding(x)
        spec_emb = F.pad(spec_emb, (0, 0, self.num_fct, 0), value=0)
        spec_emb = spec_emb + repeat(self.freq_pos_emb, "f d -> b f d", b=b * t)

        temporal_emb = repeat(self.temporal_emb, "t d -> b t d", b=b)

        for (
            get_fcts,
            spectral_attn,
            spectral_ff,
            spec_to_temp,
            temporal_attn,
            temporal_ff,
        ) in self.layers:
            fcts = get_fcts(temporal_emb)
            fcts = F.pad(fcts, (0, 0, 0, self.emb_params), value=0)
            spec_emb = spec_emb + fcts

            spec_att_out, spec_attn_map = spectral_attn(spec_emb)

            spec_emb = spec_att_out + spec_emb
            spec_emb = spectral_ff(spec_emb) + spec_emb

            temporal_emb_residual = spec_to_temp(
                spec_emb[:, 0 : self.num_fct]
            )  # 2 or 3
            temporal_emb_residual = rearrange(
                temporal_emb_residual, "(b t) d -> b t d", b=b
            )
            temporal_emb = temporal_emb + temporal_emb_residual
            temporal_att_out, _ = temporal_attn(temporal_emb)

            temporal_emb = temporal_att_out + temporal_emb
            temporal_emb = temporal_ff(temporal_emb) + temporal_emb

        # beat head
        oup["emb"] = temporal_emb

        return oup
