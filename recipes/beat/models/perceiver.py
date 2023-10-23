import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

from recipes.beat.models.networks import FeedForward, Preprocess
from recipes.beat.models.spectnt import ResFrontEnd
from samantha.core import BaseStage


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# Perceiver block
class PerceiverBlock(nn.Module):
    def __init__(
        self,
        num_fct,
        depth,  # [# of spec, # of temp]
        spec_dim,
        temporal_dim,
        temporal_heads,
        dim_temporal_head,
        block_ratio,
        attn_dropout,
        ff_dropout,
    ):
        super(PerceiverBlock, self).__init__()

        self.num_fct = num_fct
        assert len(block_ratio) == 3
        self.n_cross_attn, self.n_latent_attn, self.n_time_attn = block_ratio

        layers = nn.ModuleList([])
        for i in range(depth):
            sub_layer = nn.ModuleList([])

            # Spec-time cross attn
            cross_attns = nn.ModuleList([])
            for j in range(self.n_cross_attn):
                cross_attn = nn.ModuleList([])

                cross_attn.append(
                    PreNorm(
                        temporal_dim,
                        Attention(
                            query_dim=temporal_dim,
                            context_dim=spec_dim,
                            heads=temporal_heads,
                            dim_head=dim_temporal_head,
                            dropout=attn_dropout,
                        ),
                        context_dim=spec_dim,
                    )
                )
                cross_attn.append(
                    PreNorm(
                        temporal_dim, FeedForward(dim=temporal_dim, dropout=ff_dropout)
                    )
                )
                cross_attns.append(cross_attn)
            sub_layer.append(cross_attns)

            # Freq latent attn
            latent_attns = nn.ModuleList([])
            for j in range(self.n_latent_attn):
                latent_attn = nn.ModuleList([])
                latent_attn.append(
                    PreNorm(
                        temporal_dim,
                        Attention(
                            query_dim=temporal_dim,
                            heads=temporal_heads,
                            dim_head=dim_temporal_head,
                            dropout=attn_dropout,
                        ),
                    )
                )
                latent_attn.append(
                    PreNorm(
                        temporal_dim, FeedForward(dim=temporal_dim, dropout=ff_dropout)
                    )
                )
                latent_attns.append(latent_attn)
            sub_layer.append(latent_attns)

            # time attns, iterate thorugh each block and each latent
            time_attns = nn.ModuleList([])
            for j in range(self.n_time_attn):
                _time_attns = nn.ModuleList([])
                for k in range(self.num_fct):
                    time_attn = nn.ModuleList([])
                    time_attn.append(
                        PreNorm(
                            temporal_dim,
                            Attention(
                                query_dim=temporal_dim,
                                heads=temporal_heads,
                                dim_head=dim_temporal_head,
                                dropout=attn_dropout,
                            ),
                        )
                    )
                    time_attn.append(
                        PreNorm(
                            temporal_dim,
                            FeedForward(dim=temporal_dim, dropout=ff_dropout),
                        )
                    )
                    _time_attns.append(time_attn)

                time_attns.append(_time_attns)
            sub_layer.append(time_attns)

            layers.append(sub_layer)

        self.layers = layers

    def forward(self, b, spec_emb, temporal_emb):
        for idx, sub_layers in enumerate(self.layers):
            cross_attns, latent_attns, time_attns = sub_layers
            # spec-time corss attn
            for cross_attn in cross_attns:
                _attn, _ff = cross_attn
                temporal_emb = _attn(temporal_emb, context=spec_emb) + temporal_emb
                temporal_emb = _ff(temporal_emb) + temporal_emb

            # latent attn
            for latent_attn in latent_attns:
                _attn, _ff = latent_attn
                temporal_emb = _attn(temporal_emb) + temporal_emb
                temporal_emb = _ff(temporal_emb) + temporal_emb

            temporal_emb = rearrange(temporal_emb, "(b t) n d -> b t n d", b=b)

            # time attn
            for time_attn in time_attns:
                out_temporal_emb = []
                for idx_2, _time_attn in enumerate(time_attn):
                    _temporal_emb = temporal_emb[:, :, idx_2, :]
                    _attn, _ff = _time_attn

                    _temporal_emb = _attn(_temporal_emb) + _temporal_emb
                    _temporal_emb = _ff(_temporal_emb) + _temporal_emb
                    out_temporal_emb.append(
                        rearrange(_temporal_emb, "b t d -> b t 1 d")
                    )

                temporal_emb = torch.cat(out_temporal_emb, dim=2)

            temporal_emb = rearrange(
                temporal_emb, "b t n d -> (b t) n d", n=self.num_fct
            )

        return temporal_emb


# main class
class BeatPerceiverModelStage(BaseStage):
    def __init__(
        self,
        sample_rate: int,
        sample_len: int,
        hop_len: int,
        n_layers: int,
        spec_dim: int,
        temporal_dim: int,
        temporal_heads: int,
        resnet_pools: int,
        n_fft: int,
        semitone_scale: int,
        freq_pool_size: int,
        time_pool_size: int,
        num_fct: int,
        input_feature: str = "hcqt",
        ff_dropout=0.3,
        attn_dropout=0.3,
        n_harmonic=6,
        learn_bw="only_Q",
        takes=["audio", "aug_hop_size"],
        provides=["beat_pred", "tempo_pred"],
        serialize_opts=None,
    ):
        super().__init__(takes, provides, serialize_opts)

        self.total_length = int(sample_rate / hop_len * sample_len)
        self.n_timesteps = int(int(sample_rate * sample_len / hop_len) / time_pool_size)
        self.num_fct = num_fct
        if input_feature == "hcqt":
            n_freq = 64 * semitone_scale
        elif input_feature == "mel":
            n_freq = 128

        # input
        semitone_scale = semitone_scale
        self.dropout = nn.Dropout2d(p=ff_dropout)
        self.preprocess = Preprocess(
            sample_rate,
            n_fft,
            n_harmonic,
            semitone_scale,
            learn_bw,
            hop_len,
            input_feature,
        )

        # front end
        input_channel = n_harmonic
        emb_params = int(n_freq / freq_pool_size)
        self.input_layer = ResFrontEnd(spec_dim, input_channel, resnet_pools)

        # perceiver block
        self.merge_batch_time = nn.Sequential(Rearrange("b c f t -> (b t) f c"))
        self.freq_pos_emb = nn.Parameter(torch.randn(emb_params, spec_dim))
        self.temporal_pos_emb = nn.Parameter(
            torch.randn(self.n_timesteps, temporal_dim)
        )
        self.freq_query = nn.Parameter(torch.randn(self.num_fct, temporal_dim))

        dim_temporal_head = temporal_dim // temporal_heads
        self.perceiver = PerceiverBlock(
            num_fct,
            n_layers,
            spec_dim,
            temporal_dim,
            temporal_heads,
            dim_temporal_head,
            [1, 2, 2],
            attn_dropout,
            ff_dropout,
        )

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
        b, c, f, t = x.shape

        # batch, conv_ndim, time, freq
        spec_emb = self.merge_batch_time(x)
        spec_emb = spec_emb + repeat(self.freq_pos_emb, "f d -> b f d", b=b * t)

        # Create temporal emb
        temporal_pos_emb = rearrange(self.temporal_pos_emb, "t d -> t 1 d")
        temporal_pos_emb = repeat(
            temporal_pos_emb, "t 1 d -> (b t) n d", b=b, n=self.num_fct
        )
        temporal_emb = repeat(self.freq_query, "n d -> b n d", b=b * t)
        temporal_emb = temporal_emb + temporal_pos_emb

        temporal_emb = self.perceiver(b, spec_emb, temporal_emb)
        temporal_emb = rearrange(temporal_emb, "(b t) n d -> b t (n d)", b=b)

        # beat head
        oup["emb"] = temporal_emb

        return oup
