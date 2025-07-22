import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import librosa

from recipes.umm2.transforms.speech import SpeechTransform
from recipes.umm2.models.base import BaseStage


# classes
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_flash_attn=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.use_flash_attn = use_flash_attn

    def forward(self, x, rotary_emb=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        use_checkpoint=True,
        use_flash_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_0": RMSNorm(dim),
                        "attn": Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                        ),
                        "norm_1": RMSNorm(dim),
                        "ff": FeedForward(dim, mlp_dim, dropout=dropout),
                    }
                )
            )
        self.use_checkpoint = use_checkpoint

    def _forward_checkpoint(self, x, rotary_emb=None):
        for layer in self.layers:
            if rotary_emb is not None:
                x = (
                    checkpoint(
                        layer["attn"], 
                        checkpoint(layer["norm_0"], x, use_reentrant=False), 
                        rotary_emb, 
                        use_reentrant=False,
                    )
                    + x
                )
            else:
                x = checkpoint(layer["attn"], x, use_reentrant=False) + x

            x = checkpoint(layer["ff"], checkpoint(layer["norm_1"], x, use_reentrant=False), use_reentrant=False) + x
        return x

    def _forward(self, x, rotary_emb=None):
        for layer in self.layers:
            if rotary_emb is not None:
                x = layer["attn"](layer["norm_0"](x), rotary_emb) + x
            else:
                x = layer["attn"](x) + x

            x = layer["ff"](layer["norm_1"](x)) + x
        return x

    def forward(self, x, rotary_emb=None):
        if self.use_checkpoint:
            x = self._forward_checkpoint(x, rotary_emb)
        else:
            x = self._forward(x, rotary_emb)
        return x


class BSTransformer_block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
        use_flash_attn: bool = False,
    ):
        super(BSTransformer_block, self).__init__()

        self.transform_t = Transformer(
            dim=input_dim,
            depth=1,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=input_dim * 4,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn,
        )
        self.transform_k = Transformer(
            dim=input_dim,
            depth=1,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=input_dim * 4,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn,
        )

    def forward(self, x, time_pos_emb=None, freq_pos_emb=None):
        b, t, k, c = x.shape
        # x: (bs, t, k, c)

        x = rearrange(x, "b t k c -> (b k) t c")
        # ((bs * k), t, c)

        if time_pos_emb is not None:
            x = self.transform_t(x, time_pos_emb)
        else:
            x = self.transform_t(x)

        x = rearrange(x, "(b k) t c -> (b t) k c", b=b)
        # (bs * t, k, c)

        if freq_pos_emb is not None:
            x = self.transform_k(x, freq_pos_emb)
        else:
            x = self.transform_k(x)

        x = rearrange(x, " (b t) k c -> b t k c", b=b)

        return x


def get_bandwidth_ranges(sample_rate=44100, n_fft=2047, n_filterbank=63):
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_filterbank)
    bw_idxs = [list(np.where(row > 0)[0]) for row in mel_basis]
    bw_idxs[0] = [0] + bw_idxs[0]
    bw_ranges = [[i[0], i[-1]] for i in bw_idxs]
    return bw_ranges


class BandSplit(nn.Module):
    def __init__(
        self,
        num_feature=128,
        channel_num=4,
        subspec_idxs=[],
        time_pool_length=4,
        use_avg_pool=True,
    ):
        """
        Input:
            x: (b, c, f, t)
        Output:
            x: (b, t, k, n)
        """
        super().__init__()
        self.channel_num = channel_num 
        self.subspec_idxs = subspec_idxs
        self.use_avg_pool = use_avg_pool
        self.time_pool_length = time_pool_length
        if self.use_avg_pool:
            self.time_pool = nn.AvgPool1d(time_pool_length, stride=time_pool_length)
        else:
            self.channel_num *= time_pool_length  # intend to put time_pool_length into channel_num

        self.mel_band_transforms = nn.ModuleList([])
        for subspec_idx in subspec_idxs:
            input_dim = (subspec_idx[-1] - subspec_idx[0] + 1) * self.channel_num 
            mel_band_transform = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("norm", RMSNorm(input_dim)),
                        ("ff", nn.Linear(input_dim, num_feature)),
                    ]
                )
            )
            self.mel_band_transforms.append(mel_band_transform)

    def forward(self, x):
        # x: (b, c, f, t)
        if self.use_avg_pool:
            c = x.shape[1]
            x = rearrange(x, "b c f t -> b (c f) t")
            x = self.time_pool(x)    # reduce t by time_pool_length
            x = rearrange(x, "b (c f) t -> b c f t", c=c) 
        else:
            x = x.unfold(-1, self.time_pool_length, self.time_pool_length)
            # (b, c, f, t, s)
            x = rearrange(x, "b c f t s -> b (c s) f t")
            # (b, c, f, t)

        if self.time_pool_length == 1:
            x = x[..., 0 : x.shape[-1] - 1]  # (b, c, f, t) remove the last time step due to stft windowing

        x = rearrange(x, "b c f t -> b t (f c)")
        outs = []
        for subspec_idx, mel_band_transform in zip(self.subspec_idxs, self.mel_band_transforms):
            sub_x = mel_band_transform(x[:, :, subspec_idx[0] * self.channel_num : (subspec_idx[-1] + 1) * self.channel_num])
            outs.append(sub_x)
        x = rearrange(torch.stack(outs), "k b t n -> b t k n")
        return x


class MelMaskEstimation(nn.Module):
    def __init__(
        self,
        num_feature=128,
        channel_num=4,
        subspec_idxs=[],
        use_checkpoint: bool = True,
    ):
        """
        Input:
            x: (b, t, k, n)
        Output:
            x: (b, c, t, f)
        """
        super().__init__()
        self.channel_num = channel_num  # (real + imag) * stereo

        self.subspec_idxs = subspec_idxs
        self.overlapped_subspec_idx_range = [subspec_idxs[1][0], subspec_idxs[-2][-1] + 1] # [3, 961]
        self.frequency_dim = subspec_idxs[-1][-1] + 1  # 1024
        self.use_checkpoint = use_checkpoint

        self.mask_estimations = nn.ModuleList([])
        for subspec_idx in subspec_idxs:
            out_dim = (subspec_idx[-1] - subspec_idx[0] + 1) * channel_num

            submask_estimation = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("norm", RMSNorm(num_feature)),
                        ("ff_0", nn.Linear(num_feature, 4 * num_feature)),
                        ("tanh", nn.Tanh()),
                        ("ff_1", nn.Linear(4 * num_feature, 2 * out_dim)),
                        ("glu", nn.GLU()),
                    ]
                )
            )
            self.mask_estimations.append(submask_estimation)

    def forward(self, x):
        # x: (b, t, k, n)

        out_x = torch.zeros((x.shape[0], x.shape[1], self.frequency_dim * self.channel_num)).to(x.dtype).to(x.device)
        #(b, t, 1024*4)
        for i, submask_estimation, subspec_idx in zip(range(len(self.subspec_idxs)), self.mask_estimations, self.subspec_idxs):
            if self.use_checkpoint:
                sub_x = checkpoint_sequential(submask_estimation, 4, x[:, :, i], use_reentrant=False)
            else:
                sub_x = submask_estimation(x[:, :, i])
            out_x[:, :, subspec_idx[0] * self.channel_num : (subspec_idx[-1] + 1) * self.channel_num] += sub_x

        out_x[:, :, self.overlapped_subspec_idx_range[0] : self.overlapped_subspec_idx_range[1]] /= 2
        x = rearrange(out_x, "b t (f c) -> b c t f", c=self.channel_num)
        # b c t f
        return x


class MLP_embed(nn.Module):
    def __init__(
        self,
        num_feature=128,
        mel_bands=32,
        out_dim=64,
        use_checkpoint: bool = True,
    ):
        """
        Input:
            x: (b, t, k, n)
        Output:
            x: (b, c, t, f)
        """
        super().__init__()
        self.out_dim = out_dim
        self.use_checkpoint = use_checkpoint

        self.mlp_embeds = nn.ModuleList([])
        for i in range(mel_bands):
            mlp_embed = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("norm", RMSNorm(num_feature)),
                        ("ff_0", nn.Linear(num_feature, 4 * num_feature)),
                        ("tanh", nn.Tanh()),
                        ("ff_1", nn.Linear(4 * num_feature, 2 * out_dim)),
                        ("glu", nn.GLU()),
                    ]
                )
            )
            self.mlp_embeds.append(mlp_embed)

    def forward(self, x):
        # x: (b, t, k, n)

        outs = []
        for i, mlp_embed in enumerate(self.mlp_embeds):
            if self.use_checkpoint:
                sub_x = checkpoint_sequential(mlp_embed, 4, x[:, :, i], use_reentrant=False)
            else:
                sub_x = mlp_embed(x[:, :, i])
            outs.append(sub_x)
        outs = torch.cat(outs, dim=-1)    # (b, t, k * out_dim)

        x = rearrange(outs, "b t (k out_dim) -> b t k out_dim", out_dim=self.out_dim)

        # b t k out_dim
        return x

class Spectrogram(nn.Module):
    def __init__(
        self,
        window_size: int = 2048,
        hop_size: int = 240,
        use_power_stft: bool = True,
    ):
        super().__init__()
        self.use_power_stft = use_power_stft
        if use_power_stft:
            self.spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=window_size,
                win_length=window_size,
                hop_length=hop_size,
                power=2,
                window_fn=torch.hann_window,
                pad=0,
                normalized=False,
                wkwargs=None,
            )
        else:
            self.spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=window_size,
                win_length=window_size,
                hop_length=hop_size,
                power=None,
                window_fn=torch.hann_window,
                center=True,
                pad_mode="reflect",
            )
        
    def forward(self, x):
        if self.use_power_stft:
            x = self.spectrogram(x)
            if len(x.shape) != 4:
                x = x.unsqueeze(1)    
        else:
            complex_spec = self.spectrogram(x)
            x = torch.cat((complex_spec.real, complex_spec.imag), dim=1)

        return x

class UMM(BaseStage):
    def __init__(
        self,
        config,       
        takes=["audio"],
        provides=["latent"],
        bypasses=[],
        lr_ratio=1.0,
        loss_weight=None,
        is_frozen=False,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen)
        
        self.config = config
        #self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.sampling_rate,)

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

        stft_channel = 1 if config.use_power_stft else 2
        self.stft = Spectrogram(
            window_size=config.n_fft,
            hop_size=config.hop_length,
            use_power_stft=config.use_power_stft,
        )
        
        # Band projection
        subspec_idxs = get_bandwidth_ranges(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft-1,  # 2047
            n_filterbank=config.mel_bands, 
        )
        self.multi_band_transform = BandSplit(
            num_feature=config.num_feature,
            channel_num=config.num_channels * stft_channel, 
            subspec_idxs = subspec_idxs,
            time_pool_length = int(config.sample_rate//config.hop_length//config.frame_rate),
            use_avg_pool=config.use_avg_pool,
        )

        self.rotary_emb_t = RotaryEmbedding(dim=int(config.num_feature / 8))
        self.rotary_emb_k = RotaryEmbedding(dim=int(config.num_feature / 8))

        self.transformer_stack = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            layer = BSTransformer_block(
                input_dim=config.num_feature,
                num_heads=8,
                dim_head=int(config.num_feature // 8),
                dropout=config.enforce_dropout,
                use_checkpoint=config.use_checkpoint,
                use_flash_attn=config.use_flash_attn,
            )
            self.transformer_stack.append(layer)

        self.multi_band_embed = MLP_embed(
            num_feature=config.num_feature,
            mel_bands=config.mel_bands,
            out_dim=config.out_num_feature,
            use_checkpoint=config.use_checkpoint,
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        return input_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, batch):
        wav = batch['audio'].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        input_dict = self.preprocessing(wav)
        return input_dict
    
    def get_latent(self, x_in):
        # (batch_size, input_channels, segment_samples)
        
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.stft(x_in)

        x = x.to(x_in.dtype) # (b, c, f, t)

        # Separate into subbands
        x = self.multi_band_transform(x)
        # (b, t, k, n)

        for i, layer in enumerate(self.transformer_stack):
            x = layer(x, self.rotary_emb_t, self.rotary_emb_k)
        # (b, t, k, n)

        x = self.multi_band_embed(x)
        # (b, t, k, num_out_feature)

        x = rearrange(x, "b t k n -> b t (k n)")
        # batch x time x feature

        return x

    def _compute(self, batch):
        wav = batch['audio'].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        mel = self.get_feature(wav)["mel"]
        latent = self.get_latent(wav)

        output_dict = {
            "audio": batch['audio'],
            "mel": mel,
            "latent": latent,
        }
       
        return output_dict

if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig(
        use_power_stft=True,
        use_avg_pool=False,
        hop_length=960,
        mel_bands=16,
        num_feature=128,
        use_checkpoint=True,
        use_flash_attn=True,
        enforce_dropout=0.1,
        out_num_feature=64,
    )
    dummy_input = {
        'audio': torch.randn(2, 1, 24000 * 30).to("cuda"),
        }

    model = UMM(config)
    model = model.to("cuda")
    output = model(dummy_input)
    
    print([(k, v.shape) for k, v in output.items()])
