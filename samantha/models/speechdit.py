import logging
import math
from dataclasses import dataclass, field
from math import pi
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from samantha.criterion.masked_loss import sequence_mask
from samantha.models.based_ctiga_llama import LLaMa
from samantha.models.based_ctiga_llama import ModelArgs as LLamaArgs
from samantha.models.dpm_solver_pytorch import (
    DPM_Solver,
    NoiseScheduleVP,
    model_wrapper,
)
from samantha.models.ECAPA_TDNN_bias import ECAPA_TDNN_GN

logger = logging.getLogger(__name__)


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class FrontendEmbedding(nn.Module):
    def __init__(
        self,
        phone_embed_dim,
        tone_embed_dim,
        wordseg_embed_dim,
        out_dim,
        padding_idx=0,
        n_phone=200,
        n_tone=20,
        n_wordseg=8,
        lang_embed_dim=0,
        n_lang=0,
    ):
        super().__init__()
        self.phone_embedding = nn.Embedding(n_phone, phone_embed_dim)
        self.tone_embeddig = nn.Embedding(n_tone, tone_embed_dim)
        self.wordseg_embeddig = nn.Embedding(n_wordseg, wordseg_embed_dim)

        if lang_embed_dim > 0:
            self.lang_embedding = nn.Embedding(n_lang, lang_embed_dim)
            input_dim = (
                phone_embed_dim + tone_embed_dim + wordseg_embed_dim + lang_embed_dim
            )
        else:
            self.lang_embedding = None
            input_dim = phone_embed_dim + tone_embed_dim + wordseg_embed_dim
        self.out_linear = nn.Linear(input_dim, out_dim, bias=False)

    def forward(self, inputs):
        try:
            phone_emb = self.phone_embedding(inputs["phone"])
            tone_emb = self.tone_embeddig(inputs["tone"])
            wordseg_emb = self.wordseg_embeddig(inputs["word_seg"])
        except Exception:
            # in most cases, there will be a non-long tensor
            print("Incorrect phone, using fake one ====>")
            print(inputs["phone"])
            print(inputs["tone"])
            print(inputs["word_seg"])
            device = inputs["phone"].device
            phone_emb = self.phone_embedding(
                torch.LongTensor([[2, 302, 321, 1], [1, 1, 1, 1]]).to(device)
            )
            tone_emb = self.tone_embeddig(
                torch.LongTensor([[2, 12, 2, 1], [1, 1, 1, 1]]).to(device)
            )
            wordseg_emb = self.wordseg_embeddig(
                torch.LongTensor([[5, 5, 5, 1], [1, 1, 1, 1]]).to(device)
            )
        if self.lang_embedding is not None:
            lang_embed = self.lang_embedding(inputs["lang"])
            emb = torch.cat([phone_emb, tone_emb, wordseg_emb, lang_embed], dim=-1)
        else:
            emb = torch.cat([phone_emb, tone_emb, wordseg_emb], dim=-1)

        return self.out_linear(emb)


class RMSNorm(nn.Module):
    def __init__(self, dim, feat_dim=-1, eps=1e-5):
        super().__init__()
        self.rms = dim**-0.5
        self.feat_dim = feat_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, unscaled=False):
        norm = torch.norm(x, dim=self.feat_dim, keepdim=True) * self.rms
        if unscaled:
            return x / norm.clamp(min=self.eps)
        g = self.scale
        if self.feat_dim != -1:
            while g.ndim <= self.feat_dim:
                g = g[None]
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
        return x / norm.clamp(min=self.eps) * g


"""For VDiffusion"""


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class LogitNormalDistribution(Distribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean, self.std = mean, std

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        x = torch.from_numpy(np.random.lognormal(self.mean, self.std, num_samples)).to(
            device
        )
        return x / (1 + x)


class BernoulliDistribution(Distribution):
    def __init__(self, v1, v2):
        super().__init__()
        self.map = torch.tensor([v1, v2]).unsqueeze(0)

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        index = (torch.rand(num_samples, device=device) > 0.5).long()
        return self.map.repeat(num_samples, 1).to(device)[
            torch.arange(num_samples), index
        ]


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def Ts(t):
    """Builds a type template for a given type that accepts a list of instances"""
    return lambda *types: lambda: t(*[tp() for tp in types])


class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Repeat(m, times: int):
    ms = (m,) * times
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)


class TimeEmbedding(nn.Module):
    def __init__(self, modulation_features, num_layers: int = 2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features, bias=bias),
                nn.GELU(),
            ),
            times=num_layers,
        )

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


class PreNet(nn.Module):
    def __init__(self, in_dim, out_dim, conv_kernel=7, conv_padding=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
            nn.GELU(),
            RMSNorm(out_dim, feat_dim=1),
            nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
        )

    def forward(self, inputs):
        return self.net(inputs.transpose(1, 2)).transpose(1, 2)


class ResPostNet(nn.Module):
    def __init__(self, in_dim, out_dim, conv_kernel, conv_padding):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.net = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
            nn.GELU(),
            RMSNorm(out_dim, feat_dim=1),
            nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
        )

    def forward(self, inputs):
        x = self.linear(inputs)
        return self.net(x.transpose(1, 2)).transpose(1, 2) + x


def get_sigma(t: Tensor):
    angle = t * math.pi / 2
    sigma = torch.tan(angle)
    return sigma


def get_t_from_sigma(sigma):
    return torch.arctan(sigma) / math.pi * 2


@dataclass
class ModelArgs:
    # frontend
    use_phone_lang: bool = False
    phone_embed_dim: int = 512
    tone_embed_dim: int = 128
    wordseg_embed_dim: int = 128
    lang_embed_dim: int = 128
    n_wordseg: int = 8
    n_phone: int = 1000
    n_tone: int = 30
    n_lang: int = 8

    local_cond_dim: int = 512
    time_embed_dim: int = 256

    local_cond_project_type: str = "linear"  # conv
    local_cond_conv_kernel: int = 9
    local_cond_conv_padding: int = 4

    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    out_channels: int = 80
    max_seq_len: int = 8192

    causal: bool = False
    use_qk_norm: str = ""  # head, channel
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda: [-1, -1])
    window_type: str = "elemwise"  # elemwise, blockwise
    llama_provider: str = "ctiga"

    # speaker encoder
    prompt_mel_dim: int = 64
    spk_e_dim: int = 1024
    spk_embed_dim: int = 512

    # postnet
    postnet_type: str = "linear"  # conv
    postnet_kernel: int = 3

    target: str = "bn"
    prompt_feature: str = "bn"
    ctx_feature: str = "bn"
    in_channels: int = 64
    out_channels: int = 64
    use_textprefix: bool = True

    bias: bool = False
    target_type: str = "velocity"
    use_prompt: bool = True
    use_unet_style_skip_connect: bool = True

    # for uniform t
    min_t: float = 0.0
    max_t: float = 1.0
    t_sampling_type: str = "uniform"
    lognormal_mean: float = 0.0
    lognormal_std: float = 1.0

    p_max_t: float = 0.0

    use_seg_embed: bool = True
    use_bn_eos_bos: bool = True

    max_phone_len: int = 2000
    max_bn_len: int = 4000

    flashattn_version: str = "2.3"


class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = hp.min_t if hasattr(hp, "min_t") else 0
        self.max_t = hp.max_t if hasattr(hp, "max_t") else 1

        self.target_type = hp.target_type if hasattr(hp, "target_type") else "velocity"
        self.use_prompt = hp.use_prompt if hasattr(hp, "use_prompt") else True

        self.act_fn = nn.GELU()
        self.bias = hp.bias

        self.use_phone_lang = (
            hp.use_phone_lang if hasattr(hp, "use_phone_lang") else False
        )
        # text.
        if self.hp.use_textprefix:
            if self.use_phone_lang:
                self.frontend_embed = FrontendEmbedding(
                    hp.phone_embed_dim,
                    hp.tone_embed_dim,
                    hp.wordseg_embed_dim,
                    n_phone=hp.n_phone,
                    n_tone=hp.n_tone,
                    n_wordseg=hp.n_wordseg,
                    out_dim=hp.encoder_dim,
                    lang_embed_dim=hp.lang_embed_dim,
                    n_lang=hp.n_lang,
                )
            else:
                self.frontend_embed = FrontendEmbedding(
                    hp.phone_embed_dim,
                    hp.tone_embed_dim,
                    hp.wordseg_embed_dim,
                    n_phone=hp.n_phone,
                    n_tone=hp.n_tone,
                    n_wordseg=hp.n_wordseg,
                    out_dim=hp.encoder_dim,
                )

        # time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)

        # global speaker-embedding
        self.prompt_encoder = nn.Sequential(
            ECAPA_TDNN_GN(hp.prompt_mel_dim, hp.spk_e_dim, hp.spk_embed_dim),
            nn.Softsign(),
        )
        local_cond_in_channels = hp.out_channels + hp.spk_embed_dim

        if hp.local_cond_project_type == "linear":
            self.local_cond_project = nn.Linear(
                local_cond_in_channels, hp.local_cond_dim, bias=self.bias
            )
        elif hp.local_cond_project_type == "conv":
            self.local_cond_project = PreNet(
                local_cond_in_channels,
                hp.local_cond_dim,
                hp.local_cond_conv_kernel,
                hp.local_cond_conv_padding,
            )
        else:
            raise NotImplementedError

        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]

        # backbone
        llama_config = LLamaArgs(
            dim=hp.encoder_dim,
            n_layers=hp.encoder_n_layers,
            n_heads=hp.encoder_n_heads,
            n_kv_heads=hp.encoder_n_kv_heads,
            mlp_extend=hp.mlp_extend,
            causal=hp.causal,
            max_seq_len=hp.max_seq_len,
            use_window_mask=hp.use_window_mask,
            window_size=hp.window_size,
            window_type=hp.window_type,
            use_unet_style_skip_connect=hp.use_unet_style_skip_connect,
            use_qk_norm=hp.use_qk_norm,
            flashattn_version=hp.flashattn_version,
        )
        self.encoder = LLaMa(llama_config, hp.llama_provider)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim, bias=self.bias)
        self.prenet = nn.Linear(
            hp.time_embed_dim + hp.local_cond_dim, hp.encoder_dim, bias=self.bias
        )

        if hp.postnet_type == "linear":
            self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
        elif hp.postnet_type == "conv":
            self.postnet = ResPostNet(
                hp.encoder_dim,
                hp.out_channels,
                hp.postnet_kernel,
                hp.postnet_kernel // 2,
            )

        if hp.t_sampling_type == "uniform":
            self.sigma_distribution = UniformDistribution(
                vmin=self.min_t, vmax=self.max_t
            )
        elif hp.t_sampling_type == "logitnormal":
            self.sigma_distribution = LogitNormalDistribution(
                hp.lognormal_mean, hp.lognormal_std
            )
        else:
            raise NotImplementedError

        self.use_seg_embed = hp.use_seg_embed
        if hp.use_seg_embed:
            self.seg_embed = nn.Embedding(3, hp.encoder_dim, padding_idx=0)
            nn.init.trunc_normal_(self.seg_embed.weight, std=0.02, a=-0.04, b=0.04)

        if hp.use_bn_eos_bos:
            self.bn_eos_bos = nn.Parameter(torch.randn(2, hp.encoder_dim))

    def get_alpha_beta(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        if self.target_type == "recflow":
            alpha, beta = 1 - t, t
        else:
            angle = t * math.pi / 2
            alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def remove_text_prefix(self, x, out_len, text_lens, feat_lens):
        device = x.device
        new_x = torch.zeros(x.shape[0], out_len, x.shape[-1], device=device)

        T0 = out_len
        T1 = x.shape[1]
        indics0 = torch.arange(T0, device=device)[None, :]
        indics1 = torch.arange(T1, device=device)[None, :]

        mask0 = (indics0 < feat_lens[:, None]) & ((text_lens[:, None] + indics0) < T1)
        if self.hp.use_bn_eos_bos:
            mask1 = (text_lens[:, None] < indics1) & (
                indics1 < (text_lens[:, None] + feat_lens[:, None] + 1)
            )
        else:
            mask1 = (text_lens[:, None] <= indics1) & (
                indics1 < (text_lens[:, None] + feat_lens[:, None])
            )

        new_x[mask0] = x[mask1].to(dtype=new_x.dtype)
        return new_x

    def compute_forward_condition(self, inputs, sigmas):
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["frontend"])  # [B, T, 1024]
            text_lens = inputs["text_lens"]

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        B, device = inputs[ctx_feature].size(0), inputs[ctx_feature].device

        # speaker embedding
        spk_emb = self.prompt_encoder(inputs[prompt_feature])
        spk_emb[inputs["flag_drop"]] = 0
        spk_emb = spk_emb.unsqueeze(1).expand(-1, inputs[ctx_feature].shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([inputs[ctx_feature], spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        if sigmas is None:
            sigmas = self.sigma_distribution(num_samples=B, device=device).float()
        sigmas = torch.where(
            torch.rand_like(sigmas) < self.hp.p_max_t,
            torch.full_like(sigmas, self.hp.max_t),
            sigmas,
        )
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # time embedding
        time_emb = self.time_embedding(sigmas)
        time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)

        return text_embed, text_lens, local_cond, time_emb, alphas, betas

    def forward(self, inputs, sigmas=None, x_noisy=None, return_disc_feat=False):
        (text_embed, text_lens, local_cond, time_emb, alphas, betas) = (
            self.compute_forward_condition(inputs, sigmas)
        )

        B, device = inputs["bn_ctx"].size(0), inputs["bn_ctx"].device
        feat_lens = inputs["bn_lens"]
        x = inputs[self.hp.target]
        noise = torch.randn_like(x)
        if x_noisy is None:
            x_noisy = alphas * x + betas * noise
        residual = x_noisy
        if self.target_type == "velocity":
            target = alphas * noise - betas * x
        elif self.target_type == "x0":
            target = x
        elif self.target_type == "recflow":
            target = x - noise

        # concat condition.
        x_noisy = self.x_prenet(x_noisy) + self.prenet(
            torch.cat([time_emb, local_cond], dim=-1)
        )

        if self.hp.use_bn_eos_bos:
            inputs["text_mel_mask"] = F.pad(
                inputs["text_mel_mask"], (2, 0), "constant", 1
            )
            feat_lens = feat_lens + 2
            bn_bos = self.bn_eos_bos[0][None, None, :].expand(x_noisy.shape[0], -1, -1)
            x_noisy = torch.cat([bn_bos, x_noisy, torch.zeros_like(bn_bos)], dim=1)
            indics_x = torch.arange(x_noisy.shape[1], device=device)[None, :]
            mask_eos = indics_x == feat_lens[:, None] - 1
            x_noisy[mask_eos] = self.bn_eos_bos[1][None, None, :]

        # concat prefix-text.
        if self.hp.use_textprefix:
            T = inputs["text_mel_mask"].shape[1]
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full([B, T, C], 0, device=device, dtype=x_noisy.dtype)
            if self.use_seg_embed:
                seg_input = torch.zeros([B, T], device=x_noisy.device).long()

            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = (
                torch.arange(text_embed.shape[1], device=device)[None, :]
                < text_lens[:, None]
            )
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)
            if self.use_seg_embed:
                seg_input[mask_x] = 1

            mask_x = (
                (text_lens[:, None] <= indics_x)
                & (indics_x < (text_lens + feat_lens)[:, None])
                & (indics_x - text_lens[:, None] < T_feat)
            )
            mask_noisy = (
                torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            )
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)
            if self.use_seg_embed:
                seg_input[mask_x] = 2

            encoder_input = x_noisy_wtext
            seq_mask = inputs["text_mel_mask"]
            if self.use_seg_embed:
                seg_output = self.seg_embed(seg_input)
                encoder_input = encoder_input + seg_output
        else:
            encoder_input = x_noisy
            seq_mask = inputs[f"{self.hp.prompt_feature}_mask"]

        # encoder_input = torch.randn_like(encoder_input) # tmp

        encoder_out = self.encoder(
            encoder_input,
            encoder_input.shape[1],
            attention_mask=seq_mask,
            return_block_out=return_disc_feat,
        )
        if self.hp.use_bn_eos_bos:
            inputs["text_mel_mask"] = inputs["text_mel_mask"][:, 2:]
            feat_lens = feat_lens - 2

        if return_disc_feat:
            n_block = len(encoder_out[1]) // 2
            return encoder_out[1][n_block:]

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(
                B, x.shape[1], encoder_out.shape[-1], device=device
            )

            T0 = x.shape[1]
            T1 = encoder_out.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & (
                (text_lens[:, None] + indics0) < T1
            )
            if self.hp.use_bn_eos_bos:
                mask1 = (text_lens[:, None] < indics1) & (
                    indics1 < (text_lens[:, None] + feat_lens[:, None] + 1)
                )
            else:
                mask1 = (text_lens[:, None] <= indics1) & (
                    indics1 < (text_lens[:, None] + feat_lens[:, None])
                )

            pred_v_wotext[mask0] = encoder_out[mask1].to(dtype=pred_v_wotext.dtype)

            pred_v = pred_v_wotext

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            pred = pred_v

        ret_dict = {"pred_v": pred.transpose(1, 2), "target_v": target.transpose(1, 2)}

        return ret_dict

    def _forward(self, x, local_cond, text_embed, timesteps):
        residual = x
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(
            local_cond.shape[0], local_cond.shape[1], -1
        )
        x = self.x_prenet(x) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        if self.hp.use_bn_eos_bos:
            bn_bos = self.bn_eos_bos[0][None, None, :].expand(x.shape[0], -1, -1)
            bn_eos = self.bn_eos_bos[1][None, None, :].expand(x.shape[0], -1, -1)
            x = torch.cat([bn_bos, x, bn_eos], dim=1)

        if self.hp.use_textprefix:
            x = torch.cat([text_embed, x], dim=1)
            if self.use_seg_embed:
                seg_input = torch.zeros([1, x.shape[1]], device=x.device).long()
                seg_input[:, : text_embed.shape[1]] = 1
                seg_input[:, text_embed.shape[1] :] = 2
                seg_output = self.seg_embed(seg_input)
                x = x + seg_output

        pred_v = self.encoder(x, x.shape[1])

        if self.hp.use_textprefix:
            pred_v = pred_v[:, text_embed.shape[1] :, :]

        if self.hp.use_bn_eos_bos:
            pred_v = pred_v[:, 1:-1, :]

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            pred = pred_v

        return pred

    def euler_sample(self, timesteps, local_cond, text_embed, cfg_w=1.0):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        ts = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        ts = torch.flip(ts, [0])
        sigmas = repeat(ts, "i -> i b", b=1)

        for i in range(t):
            if cfg_w != 1:
                v_pred, v_pred_uncond = self._forward(
                    x, local_cond, text_embed, timesteps=sigmas[i]
                ).chunk(2)
                v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
            else:
                v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])
            dt = ts[i + 1] - ts[i]
            x = x + v_pred * dt
            # x = x - v_pred * dt
        return x

    def ddim_sample(
        self,
        timesteps,
        local_cond,
        text_embed,
        cfg_w=1.0,
        cfg_interval=[0, 1],
        inpaint_x=None,
        eta=0.0,
    ):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        # uniform t
        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # sigmas = torch.exp(torch.linspace(np.log(100), np.log(0.001), t + 1, device=device))
        # sigmas = get_t_from_sigma(sigmas)[:, None, None]
        # alphas, betas = self.get_alpha_beta(sigmas)

        for i in range(t):
            if self.target_type == "velocity":
                if (
                    cfg_w != 1
                    and sigmas[i] <= cfg_interval[1]
                    and sigmas[i] >= cfg_interval[0]
                ):
                    v_pred, v_pred_uncond = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    )

                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred

                # disable
                if inpaint_x is not None:
                    x_pred[:, : inpaint_x.shape[1], :] = inpaint_x
                    noise_pred[:, : inpaint_x.shape[1], :] = (
                        x[:, : inpaint_x.shape[1], :] - alphas[i] * inpaint_x
                    ) / betas[i]

            if eta > 0:
                sigma = (
                    eta
                    * betas[i + 1]
                    / betas[i]
                    * torch.sqrt(1 - (alphas[i] / alphas[i + 1]) ** 2)
                )
                noise = torch.randn_like(noise_pred)
                x = (
                    alphas[i + 1] * x_pred
                    + torch.sqrt(betas[i + 1] ** 2 - sigma**2) * noise_pred
                    + sigma * noise
                )
            else:
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x

    def heun_sample(self, timesteps, local_cond, text_embed, cfg_w=1.0):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        # uniform t
        self.max_t = 0.999
        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        for i in range(t):
            if self.target_type == "velocity":
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    )

                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred

            x_euler = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

            if self.target_type == "velocity":
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        x_euler, local_cond, text_embed, timesteps=sigmas[i + 1]
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x_euler, local_cond, text_embed, timesteps=sigmas[i + 1]
                    )
                x_pred_2 = alphas[i + 1] * x_euler - betas[i + 1] * v_pred
                noise_pred_2 = betas[i + 1] * x_euler + alphas[i + 1] * v_pred

            x_pred = (x_pred_2 + x_pred) / 2
            noise_pred = (noise_pred_2 + noise_pred) / 2
            x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

            # noise_pred = (noise_pred_2 + noise_pred) / 2
            # x = (alphas[i+1]*x + (alphas[i]*betas[i+1]-alphas[i+1]*betas[i])*noise_pred) / alphas[i]

        return x

    def dpmsolver_sample(
        self, timesteps, local_cond, text_embed, cfg_w=1.0, inpaint_x=None
    ):
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        noise_schedule = NoiseScheduleVP(schedule="cosine")

        def my_wrapper(fn):
            def wrapped(x, t, **kwargs):
                if cfg_w > 1:
                    out, uncond_out = fn(
                        x.expand(batch_size, -1, -1),
                        timesteps=t.expand(batch_size),
                        **kwargs,
                    ).chunk(2)
                    out = cfg_w * out + (1 - cfg_w) * uncond_out
                else:
                    out = fn(x, timesteps=t, **kwargs)
                return out

            return wrapped

        model_fn = model_wrapper(
            my_wrapper(self._forward),
            noise_schedule,
            model_type="v",
            model_kwargs={"local_cond": local_cond, "text_embed": text_embed},
        )
        dpm_solver = DPM_Solver(
            model_fn, noise_schedule, predict_x0=True
        )  # dpmsolver++
        # dpm_solver = DPM_Solver(model_fn, noise_schedule)
        x = dpm_solver.sample(
            x,
            t_end=0.001,
            steps=timesteps,
            order=2,
            skip_type="time_uniform",
            method="singlestep",
        )
        return x

    def plms_sample(self, timesteps, local_cond, text_embed, cfg_w=1.0):
        t = timesteps
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        pred_list = []
        for i in tqdm(range(t)):
            if cfg_w > 1:
                pred, pred_uncond = self._forward(
                    x,
                    local_cond,
                    text_embed,
                    timesteps=sigmas[i].expand(batch_size, -1),
                ).chunk(2)
                pred = cfg_w * pred + (1 - cfg_w) * pred_uncond
            else:
                pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])

            if self.target_type == "velocity":
                x_pred = alphas[i] * x - betas[i] * pred
                noise_pred = betas[i] * x + alphas[i] * pred
            elif self.target_type == "noise":
                x_pred = (x - betas[i] * pred) / alphas[i]
                noise_pred = pred
            else:
                raise NotImplementedError

            if len(pred_list) == 0:
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                if cfg_w > 1:
                    pred_prev, pred_prev_uncond = self._forward(
                        x_noisy,
                        local_cond,
                        text_embed,
                        timesteps=sigmas[i + 1].expand(batch_size, -1),
                    ).chunk(2)
                    pred_prev = cfg_w * pred_prev + (1 - cfg_w) * pred_prev_uncond
                else:
                    pred_prev = self._forward(
                        x_noisy, local_cond, text_embed, timesteps=sigmas[i + 1]
                    )
                pred_prime = (pred + pred_prev) / 2
            elif len(pred_list) == 1:
                pred_prime = (3 * pred - pred_list[-1]) / 2
            elif len(pred_list) == 2:
                pred_prime = (23 * pred - 16 * pred_list[-1] + 5 * pred_list[-2]) / 12
            elif len(pred_list) >= 3:
                pred_prime = (
                    55 * pred
                    - 59 * pred_list[-1]
                    + 37 * pred_list[-2]
                    - 9 * pred_list[-3]
                ) / 24

            if self.target_type == "velocity":
                x_pred_prime = alphas[i] * x - betas[i] * pred_prime
                noise_pred_prime = betas[i] * x + alphas[i] * pred_prime
            elif self.target_type == "noise":
                x_pred_prime = (x - betas[i] * pred_prime) / alphas[i]
                noise_pred_prime = pred
            else:
                raise NotImplementedError

            x = alphas[i + 1] * x_pred_prime + betas[i + 1] * noise_pred_prime
            pred_list.append(pred)
        return x

    def consistency_sample(
        self, timesteps, local_cond, text_embed, cfg_w, cfg_interval=[0, 1]
    ):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))

        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        for i in range(t):
            v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])
            if (
                cfg_w != 1
                and sigmas[i] <= cfg_interval[1]
                and sigmas[i] >= cfg_interval[0]
            ):
                v_pred, v_pred_uncond = self._forward(
                    x, local_cond, text_embed, timesteps=sigmas[i]
                ).chunk(2)
                v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
            else:
                v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])
            x_pred = alphas[i] * x - betas[i] * v_pred
            noise = torch.randn_like(v_pred)
            x = alphas[i + 1] * x_pred + betas[i + 1] * noise
        return x

    def compute_condition(self, inputs, uncond=False):
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["frontend"])  # [B, T, 1024]
        else:
            text_embed = None

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        # speaker embedding
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])
        if uncond:
            spk_emb[:, :] = 0  # whether enable global spk cfg
        spk_emb = spk_emb.unsqueeze(1).expand(-1, inputs[ctx_feature].shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([inputs[ctx_feature], spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)
        return local_cond, text_embed

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", cfg_w=1.0, **kwargs):
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["frontend"])  # [B, T, 1024]
        else:
            text_embed = None

        ctx_feature = f"{self.hp.ctx_feature}_ctx"

        # speaker embedding
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        spk_emb = self.prompt_encoder(inputs[prompt_feature])
        if cfg_w != 1:
            spk_emb = spk_emb.repeat(2, 1)
            spk_emb[1, :] = 0  # whether enable global spk cfg
        spk_emb = spk_emb.unsqueeze(1).expand(-1, inputs[ctx_feature].shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([inputs[ctx_feature], spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        if sampler == "ddim":
            x = self.ddim_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        elif sampler == "heun":
            x = self.heun_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        elif sampler == "plms":
            x = self.plms_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        elif sampler == "consistency":
            x = self.consistency_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        elif sampler == "euler":
            x = self.euler_sample(
                timesteps, local_cond, text_embed, cfg_w=cfg_w, **kwargs
            )
        else:
            raise NotImplementedError

        return x.transpose(1, 2)


class DualSampler:
    def __init__(
        self,
        cond_model,
        uncond_model,
        max_t=1,
        min_t=0,
        target_type="velocity",
        out_channels=64,
    ):
        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.max_t = max_t
        self.min_t = min_t
        self.target_type = target_type
        self.out_channels = out_channels

    def ddim_sample(
        self,
        cfg_w,
        local_cond1,
        text_embed1,
        local_cond2,
        text_embed2,
        sigmas,
        alphas,
        betas,
    ):

        t = len(alphas) - 1
        _, device, frm_len = (
            local_cond1.size(0),
            local_cond1.device,
            local_cond1.size(1),
        )
        x = torch.randn([1, frm_len, self.out_channels], device=device)

        for i in range(t):
            if self.target_type == "velocity":
                if cfg_w != 1:
                    v_pred = self.cond_model._forward(
                        x, local_cond1, text_embed1, timesteps=sigmas[i]
                    )

                    v_pred_uncond = self.uncond_model._forward(
                        x, local_cond2, text_embed2, timesteps=sigmas[i]
                    )
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self.cond_model._forward(
                        x, local_cond1, text_embed1, timesteps=sigmas[i]
                    )

                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
        return x

    def inference(self, inputs, uncond_inputs, nfe, sampler, cfg_w, **kwargs):
        local_cond1, text_embed1 = self.cond_model.compute_condition(inputs, False)
        local_cond2, text_embed2 = self.uncond_model.compute_condition(
            uncond_inputs, True
        )
        device = local_cond1.device

        sigmas = torch.linspace(self.max_t, self.min_t, nfe + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.cond_model.get_alpha_beta(sigmas_batch)

        if sampler == "ddim":
            x = self.ddim_sample(
                cfg_w,
                local_cond1,
                text_embed1,
                local_cond2,
                text_embed2,
                sigmas,
                alphas,
                betas,
                **kwargs,
            )
        else:
            raise NotImplementedError
        return x.transpose(1, 2)


class LCMDiT(Diffusion):
    def __init__(self, hp):
        super().__init__(hp)
        self.guidance_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)

    def compute_forward_condition(self, inputs, sigmas=None):
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["frontend"])  # [B, T, 1024]
            text_lens = inputs["text_lens"]

        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        prompt_feature = f"prompt_{self.hp.prompt_feature}"
        B, device = inputs[ctx_feature].size(0), inputs[ctx_feature].device

        # speaker embedding
        spk_emb = self.prompt_encoder(inputs[prompt_feature])
        spk_emb[inputs["flag_drop"]] = 0
        spk_emb = spk_emb.unsqueeze(1).expand(-1, inputs[ctx_feature].shape[1], -1)

        # local conditioning.
        local_cond = torch.cat([inputs[ctx_feature], spk_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        if sigmas is None:
            sigmas = self.sigma_distribution(num_samples=B, device=device).float()
        sigmas = torch.where(
            torch.rand_like(sigmas) < self.hp.p_max_t,
            torch.full_like(sigmas, self.hp.max_t),
            sigmas,
        )
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # time embedding
        time_emb = self.time_embedding(sigmas)
        guidance_emb = self.guidance_embedding(inputs["guidance_scale"])
        time_emb = time_emb + guidance_emb

        time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
        return text_embed, text_lens, local_cond, time_emb, alphas, betas

    def _forward(self, x, local_cond, text_embed, timesteps, cfg_w):
        residual = x

        time_emb = self.time_embedding(timesteps)
        guidance_emb = self.guidance_embedding(cfg_w)
        time_emb = time_emb + guidance_emb

        time_emb = time_emb.unsqueeze(1).expand(
            local_cond.shape[0], local_cond.shape[1], -1
        )
        x = self.x_prenet(x) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        if self.hp.use_bn_eos_bos:
            bn_bos = self.bn_eos_bos[0][None, None, :].expand(x.shape[0], -1, -1)
            bn_eos = self.bn_eos_bos[1][None, None, :].expand(x.shape[0], -1, -1)
            x = torch.cat([bn_bos, x, bn_eos], dim=1)

        if self.hp.use_textprefix:
            x = torch.cat([text_embed, x], dim=1)
            if self.use_seg_embed:
                seg_input = torch.zeros([1, x.shape[1]], device=x.device).long()
                seg_input[:, : text_embed.shape[1]] = 1
                seg_input[:, text_embed.shape[1] :] = 2
                seg_output = self.seg_embed(seg_input)
                x = x + seg_output

        pred_v = self.encoder(x, x.shape[1])

        if self.hp.use_textprefix:
            pred_v = pred_v[:, text_embed.shape[1] :, :]

        if self.hp.use_bn_eos_bos:
            pred_v = pred_v[:, 1:-1, :]

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            pred = pred_v

        return pred
