import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn, Tensor
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union
from recipes.voicebox.diffusion_modules.util_layers import RMSNorm
from dataclasses import dataclass, field
from recipes.voicebox.modules.based_ctiga_llama import ModelArgs as LLamaArgs, LLaMa
from functools import partial
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils import groundtruth
from einops import rearrange, reduce, repeat
from samantha.models.dpm_solver_pytorch import (
    DPM_Solver,
    NoiseScheduleVP,
    model_wrapper,
)
import numpy as np

class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x



"""For VDiffusion"""
class Distribution:
    """Interface used by different distributions"""
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()

def Ts(t):
    """Builds a type template for a given type that accepts a list of instances"""
    return lambda *types: lambda: t(*[tp() for tp in types])

class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


def Repeat(m: Union[nn.Module, Type[nn.Module]], times: int) -> Any:
    ms = (m,) * times
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)

class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
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


class TimeEmbedding(nn.Module):
    def __init__(self, modulation_features, num_layers: int=2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features, bias=bias), nn.GELU()
            ),
            times=num_layers
        )

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

@dataclass
class ModelArgs:
    cond_type: str = "embed" # {embed, token}
    cond_mode: str = "align"   # {prefix, align}
    local_cond_dim: int = 1536
    time_embed_dim: int = 1536

    # token
    n_token: int = 32768
    n_token_hierarchy: int = 1
    token_embed_dim: int = 512
    token_hidden_dim: int = 768 
    token_upscales: list = field(default_factory=lambda : [1]) 
    token_downscales: list = field(default_factory=lambda : [1])
    use_token_vector: bool = False
    token_vector_dim: int = 32
    
    local_cond_project_type: str = "linear" # conv
    local_cond_conv_kernel: int = 9
    local_cond_conv_padding: int = 4
    
    # llama
    encoder_dim: int = 1536
    encoder_n_layers: int = 24
    encoder_n_heads: int = 24
    causal: bool = False
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda : [-1, -1]) 
    window_type: str = "elemwise" # elemwise, blockwise
    flashattn_version: float = 2.3
    
    llama_provider: str = "ctiga"
    postnet_type: str = "linear" # conv
    postnet_kernel: int = 3

    # features ["mel", "bn"]
    target: str = "bn"
    prompt_feature: str = "bn"
    ctx_feature: str = "bn"
    in_channels: int = 80
    out_channels :  int = 80
    use_textprefix: bool = False
    x_padding_value: int = -2
    bn_frame_rate: int = 49

    bias: bool = False
    use_unet_style_skip_connect: bool = False
    target_type: str = "velocity"
    use_prompt: bool = False

    min_t: float = 0.0
    max_t: float = 1.0

    use_mla: bool = False

    # min_snr
    min_snr_gamma: int = 5
    min_snr_loss_weight: bool = True
    x_clip_val: float = 1.0

    # dropout
    cond_dropout: float = 0.0
    cond_padding: float = -5.0


class LlamaDiffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = hp.min_t if hasattr(hp, "min_t") else 0
        self.max_t = hp.max_t if hasattr(hp, "max_t") else 1
        self.target_type = hp.target_type if hasattr(hp, "target_type") else "velocity"
        self.use_prompt = hp.use_prompt if hasattr(hp, "use_prompt") else True

        self.act_fn = nn.GELU()
        self.bias = hp.bias

        # 1) input: text (not available).

        # 2) input: token.
        if hp.cond_type == "token":
            if hp.n_token_hierarchy > 1:
                self.token_embedding = nn.ModuleList([nn.Embedding(hp.n_token, hp.token_embed_dim) for _ in range(hp.n_token_hierarchy)])
                self.token_concat_net = nn.Sequential(
                    nn.Linear(hp.token_embed_dim * hp.n_token_hierarchy, hp.token_embed_dim * 2, bias=self.bias),
                    nn.SiLU(), 
                    nn.Linear(hp.token_embed_dim * 2, hp.token_embed_dim, bias=self.bias),
                )
            else:
                self.token_embedding = nn.Embedding(hp.n_token, hp.token_embed_dim)

        self.token_prenet = self.create_token_prenet(hp)

        # 3) input: time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)

        # 4) input: prompt (not available)
        local_cond_in_channels = hp.token_hidden_dim
        self.local_cond_project = nn.Linear(
                local_cond_in_channels,
                hp.local_cond_dim,  
                bias=self.bias)
        
        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]
        # backbone
        llama_config = LLamaArgs(dim=hp.encoder_dim,
                                n_layers=hp.encoder_n_layers, 
                                n_heads=hp.encoder_n_heads,
                                causal=hp.causal,
                                use_window_mask=hp.use_window_mask,
                                window_size=hp.window_size,
                                window_type=hp.window_type,
                                use_unet_style_skip_connect=hp.use_unet_style_skip_connect,
                                flashattn_version=hp.flashattn_version,
                                use_mla=hp.use_mla,)
        self.encoder = LLaMa(llama_config, hp.llama_provider)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim, bias=self.bias)

        # if hp.cond_mode == "align":
        self.prenet = nn.Linear(hp.time_embed_dim + hp.local_cond_dim, hp.encoder_dim, bias=self.bias)
        # if hp.cond_mode == "prefix":
        #     self.sos_embed = nn.Embedding(1, hp.encoder_dim)
        #     self.sos_id = 0

        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
        
        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)
        self.x_clip_val = hp.x_clip_val 
        self.clip_fn = partial(
            torch.clamp, min=self.x_clip_val * -1, max=self.x_clip_val
        )
        self.min_snr_gamma = hp.min_snr_gamma
        self.min_snr_loss_weight = hp.min_snr_loss_weight

    def create_token_prenet(self, hp):
        token_prenet = nn.ModuleList([
            nn.Conv1d(hp.token_embed_dim, hp.token_hidden_dim, kernel_size=1)
            ])

        for scale in hp.token_upscales:
            token_prenet.append(nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="nearest"),
                nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=3, padding=1),
                self.act_fn,
                RMSNorm(hp.token_hidden_dim, feat_dim=1)
                ))
        for scale in hp.token_downscales:
            if scale == 1:
                conv = nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=1, stride=1)
            else:
                conv = nn.Conv1d(hp.token_hidden_dim, hp.token_hidden_dim, kernel_size=scale*2, stride=scale, padding=scale//2+scale%2)

            token_prenet.append(nn.Sequential(
                conv,
                self.act_fn,
                RMSNorm(hp.token_hidden_dim, feat_dim=1)
                ))

        return token_prenet
    

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta
    
    def prepare_token_embed(self, inputs):
        # token encoder to align frame-rate.
        if 'token' in inputs:
            if self.hp.n_token_hierarchy > 1:
                token_embeds = []
                B = inputs['token'].size(0)
                for h in range(self.hp.n_token_hierarchy):
                    token_embeds.append(self.token_embedding[h](inputs['token'][..., h]))
                token_embeds = torch.cat(token_embeds, dim=-1)  # [B, T, D*H]
                if self.training and torch.rand(1) < 0.1:   # apply quantizer dropout 
                    quantizer_dropout_R = torch.randint(self.hp.n_token_hierarchy, size=(B,)) + 1   # values can be {0,...R-1}+1
                    quantizer_dropout_dim = quantizer_dropout_R * self.hp.token_embed_dim    # [B]
                    mask = torch.ones_like(token_embeds)
                    for bs in range(B):
                        mask[bs, ..., quantizer_dropout_dim[bs]:] = 0
                    token_embeds = mask * token_embeds
                token_embed = self.token_concat_net(token_embeds)  
            else:
                token_embed = self.token_embedding(inputs["token"])
        elif "embed" in inputs:
            token_embed = inputs["embed"]
        else:
            raise ValueError("token or embed must be provided")
        token_embed = token_embed.transpose(1, 2)   # [B. N  T]
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2) # B, T, C
        return token_embed
    
    def prepare_training_input(self, input_dict):
        """
        Necessary items:
            - bn: bottleneck feature, with shape [B, T (50Hz), N]
            - 
        Optional items:
            - embed: token embedding, with shape [B, T (25Hz), D]
            - token: token, with shape [B, T (25Hz), (R)]
        """
        # feat_lens = input_dict["bn_lens"]

        # targets
        x1 = input_dict["bn"]
        B, device = x1.shape[0], x1.device
        # sigmas
        with torch.cuda.amp.autocast(enabled=False):
            sigmas = self.sigma_distribution(num_samples=B, device=device)
            sigmas_batch = extend_dim(sigmas, dim=x1.ndim)
            alphas, betas = self.get_alpha_beta(sigmas_batch)

            # noisy target        
            noise = torch.randn_like(x1)
            x_noisy = alphas * x1 + betas * noise

            if self.target_type == "velocity":
                target = alphas * noise - betas * x1
            elif self.target_type == "x1":
                target = x1
            elif self.target_type == "noise":
                target = noise

        if self.min_snr_loss_weight:
            snr = alphas ** 2 / betas ** 2
        else:
            snr = torch.ones_like(alphas)
        maybe_clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

        if self.target_type == "velocity":
            loss_weight = maybe_clipped_snr / (snr + 1)
        elif self.target_type == "x1":
            loss_weight = maybe_clipped_snr
        elif self.target_type == "noise":
            loss_weight = maybe_clipped_snr / snr

        # token encoder to align frame-rate.
        token_embed = self.prepare_token_embed(input_dict)

        # 1. Concat temporal aligned inputs with bn
        # time embedding
        time_emb = self.time_embedding(sigmas)
        time_emb = time_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)    # [B, T, D]


        input_dict.update({
            'token_embed': token_embed,
            'xt': x_noisy,
            'x1': x1,
            # 'noise': noise,
            'time_emb': time_emb,
            'alphas': alphas,
            'betas': betas,
            'target': target,
            'loss_weight': loss_weight,
        })
        return input_dict 

    def get_sos_embed(self, batch_size=1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        # sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        sos_embs = torch.full(size=(batch_size, 1, self.hp.encoder_dim), fill_value=0., dtype=dtype, device=device)
        # return self.sos_embed(sos_ids)
        return sos_embs

    def forward(self, input_dict):
        training_input = self.prepare_training_input(input_dict)

        # token condition
        local_cond = self.local_cond_project(training_input["token_embed"])

        # prepare target 
        # residual, alphas, noise, betas, x1, xt = training_input["xt"], training_input["alphas"], \
        #     training_input["noise"], training_input["betas"], training_input["x1"], training_input["xt"]
        residual, alphas, betas, xt, target = training_input["xt"], training_input["alphas"], \
            training_input["betas"], training_input["xt"], training_input["target"]
        
        seq_mask = input_dict["bn_mask"]
        # concat condition [time emb + token(emb) condition]    BF16
        if self.hp.cond_mode == 'align':
            xt = self.x_prenet(xt) + self.prenet(torch.cat([training_input["time_emb"].float(), local_cond.float()], dim=-1))
            pred_v = self.encoder(xt, xt.shape[1], attention_mask=seq_mask) # BF16
        elif self.hp.cond_mode == "prefix":
            xt = self.x_prenet(xt) 
            prefix_emb = self.prenet(torch.cat([training_input["time_emb"].float(), local_cond.float()], dim=-1)) 
            sos_emb = self.get_sos_embed(batch_size=xt.shape[0])
            T_prefix = prefix_emb.shape[1] + 1  # (sos_embed)

            # 60*25, 60*50 = 4500
            encoder_input = torch.cat([prefix_emb, sos_emb, xt], dim=1)
            seq_mask = torch.cat([
                torch.ones_like(seq_mask[:, :1]).repeat(1, T_prefix),   # [B, T_prefix]
                seq_mask,                                               # [B, T_target]
                ], dim=1)
            pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask) # BF16
            pred_v = pred_v[:, T_prefix:, :]

        with torch.cuda.amp.autocast(enabled=False):
            pred_v = self.postnet(pred_v.float())

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
            else:
                pred = pred_v
        elif self.target_type == "x1":
            pred = betas * residual + alphas * pred_v

        return pred.transpose(1, 2), target.transpose(1, 2)
    

    def clear_infer_params(self, t, total_frame=None, bs=1, text_cfg=1.0, mem_efficient=False):
        if total_frame is None:
            self.infer_params = None
            self.cached_noise = None
        else:
            self.infer_params = []
            self.cached_noise = []

            # if self.hp.cond_mode == "prefix":
            #     total_frame = 2 * total_frame   # @(TODO) qinxin: hardcode here
            for i in range(t):
                self.infer_params.append(
                    InferenceParams(
                        max_sequence_len=total_frame,
                        max_batch_size=bs * (2 if text_cfg != 1.0 else 1),
                        last=False,
                        mem_efficient=mem_efficient,
                    )
                )
                self.cached_noise.append(
                    torch.randn(1, total_frame, self.hp.out_channels).expand(bs, -1, -1)
                )

    def update_infer_params(self, cache_len, last=False):
        for i in range(len(self.infer_params)):
            self.infer_params[i].sequence_len_offset += cache_len
            self.infer_params[i].last = last

    def clear_cache(self, t, total_frame=None, bs=1, text_cfg=1.0):
        if total_frame is None:
            self.cached_noise = None
        else:
            self.cached_noise = [
                torch.randn([1, total_frame, self.hp.out_channels]).expand(bs, -1, -1)
                for i in range(t + 1)
            ]
        self.cached_v = dict([(i, None) for i in range(t)])

    
    @torch.no_grad()
    def chunk_inference(
        self, inputs, timesteps=20, sampler="ddim", text_cfg_w=1.0, first=True, n_token_hierarchy_eval=None, **kwargs
    ):
        if not hasattr(self, "token_overlap"):
            self.token_overlap = int(np.prod(self.hp.token_downscales))
        if not hasattr(self, "token_embed_overlap"):
            self.token_embed_overlap = int(np.prod(self.hp.token_upscales))

        assert self.infer_params is not None and self.cached_noise is not None
        text_embed = None

        # token encoder to align frame-rate.
        if self.hp.cond_type == "token":
            if self.hp.n_token_hierarchy > 1:
                token_embeds = []
                if inputs["token"].ndim == 2:
                    inputs["token"] = inputs["token"].unsqueeze(-1)
                B, T, H = inputs['token'].size()
                if self.hp.n_token_hierarchy > 2:   # concat version
                    for h in range(H):
                        token_embeds.append(self.token_embedding[h](inputs['token'][..., h]))
                    token_embeds = torch.cat(token_embeds, dim=-1)  # [B, T, D*H]
                    if n_token_hierarchy_eval is not None and n_token_hierarchy_eval < self.hp.n_token_hierarchy or H < self.hp.n_token_hierarchy:
                        if H < self.hp.n_token_hierarchy:
                            h = min(n_token_hierarchy_eval, H)
                            target_token_embeds = torch.zeros([B, T, self.hp.token_embed_dim*self.hp.n_token_hierarchy]).to(token_embeds)
                            target_token_embeds[..., :h*self.hp.token_embed_dim] = token_embeds[..., :h*self.hp.token_embed_dim]
                            token_embeds = target_token_embeds
                        else:
                            mask = torch.ones_like(token_embeds)
                            mask[:,..., n_token_hierarchy_eval*self.hp.token_embed_dim:] = 0
                            token_embeds = mask * token_embeds
                    token_embed = self.token_concat_net(token_embeds)
                else:   # sum version
                    for h in range(self.hp.n_token_hierarchy):
                        token_embeds.append(self.token_embedding[h](inputs['token'][..., h]))
                    token_embed = torch.stack(token_embeds, dim=-1).sum(-1)
            else:
                token_embed = self.token_embedding(inputs["token"])
        else:
            token_embed = inputs["token_embed"]
        token_embed = token_embed.transpose(1, 2)
        # print(f"[chunk_inference:before] {token_embed.shape=}")

        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        # print(f"[chunk_inference:after] {token_embed.shape=}")
        start_idx = 0 if first else self.token_embed_overlap
        end_idx = -self.token_embed_overlap
        token_embed = token_embed[..., start_idx:end_idx]
        token_embed = token_embed.transpose(1, 2)  # B, T, C
        # print(f"[chunk_inference:slice] {token_embed.shape=}")

        # TODO: make ctx_feature
        ctx_feature = f"{self.hp.ctx_feature}_ctx"
        assert token_embed.shape[1] >= inputs[ctx_feature].shape[1]
        token_embed = token_embed[:, : inputs[ctx_feature].shape[1], :]

        local_cond = self.local_cond_project(token_embed)

        if sampler == "ddim":
            x = self.ddim_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "plms":
            x = self.plms_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "consistency":
            x = self.consistency_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        else:
            raise NotImplementedError

        x = x.transpose(1, 2)

        groundtruth.emit(
            "diffusion",
            data={
                "inputs": inputs,
                "local_cond": local_cond,
                "out_mel": x,
                "cached_noise": self.cached_noise if hasattr(self, "cached_noise") else None,
                "params": {
                    "timesteps": timesteps,
                    "sampler": sampler,
                    "text_cfg_w": text_cfg_w,
                    "token_overlap": self.token_overlap,
                    "token_embed_overlap": self.token_embed_overlap,
                },
            },
        )

        return x
    
    @torch.no_grad()
    def prefix_inference(
        self, inputs, timesteps=20, sampler="ddim", text_cfg_w=1.0, first=True, n_token_hierarchy_eval=None, **kwargs
    ):
        assert self.infer_params is not None and self.cached_noise is not None
        text_embed = None

        # token encoder to align frame-rate.
        if self.hp.cond_type == "token":
            if self.hp.n_token_hierarchy > 1:
                token_embeds = []
                if inputs["token"].ndim == 2:
                    inputs["token"] = inputs["token"].unsqueeze(-1)
                B, T, H = inputs['token'].size()
                if self.hp.n_token_hierarchy > 2:   # concat version
                    for h in range(H):
                        token_embeds.append(self.token_embedding[h](inputs['token'][..., h]))
                    token_embeds = torch.cat(token_embeds, dim=-1)  # [B, T, D*H]
                    if n_token_hierarchy_eval is not None and n_token_hierarchy_eval < self.hp.n_token_hierarchy or H < self.hp.n_token_hierarchy:
                        if H < self.hp.n_token_hierarchy:
                            h = min(n_token_hierarchy_eval, H)
                            target_token_embeds = torch.zeros([B, T, self.hp.token_embed_dim*self.hp.n_token_hierarchy]).to(token_embeds)
                            target_token_embeds[..., :h*self.hp.token_embed_dim] = token_embeds[..., :h*self.hp.token_embed_dim]
                            token_embeds = target_token_embeds
                        else:
                            mask = torch.ones_like(token_embeds)
                            mask[:,..., n_token_hierarchy_eval*self.hp.token_embed_dim:] = 0
                            token_embeds = mask * token_embeds
                    token_embed = self.token_concat_net(token_embeds)
                else:   # sum version
                    for h in range(self.hp.n_token_hierarchy):
                        token_embeds.append(self.token_embedding[h](inputs['token'][..., h]))
                    token_embed = torch.stack(token_embeds, dim=-1).sum(-1)
            else:
                token_embed = self.token_embedding(inputs["token"])
        else:
            token_embed = inputs["token_embed"]
        token_embed = token_embed.transpose(1, 2)
        # print(f"[prefix_inference:before] {token_embed.shape=}")

        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        # print(f"[prefix_inference:after] {token_embed.shape=}")
        
        token_embed = token_embed.transpose(1, 2)  # B, T, C
        # print(f"[prefix_inference:slice] {token_embed.shape=}")

        # TODO: make ctx_feature
        # ctx_feature = f"{self.hp.ctx_feature}_ctx"
        # assert token_embed.shape[1] >= inputs[ctx_feature].shape[1]
        # token_embed = token_embed[:, : inputs[ctx_feature].shape[1], :]

        local_cond = self.local_cond_project(token_embed)
        
        if sampler == "ddim":
            x = self.ddim_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "plms":
            x = self.plms_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        elif sampler == "consistency":
            x = self.consistency_sample(
                timesteps, local_cond, text_embed, text_cfg_w=text_cfg_w, **kwargs
            )
        else:
            raise NotImplementedError

        x = x.transpose(1, 2)

        # groundtruth.emit(
        #     "diffusion",
        #     data={
        #         "inputs": inputs,
        #         "local_cond": local_cond,
        #         "out_mel": x,
        #         "cached_noise": self.cached_noise if hasattr(self, "cached_noise") else None,
        #         "params": {
        #             "timesteps": timesteps,
        #             "sampler": sampler,
        #             "text_cfg_w": text_cfg_w,
        #             # "token_overlap": self.token_overlap,
        #             # "token_embed_overlap": self.token_embed_overlap,
        #         },
        #     },
        # )

        return x
    

    def ddim_sample(
        self,
        timesteps,
        local_cond,
        text_embed,
        text_cfg_w=1.0,
        rescale_factor=0.7,
        inpaint_x=None,
        use_cache=False,
        cached_v_len=None,
        eta=0.0,
        use_infer_params=False,
    ):
        assert use_cache ^ use_infer_params
        t = timesteps
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        real_batch_size = batch_size if text_cfg_w == 1.0 else batch_size // 2
        print(
            self.infer_params[0].sequence_len_offset,
            self.infer_params[0].n_look_past,
            self.infer_params[0].n_look_future,
        )

        # if len(self.infer_params[0].key_value_memory_dict.keys()) > 0:
        #     print(self.infer_params[0].key_value_memory_dict[0].shape)
            
        if self.hp.cond_mode == "prefix":
            frm_len = 2 * frm_len   # @(TODO) qinxin: hardcode here

        if use_cache:
            assert self.cached_noise is not None
            if self.cached_noise is not None:
                x = self.cached_noise[0][:, :frm_len, :].to(device)
        elif use_infer_params:
            assert self.cached_noise is not None
            assert self.infer_params is not None
            seqlen_offset = self.infer_params[0].sequence_len_offset
            x = self.cached_noise[0][:, seqlen_offset : seqlen_offset + frm_len, :].to(
                device
            )
        else:
            x = torch.randn([1, frm_len, self.hp.out_channels], device=device).expand(
                real_batch_size, -1, -1
            )

        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        # sigmas += 0.6
        # sigmas /= sigmas.max()

        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        for i in range(t):
            # print(f"step({i}/{t})")
            if self.target_type == "velocity":
                v_pred = self._forward(
                    x.repeat(2 if text_cfg_w != 1 else 1, 1, 1),
                    local_cond,
                    text_embed,
                    timesteps=sigmas[i].expand(batch_size, -1),
                    infer_params=self.infer_params[i] if use_infer_params else None,
                )
                if text_cfg_w != 1:
                    v_pred, v_pred_uncond = v_pred.chunk(2)
                    v_pred = text_cfg_w * v_pred + (1 - text_cfg_w) * v_pred_uncond

                # TODO: 只是模拟cache过程
                if use_cache:
                    if self.cached_v[i] is not None:
                        cached_v_len = (
                            self.cached_v[i].shape[1]
                            if cached_v_len is None
                            else cached_v_len
                        )
                        v_pred[:, :cached_v_len, :] = self.cached_v[i][
                            :, :cached_v_len, :
                        ]
                    self.cached_v[i] = v_pred
                # print(f"{x.shape=} {v_pred.shape=} {alphas[i].shape=}")
                x_pred = alphas[i] * x[:real_batch_size] - betas[i] * v_pred
                noise_pred = betas[i] * x[:real_batch_size] + alphas[i] * v_pred

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

    def dpmsolver_sample(
        self, timesteps, local_cond, text_embed, text_cfg_w=1.0, inpaint_x=None
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
                if text_cfg_w > 1:
                    out, uncond_out = fn(
                        x.expand(batch_size, -1, -1),
                        timesteps=t.expand(batch_size),
                        **kwargs,
                    ).chunk(2)
                    out = text_cfg_w * out + (1 - text_cfg_w) * uncond_out
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
            skip_type="logSNR",
            method="multistep",
        )
        return x

    def plms_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
        t = timesteps
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        if t > 20:
            sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        else:
            sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device) ** 2
            # sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        pred_list = []
        for i in tqdm(range(t)):
            if text_cfg_w > 1:
                pred, pred_uncond = self._forward(
                    x,
                    local_cond,
                    text_embed,
                    timesteps=sigmas[i].expand(batch_size, -1),
                ).chunk(2)
                pred = text_cfg_w * pred + (1 - text_cfg_w) * pred_uncond
            else:
                pred = self._forward(
                    x,
                    local_cond,
                    text_embed,
                    timesteps=sigmas[i].expand(batch_size, -1),
                )

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
                if text_cfg_w > 1:
                    pred_prev, pred_prev_uncond = self._forward(
                        x_noisy,
                        local_cond,
                        text_embed,
                        timesteps=sigmas[i + 1].expand(batch_size, -1),
                    ).chunk(2)
                    pred_prev = (
                        text_cfg_w * pred_prev + (1 - text_cfg_w) * pred_prev_uncond
                    )
                else:
                    pred_prev = self._forward(
                        x_noisy,
                        local_cond,
                        text_embed,
                        timesteps=sigmas[i + 1].expand(batch_size, -1),
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
        self,
        timesteps,
        local_cond,
        text_embed,
        text_cfg_w=1.0,
        rescale_factor=0.7,
        use_cache=False,
        cached_v_len=None,
        use_infer_params=False,
    ):
        t = timesteps
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        real_batch_size = batch_size if text_cfg_w == 1.0 else batch_size // 2
        if use_cache:
            x = self.cached_noise[0][:, :frm_len, :].to(device)
        elif use_infer_params:
            assert self.cached_noise is not None
            assert self.infer_params is not None
            seqlen_offset = self.infer_params[0].sequence_len_offset
            x = self.cached_noise[0][:, seqlen_offset : seqlen_offset + frm_len, :].to(
                device
            )
        else:
            x = torch.randn([1, frm_len, self.hp.out_channels], device=device).expand(
                real_batch_size, -1, -1
            )

        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        for i in range(t):
            v_pred = self._forward(
                x.repeat(2 if text_cfg_w != 1 else 1, 1, 1),
                local_cond,
                text_embed,
                timesteps=sigmas[i].expand(batch_size, -1),
                infer_params=self.infer_params[i] if use_infer_params else None,
            )
            if text_cfg_w != 1:
                v_pred, v_pred_uncond = v_pred.chunk(2)
                if rescale_factor <= 0:
                    v_pred = text_cfg_w * v_pred + (1 - text_cfg_w) * v_pred_uncond
                else:
                    v_pred = self.apply_rescale_cfg(v_pred, v_pred_uncond, text_cfg_w, rescale_factor)

            # TODO: 只是模拟cache过程
            if use_cache:
                if self.cached_v[i] is not None:
                    cached_v_len = (
                        self.cached_v[i].shape[1]
                        if cached_v_len is None
                        else cached_v_len
                    )
                    v_pred[:, :cached_v_len, :] = self.cached_v[i][:, :cached_v_len, :]
                self.cached_v[i] = v_pred

            x_pred = alphas[i] * x - betas[i] * v_pred

            if use_cache:
                noise = self.cached_noise[i + 1][:, :frm_len, :].to(device)
            else:
                noise = torch.randn_like(v_pred)

            x = alphas[i + 1] * x_pred + betas[i + 1] * noise

        return x
    
    def make_cfg_input(self, inputs, padding_value):
        # [batch*2, seqlen, dim]
        if self.hp.cond_type == "token":
            token_key = "all_token" 
        else:
            token_key = "all_token_embed"
        inputs[token_key] = torch.cat(
            (
                inputs[token_key],
                torch.ones_like(inputs[token_key]) * padding_value,
            ),
            0,
        )

        # bn_ctx_key = "all_bn_ctx"
        # inputs[bn_ctx_key] = inputs[bn_ctx_key].repeat(2, 1, 1)
        return inputs
    

    def _forward(
        self, x, local_cond, text_embed, timesteps, infer_params: InferenceParams = None
    ):
        residual = x
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)

        if self.hp.cond_mode == "align":
            x = self.x_prenet(x) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))
        elif self.hp.cond_mode == "prefix":
            bs, T_prefix = local_cond.shape[:2]
            prefix_emb = self.prenet(torch.cat([time_emb, local_cond], dim=-1))
            xt = self.x_prenet(x) 
            x = torch.cat([prefix_emb, self.get_sos_embed(bs), xt], dim=1)
            # x = torch.cat([prefix_emb, xt], dim=1)

        pred_v = self.encoder(x, x.shape[1], inference_params=infer_params)
        pred_v = self.postnet(pred_v)

        if self.hp.cond_mode == "prefix":
            pred_v = pred_v[:, T_prefix+1:, :]
            # pred_v = pred_v[:, T_prefix:, :]

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            raise NotImplementedError
        return pred