import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from torch import nn, Tensor
import math
from typing import Tuple
from einops import reduce, repeat
from tqdm import tqdm
import logging

from recipes.music_dit.model.util_layers import RMSNorm
from recipes.music_dit.model.loss import MaskedMAELoss, sequence_mask
from recipes.music_dit.model.based_ctiga_llama import ModelArgs as LLamaArgs, LLaMa
from recipes.music_dit.model.a_unet.blocks import NumberEmbedder, Repeat
from recipes.music_dit.model.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
logger = logging.getLogger(__name__)



"""For MusicDiT"""

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

    
class BernoulliDistribution(Distribution):
    def __init__(self, v1, v2):
        super().__init__()
        self.map = torch.tensor([v1, v2]).unsqueeze(0)

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        index =  (torch.rand(num_samples, device=device) > 0.5).long()
        return self.map.repeat(num_samples, 1).to(device)[torch.arange(num_samples), index]


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))



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

class PreNet(nn.Module):
    def __init__(self, in_dim, out_dim, conv_kernel=7, conv_padding=3):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
                nn.GELU(),
                RMSNorm(out_dim, feat_dim=1),
                nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding)
                )

    def forward(self, inputs):
        return self.net(inputs.transpose(1, 2)).transpose(1, 2)

# deprecated
class ResPostNet(nn.Module):
    def __init__(self, in_dim, out_dim, conv_kernel, conv_padding):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.net = nn.Sequential(
                nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding),
                nn.GELU(),
                RMSNorm(out_dim, feat_dim=1),
                nn.Conv1d(out_dim, out_dim, kernel_size=conv_kernel, padding=conv_padding)
                )

    def forward(self, inputs):
        x = self.linear(inputs)
        return self.net(x.transpose(1, 2)).transpose(1, 2) + x


@dataclass
class ModelArgs:
    local_cond_dim: int = 1536
    time_embed_dim: int = 1536

    prefix_hidden_size: int = 0
    temporal_aligned_hidden_size: int = 0
    xattn_hidden_size: int = 0
   
    local_cond_project_type: str = "linear" # conv
    local_cond_conv_kernel: int = 9
    local_cond_conv_padding: int = 4
    
    # llama
    in_channels: int = 32
    out_channels :  int = 32    
    encoder_dim: int = 1536
    encoder_n_layers: int = 24
    encoder_n_heads: int = 24
    causal: bool = False
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda : [-1, -1]) 
    window_type: str = "elemwise" # elemwise, blockwise
    llama_provider: str = "ctiga"

    postnet_type: str = "linear" # conv
    postnet_kernel: int = 3

    bias: bool = False
    use_unet_style_skip_connect: bool = False
    target_type: str = "velocity"
    
    min_t: float = 0.0
    max_t: float = 1.0


class LlamaDiffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = hp.min_t if hasattr(hp, "min_t") else 0
        self.max_t = hp.max_t if hasattr(hp, "max_t") else 1
        self.target_type = hp.target_type if hasattr(hp, "target_type") else "velocity"
        
        self.act_fn = nn.GELU()
        self.bias = hp.bias

        # time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)

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
                                 use_unet_style_skip_connect=hp.use_unet_style_skip_connect)
        
        self.encoder = LLaMa(llama_config, hp.llama_provider)

        # TODO (qq) add conv layers if temporal_aligned_inputs needs resample to align with bn.

        # Project temporal aligned x_noisy to encoder_dim
        x_noisy_dim = hp.in_channels * 2 + 1 + hp.temporal_aligned_hidden_size
        self.x_prenet = nn.Linear(x_noisy_dim, hp.encoder_dim, bias=self.bias)

        # Project prefix inputs to encoder_dim
        if hp.prefix_hidden_size > 0:
            self.prefix_prenet = nn.Linear(hp.prefix_hidden_size, hp.encoder_dim, bias=self.bias)

        # Project xattn inputs to encoder_dim
        if hp.xattn_hidden_size > 0:
            self.xattn_prenet = nn.Linear(hp.xattn_hidden_size, hp.encoder_dim, bias=self.bias)

        # Project time embedding to encoder_dim
        self.prenet = nn.Linear(hp.time_embed_dim, hp.encoder_dim, bias=self.bias)
        
        if hp.postnet_type == "linear":
            self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
        elif hp.postnet_type == "conv":
            self.postnet = ResPostNet(
                    hp.encoder_dim,
                    hp.out_channels,
                    hp.postnet_kernel,
                    hp.postnet_kernel//2
                    )

        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)
        
    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, inputs):        
        # diffusion target
        x = inputs['bn']        
        B, device = x.size(0), x.device

        sigmas = self.sigma_distribution(num_samples=B, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # 1. Concat temporal aligned inputs with bn
        # time embedding
        time_emb = self.time_embedding(sigmas)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        # noisy target        
        noise = torch.randn_like(x)
        x_noisy = alphas * x + betas * noise
        residual = x_noisy
        if self.target_type == "velocity":
            target = alphas * noise - betas * x
        elif self.target_type == "x0":
            target = x
        elif self.target_type == "noise":
            target = noise

        # concat temporal aligned inputs to x_noisy.
        x_noisy = torch.cat([x_noisy, inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)
        if 'temporal_aligned_inputs_emb' in inputs:
            # TODO (qq) add conv layers if temporal_aligned_inputs needs resample to align with bn.
            x_noisy = torch.cat([x_noisy, inputs['temporal_aligned_inputs_emb']], -1)
        x_noisy = self.x_prenet(x_noisy) + self.prenet(time_emb)

        # 2. Concat prefix inputs with bn.
        encoder_input = x_noisy
        prefix_lens = 0
        if 'prefix_inputs_emb' in inputs:            
            # TODO (qq) also support var len prefix
            prefix_embed = inputs['prefix_inputs_emb']
            prefix_embed = self.prefix_prenet(prefix_embed)
            prefix_lens = prefix_embed.shape[1]
            feat_lens = x_noisy.shape[1]
            encoder_input = torch.cat([prefix_embed, x_noisy], dim=1)
            seq_mask = sequence_mask(torch.full((B,), feat_lens + prefix_lens).to(device), device=device)

        # TODO (qq) add xattn input to encoder forward
        pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask)
        pred_v = pred_v[:, prefix_lens:, :]
        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        elif self.target_type == "x0":
            pred = betas * residual + alphas * pred_v

        return pred.transpose(1, 2), target.transpose(1, 2)
    
    def _forward(self, x, local_cond, 
                 temporal_aligned_inputs_emb=None,
                 prefix_inputs_emb=None,
                 xattn_inputs_emb=None,
                 timesteps=0,
            alphas=None, betas=None):
        residual = x
        # time embedding
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        # x_noisy
        x_noisy = torch.cat([x, local_cond], -1)
        if temporal_aligned_inputs_emb:
            x_noisy = torch.cat([x_noisy, temporal_aligned_inputs_emb])
        x = self.x_prenet(x_noisy) + self.prenet(time_emb)

        # Append prefix inputs
        if prefix_inputs_emb is not None:
            prefix_inputs_emb = self.prefix_prenet(prefix_inputs_emb)
            x = torch.cat([prefix_inputs_emb, x], dim=1)

        # TODO (qq) add xattn_inputs_emb to encoder input.
        pred_v = self.encoder(x, x.shape[1])

        prefix_len = 0
        if prefix_inputs_emb is not None:
            prefix_len = prefix_inputs_emb.shape[1]
        pred_v = pred_v[:, prefix_len:, :]

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        elif self.target_type == "x0":
            pred = betas * residual + alphas * pred_v
        else:
            raise NotImplementedError
        return pred

    def clear_cache(self, t, total_frame=None):
        if total_frame is None:
            self.cached_noise = None
        else:
            self.cached_noise = [torch.randn([1, total_frame, self.hp.out_channels]) for i in range(t+1)]
        self.cached_v = dict([(i, None) for i in range(t)])

    def ddim_sample(self, frm_len, timesteps, local_cond, 
                    temporal_aligned_inputs_emb=None,
                    prefix_inputs_emb=None,
                    xattn_inputs_emb=None,
                    text_cfg_w=1.0, 
                    inpaint_x=None, 
                    use_cache=False, 
                    cached_v_len=None,
                    eta=0.0):
        t = timesteps
        batch_size, device = local_cond.size(0), local_cond.device
        if use_cache:
            assert self.cached_noise is not None
            if self.cached_noise is not None:
                x = self.cached_noise[0][:, :frm_len, :].to(device)
        else:
            if text_cfg_w != 1:
                # TODO: text_cfg_w>1
                x = torch.randn([1, frm_len, self.hp.out_channels], device=device).expand(batch_size // 2, -1, -1)
            else:
                x = torch.randn([1, frm_len, self.hp.out_channels], device=device).expand(batch_size, -1, -1)

        if t > 20:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        else:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)**2
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
            
        for i in range(t):
            if self.target_type == "velocity":
                if text_cfg_w != 1:
                    # TODO probably need to double check timesteps shape
                    v_pred, v_pred_uncond = self._forward(x.repeat(batch_size, 1, 1), 
                                                          local_cond, 
                                                          temporal_aligned_inputs_emb,
                                                          prefix_inputs_emb,
                                                          xattn_inputs_emb,
                                                          timesteps=sigmas[i].expand(batch_size, -1)).chunk(2)
                    v_pred = text_cfg_w * v_pred + (1 - text_cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(x, 
                                           local_cond, 
                                           temporal_aligned_inputs_emb,
                                            prefix_inputs_emb,
                                            xattn_inputs_emb,
                                            timesteps=sigmas[i])

                # TODO: 只是模拟cache过程
                if use_cache:
                    if self.cached_v[i] is not None:
                        cached_v_len = self.cached_v[i].shape[1] if cached_v_len is None else cached_v_len
                        v_pred[:, :cached_v_len, :] = self.cached_v[i][:, :cached_v_len, :]
                    self.cached_v[i] = v_pred

                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred

                # disable
                if inpaint_x is not None:
                    x_pred[:, :inpaint_x.shape[1], :] = inpaint_x
                    noise_pred[:, :inpaint_x.shape[1], :] = (x[:, :inpaint_x.shape[1], :] - alphas[i]*inpaint_x) / betas[i]

            if eta > 0:
                sigma = eta * betas[i + 1] / betas[i] * torch.sqrt(1 - (alphas[i]/alphas[i+1])**2)
                noise = torch.randn_like(noise_pred)
                x = alphas[i + 1] * x_pred + \
                        torch.sqrt(betas[i + 1]**2 - sigma**2) * noise_pred + \
                        sigma * noise
            else:
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x

    # def dpmsolver_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
    #     batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)
    #     x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
    #     noise_schedule = NoiseScheduleVP(schedule="cosine")

    #     def my_wrapper(fn):
    #         def wrapped(x, t, **kwargs):
    #             if text_cfg_w > 1:
    #                 out, uncond_out = fn(x.expand(batch_size, -1, -1), 
    #                         timesteps=t.expand(batch_size), **kwargs).chunk(2)
    #                 out = text_cfg_w * out + (1 - text_cfg_w) * uncond_out
    #             else:
    #                 out = fn(x, timesteps=t, **kwargs)
    #             return out
    #         return wrapped

    #     model_fn = model_wrapper(
    #             my_wrapper(self._forward),
    #             noise_schedule,
    #             model_type="v",
    #             model_kwargs={"local_cond": local_cond, "text_embed": text_embed}
    #             )
    #     dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True) # dpmsolver++
    #     #dpm_solver = DPM_Solver(model_fn, noise_schedule)
    #     x = dpm_solver.sample(
    #             x,
    #             t_end=0.001,
    #             steps=timesteps,
    #             order=2,
    #             skip_type="logSNR",
    #             method="multistep")
    #     return x
    
    # def plms_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
    #     t = timesteps
    #     batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)
    #     x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
    #     if t > 20:
    #         sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
    #     else:
    #         sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)**2
    #         #sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
    #     sigmas = repeat(sigmas, "i -> i b", b=1)
    #     sigmas_batch = extend_dim(sigmas, dim=x.ndim)
    #     alphas, betas = self.get_alpha_beta(sigmas_batch)

    #     pred_list = []
    #     for i in tqdm(range(t)):
    #         if text_cfg_w > 1:
    #             pred, pred_uncond = self._forward(x, local_cond, text_embed, 
    #                     timesteps=sigmas[i].expand(batch_size, -1)).chunk(2)
    #             pred = text_cfg_w * pred + (1 - text_cfg_w) * pred_uncond
    #         else:
    #             pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i]) 

    #         if self.target_type == "velocity":
    #             x_pred = alphas[i] * x - betas[i] * pred
    #             noise_pred = betas[i] * x + alphas[i] * pred
    #         elif self.target_type == "noise":
    #             x_pred = (x - betas[i] * pred) / alphas[i]
    #             noise_pred = pred
    #         else:
    #             raise NotImplementedError

    #         if len(pred_list) == 0:
    #             x_noisy = alphas[i+1] * x_pred + betas[i+1] * noise_pred
    #             if text_cfg_w > 1:
    #                 pred_prev, pred_prev_uncond = self._forward(x_noisy, local_cond, text_embed, 
    #                             timesteps=sigmas[i+1].expand(batch_size, -1)).chunk(2)
    #                 pred_prev = text_cfg_w * pred_prev + (1 - text_cfg_w) * pred_prev_uncond
    #             else:
    #                 pred_prev = self._forward(x_noisy, local_cond, text_embed, timesteps=sigmas[i+1]) 
    #             pred_prime = (pred + pred_prev) / 2
    #         elif len(pred_list) == 1:
    #             pred_prime = (3 * pred - pred_list[-1]) / 2
    #         elif len(pred_list) == 2:
    #             pred_prime = (23 * pred - 16 * pred_list[-1] + 5 * pred_list[-2]) / 12
    #         elif len(pred_list) >= 3:
    #             pred_prime = (55 * pred - 59 * pred_list[-1] + 37 * pred_list[-2] - 9 * pred_list[-3]) / 24

    #         if self.target_type == "velocity":
    #             x_pred_prime = alphas[i] * x - betas[i] * pred_prime
    #             noise_pred_prime = betas[i] * x + alphas[i] * pred_prime
    #         elif self.target_type == "noise":
    #             x_pred_prime = (x - betas[i] * pred_prime) / alphas[i]
    #             noise_pred_prime = pred
    #         else:
    #             raise NotImplementedError
            
    #         x = alphas[i+1] * x_pred_prime + betas[i+1] * noise_pred_prime
    #         pred_list.append(pred)
    #     return x

    # def consistency_sample(self, timesteps, local_cond, text_embed, 
    #         text_cfg_w=1.0, use_cache=False, cached_v_len=None):
    #     t = timesteps
    #     batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)

    #     if use_cache:
    #         x = self.cached_noise[0][:, :frm_len, :].to(device)
    #     else:
    #         x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

    #     if t > 20:
    #         sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
    #     else:
    #         sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)**2
    #     sigmas = repeat(sigmas, "i -> i b", b=1)
    #     sigmas_batch = extend_dim(sigmas, dim=3)
    #     alphas, betas = self.get_alpha_beta(sigmas_batch)
            
    #     for i in range(t):

    #         v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i]) 

    #         # TODO: 只是模拟cache过程
    #         if use_cache:
    #             if self.cached_v[i] is not None:
    #                 cached_v_len = self.cached_v[i].shape[1] if cached_v_len is None else cached_v_len
    #                 v_pred[:, :cached_v_len, :] = self.cached_v[i][:, :cached_v_len, :]
    #             self.cached_v[i] = v_pred

    #         x_pred = alphas[i] * x - betas[i] * v_pred

    #         if use_cache:
    #             noise = self.cached_noise[i+1][:, :frm_len, :].to(device)
    #         else:
    #             noise = torch.randn_like(v_pred)

    #         x = alphas[i + 1] * x_pred + betas[i + 1] * noise

    #     return x


    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", 
            text_cfg_w=1.0,
            **kwargs):
        temporal_aligned_inputs_emb = inputs.get('temporal_aligned_inputs_emb', None)
        prefix_inputs_emb = inputs.get('prefix_inputs_emb', None)
        xattn_inputs_emb = inputs.get('xattn_inputs_emb', None)

        # concat condition.
        local_cond = torch.cat([inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)
        if sampler == "ddim":
            x = self.ddim_sample(inputs['bn'].shape[1], 
                                 timesteps, 
                                 local_cond,
                                 temporal_aligned_inputs_emb,
                                 prefix_inputs_emb,
                                 xattn_inputs_emb,
                                 text_cfg_w=text_cfg_w,
                                 **kwargs)
        # elif sampler == "dpmsolver":
        #     x = self.dpmsolver_sample(timesteps, local_cond, text_embed,
        #                               text_cfg_w=text_cfg_w,
        #                               **kwargs)
        # elif sampler == "plms":
        #     x = self.plms_sample(timesteps, local_cond, text_embed, 
        #             text_cfg_w=text_cfg_w,
        #             **kwargs)
        # elif sampler == "consistency":
        #     x = self.consistency_sample(timesteps, local_cond, text_embed, 
        #             text_cfg_w=text_cfg_w,
        #             **kwargs)
        else:
            raise NotImplementedError

        return x.transpose(1, 2)


if __name__ == "__main__":
    config = ModelArgs(
        prefix_hidden_size=128,
        temporal_aligned_hidden_size=16,
    )
    
    inputs = {}
    inputs["bn"] = torch.randn((4, 125, 32), dtype=torch.bfloat16).to("cuda:0")
    inputs["bn_ctx"] = torch.randn((4, 125, 32), dtype=torch.bfloat16).to("cuda:0")
    inputs["prefix_inputs_emb"] = torch.randn((4, 30, 128), dtype=torch.bfloat16).to("cuda:0")
    inputs["temporal_aligned_inputs_emb"] = torch.randn((4, 125, 16), dtype=torch.bfloat16).to("cuda:0")
    
    from recipes.music_dit.model.loss import MaskedMAELoss
    loss_funcs = MaskedMAELoss()
    with torch.autocast(device_type="cuda", enabled=True):
        breakpoint()
        model = LlamaDiffusion(config).to("cuda:0")
        pred_v, v_target = model(inputs)
        mask = torch.ones(4, 100).to("cuda:0")
        loss = loss_funcs(pred_v, v_target, mask)
        
