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

from recipes.bigmusic.lightning.embedding_modules import TagCategoricalEmbedder
from recipes.music_dit.model.util_layers import RMSNorm
from recipes.music_dit.model.loss import MaskedMAELoss, sequence_mask
from recipes.music_dit.model.based_ctiga_llama import ModelArgs as LLamaArgs, LLaMa
from recipes.music_dit.model.a_unet.blocks import NumberEmbedder, Repeat
from recipes.music_dit.model.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
logger = logging.getLogger(__name__)



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

    local_cond_dim: int = 1536
    time_embed_dim: int = 1536

    # token
    n_token: int = 32768
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
    out_channels :  int = 80
    causal: bool = False
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda : [-1, -1]) 
    window_type: str = "elemwise" # elemwise, blockwise

    # speaker encoder
    prompt_mel_dim: int = 80
    spk_e_dim: int = 1024
    spk_embed_dim : int = 512
    prompt_loss_weight: float = 0.2 # deprecated
    cfg_drop_spk_emb: bool = True
    
    llama_provider: str = "ctiga"
    postnet_type: str = "linear" # conv
    postnet_kernel: int = 3

    # features ["mel", "bn"]
    target: str = "mel"
    prompt_feature: str = "mel"
    ctx_feature: str = "mel"
    in_channels: int = 80
    out_channels :  int = 80
    use_textprefix: bool = True
    text_prefix_type: str = 'text'
    x_padding_value: int = -2

    bias: bool = False
    use_unet_style_skip_connect: bool = False
    target_type: str = "velocity"
    use_prompt: bool = True
    use_token: bool = True
    
    mask_token: bool = True  # according to speech, mask token is used to resolve train inference mismatch

    min_t: float = 0.0
    max_t: float = 1.0


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

        self.use_phone_lang = hp.use_phone_lang if hasattr(hp, "use_phone_lang") else False
        # text.
        padding_idx = 0
        n_phone = 1024
        self.frontend_embed = nn.Embedding(
            n_phone,
            hp.encoder_dim,
            padding_idx=padding_idx)

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

        self.x_prenet = nn.Linear(hp.in_channels * 2 + 1, hp.encoder_dim, bias=self.bias)

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
        
        logger.info(f"prompt_feature: {self.hp.prompt_feature}")
        logger.info(f"cxt_feature: {self.hp.ctx_feature}")
        logger.info(f"target_feature: {self.hp.target}")
        logger.info(f"use_textprefix: {self.hp.use_textprefix}")
        logger.info(f"use_prompt: {self.hp.use_prompt}")
    
    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, inputs):
        # diffusion target
        x = inputs['bn']
        feat_lens = inputs['seqlen']
        B, device = x.size(0), x.device

        sigmas = self.sigma_distribution(num_samples=B, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # time embedding
        time_emb = self.time_embedding(sigmas)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        noise = torch.randn_like(x)
        x_noisy = alphas * x + betas * noise
        residual = x_noisy
        if self.target_type == "velocity":
            target = alphas * noise - betas * x
        elif self.target_type == "x0":
            target = x
        elif self.target_type == "noise":
            target = noise

        # concat condition.
        x_noisy = self.x_prenet(torch.cat([x_noisy, inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)) \
                  + self.prenet(time_emb)

        # concat prefix-text.
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["text_tokens"])  # [B, T, 1024]
            text_lens = inputs["text_lens"]

            T = (feat_lens + text_lens).max()
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full([B, T, C], self.hp.x_padding_value, device=device, dtype=x_noisy.dtype)

            x_noisy_wtext = alphas * x_noisy_wtext + betas * torch.randn_like(x_noisy_wtext)

            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = torch.arange(text_embed.shape[1], device=device)[None, :] < text_lens[:, None]
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)

            mask_x = (text_lens[:, None] <= indics_x) & (indics_x < (text_lens + feat_lens)[:, None]) & (
                    indics_x - text_lens[:, None] < T_feat)
            mask_noisy = torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)

            encoder_input = x_noisy_wtext
            seq_mask = sequence_mask(feat_lens + text_lens, device=feat_lens.device)

        pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask)

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(B, x.shape[1], pred_v.shape[-1], device=device)

            T0 = x.shape[1]
            T1 = pred_v.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & ((text_lens[:, None] + indics0) < T1)
            mask1 = (text_lens[:, None] <= indics1) & (indics1 < (text_lens[:, None] + feat_lens[:, None]))
            pred_v_wotext[mask0] = pred_v[mask1].to(dtype=pred_v_wotext.dtype)

            pred_v = pred_v_wotext

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        elif self.target_type == "x0":
            pred = betas * residual + alphas * pred_v

        return pred.transpose(1, 2), target.transpose(1, 2)
    
    def _forward(self, x, local_cond, text_embed, timesteps,
            alphas=None, betas=None):
        residual = x
        time_emb = self.time_embedding(timesteps)
        # TODO or time_emb = time_emb.unsqueeze(1).expand(local_cond.shape[0], local_cond.shape[1], -1)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        # concat condition.
        x = self.x_prenet(torch.cat([x, local_cond], -1)) \
            + self.prenet(time_emb)

        print("_forward1", x.shape)
        if self.hp.use_textprefix:
            x = torch.cat([text_embed, x], dim=1)
        print("_forward2", x.shape)

        pred_v = self.encoder(x, x.shape[1])

        if self.hp.use_textprefix:
            pred_v = pred_v[:, text_embed.shape[1]:, :]

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

    def ddim_sample(self, frm_len, timesteps, local_cond, text_embed,
            text_cfg_w=1.0, inpaint_x=None, use_cache=False, cached_v_len=None,
            eta=0.0):
        t = timesteps
        batch_size, device = text_embed.size(0), text_embed.device
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
                    v_pred, v_pred_uncond = self._forward(x.repeat(batch_size, 1, 1), local_cond, text_embed,
                                                          timesteps=sigmas[i].expand(batch_size, -1)).chunk(2)
                    v_pred = text_cfg_w * v_pred + (1 - text_cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])

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

    def dpmsolver_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
        batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        noise_schedule = NoiseScheduleVP(schedule="cosine")

        def my_wrapper(fn):
            def wrapped(x, t, **kwargs):
                if text_cfg_w > 1:
                    out, uncond_out = fn(x.expand(batch_size, -1, -1), 
                            timesteps=t.expand(batch_size), **kwargs).chunk(2)
                    out = text_cfg_w * out + (1 - text_cfg_w) * uncond_out
                else:
                    out = fn(x, timesteps=t, **kwargs)
                return out
            return wrapped

        model_fn = model_wrapper(
                my_wrapper(self._forward),
                noise_schedule,
                model_type="v",
                model_kwargs={"local_cond": local_cond, "text_embed": text_embed}
                )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True) # dpmsolver++
        #dpm_solver = DPM_Solver(model_fn, noise_schedule)
        x = dpm_solver.sample(
                x,
                t_end=0.001,
                steps=timesteps,
                order=2,
                skip_type="logSNR",
                method="multistep")
        return x
    
    def plms_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
        t = timesteps
        batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        if t > 20:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        else:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)**2
            #sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        pred_list = []
        for i in tqdm(range(t)):
            if text_cfg_w > 1:
                pred, pred_uncond = self._forward(x, local_cond, text_embed, 
                        timesteps=sigmas[i].expand(batch_size, -1)).chunk(2)
                pred = text_cfg_w * pred + (1 - text_cfg_w) * pred_uncond
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
                x_noisy = alphas[i+1] * x_pred + betas[i+1] * noise_pred
                if text_cfg_w > 1:
                    pred_prev, pred_prev_uncond = self._forward(x_noisy, local_cond, text_embed, 
                                timesteps=sigmas[i+1].expand(batch_size, -1)).chunk(2)
                    pred_prev = text_cfg_w * pred_prev + (1 - text_cfg_w) * pred_prev_uncond
                else:
                    pred_prev = self._forward(x_noisy, local_cond, text_embed, timesteps=sigmas[i+1]) 
                pred_prime = (pred + pred_prev) / 2
            elif len(pred_list) == 1:
                pred_prime = (3 * pred - pred_list[-1]) / 2
            elif len(pred_list) == 2:
                pred_prime = (23 * pred - 16 * pred_list[-1] + 5 * pred_list[-2]) / 12
            elif len(pred_list) >= 3:
                pred_prime = (55 * pred - 59 * pred_list[-1] + 37 * pred_list[-2] - 9 * pred_list[-3]) / 24

            if self.target_type == "velocity":
                x_pred_prime = alphas[i] * x - betas[i] * pred_prime
                noise_pred_prime = betas[i] * x + alphas[i] * pred_prime
            elif self.target_type == "noise":
                x_pred_prime = (x - betas[i] * pred_prime) / alphas[i]
                noise_pred_prime = pred
            else:
                raise NotImplementedError
            
            x = alphas[i+1] * x_pred_prime + betas[i+1] * noise_pred_prime
            pred_list.append(pred)
        return x

    def consistency_sample(self, timesteps, local_cond, text_embed, 
            text_cfg_w=1.0, use_cache=False, cached_v_len=None):
        t = timesteps
        batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)

        if use_cache:
            x = self.cached_noise[0][:, :frm_len, :].to(device)
        else:
            x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        if t > 20:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        else:
            sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)**2
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
            
        for i in range(t):

            v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i]) 

            # TODO: 只是模拟cache过程
            if use_cache:
                if self.cached_v[i] is not None:
                    cached_v_len = self.cached_v[i].shape[1] if cached_v_len is None else cached_v_len
                    v_pred[:, :cached_v_len, :] = self.cached_v[i][:, :cached_v_len, :]
                self.cached_v[i] = v_pred

            x_pred = alphas[i] * x - betas[i] * v_pred

            if use_cache:
                noise = self.cached_noise[i+1][:, :frm_len, :].to(device)
            else:
                noise = torch.randn_like(v_pred)

            x = alphas[i + 1] * x_pred + betas[i + 1] * noise

        return x


    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", 
            text_cfg_w=1.0,mask_token=False,
            **kwargs):

        if self.hp.use_textprefix:
            if self.hp.text_prefix_type == 'leadsheet':
                text_embed = self.frontend_embed(inputs["leadsheet_token"])  # [B, T, 1024]
            else:
                text_embed = self.frontend_embed(inputs["lyrics_tokens"])  # [B, T, 1024]
        else:
            text_embed = None
        # concat condition.
        local_cond = torch.cat([inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)
        if sampler == "ddim":
            x = self.ddim_sample(inputs['bn'].shape[1], timesteps, local_cond, text_embed,
                                 text_cfg_w=text_cfg_w,
                                 **kwargs)
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(timesteps, local_cond, text_embed,
                                      text_cfg_w=text_cfg_w,
                                      **kwargs)
        elif sampler == "plms":
            x = self.plms_sample(timesteps, local_cond, text_embed, 
                    text_cfg_w=text_cfg_w,
                    **kwargs)
        elif sampler == "consistency":
            x = self.consistency_sample(timesteps, local_cond, text_embed, 
                    text_cfg_w=text_cfg_w,
                    **kwargs)
        else:
            raise NotImplementedError

        return x.transpose(1, 2)
    
    
    def infer_one_step(self, inputs, x_noisy, t, prev_t, text_cfg_w, sampler="ddim"):
        local_cond, text_embed, _, feat_lens, text_lens = self.compute_condition(inputs, t)

        B, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(1)
        t_batch = extend_dim(t, dim=x_noisy.ndim)
        alphas, betas = self.get_alpha_beta(t_batch)
        prev_t_batch = extend_dim(prev_t, dim=x_noisy.ndim)
        prev_alphas, prev_betas = self.get_alpha_beta(prev_t_batch)

        residual = x_noisy
        time_emb = self.time_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
        # print(f"{x_noisy.shape=}, {time_emb.shape=}, {local_cond.shape=}")
        x_noisy = self.x_prenet(x_noisy) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        residual = residual.repeat(2, 1 ,1 )
        x_noisy = x_noisy.repeat(2, 1, 1)
        alphas = alphas.repeat(2, 1, 1)
        betas = betas.repeat(2, 1, 1)
        if text_lens is not None:
            text_lens = text_lens.repeat(2)
        feat_lens = feat_lens.repeat(2)
        B *= 2

        # concat prefix-text.
        if self.hp.use_textprefix: 
            T = inputs[f"text_mel_mask"].shape[1]
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full([B, T, C], self.hp.x_padding_value, device=device, dtype=x_noisy.dtype)
            #x_noisy_wtext = alphas * x_noisy_wtext + betas * torch.randn_like(x_noisy_wtext)
            
            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = torch.arange(text_embed.shape[1], device=device)[None, :] < text_lens[:, None]
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)

            mask_x = (text_lens[:, None] <= indics_x) & (indics_x < (text_lens+feat_lens)[:, None]) & (indics_x - text_lens[:, None] < T_feat)
            mask_noisy = torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)
            
            encoder_input = x_noisy_wtext
            seq_mask = inputs["text_mel_mask"]
        else:
            encoder_input = x_noisy
            seq_mask = inputs[f"{self.hp.prompt_feature}_mask"]

        pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask.repeat(2, 1))

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(B, x_noisy.shape[1], pred_v.shape[-1], device=device, dtype=pred_v.dtype)
            
            T0 = x_noisy.shape[1]
            T1 = pred_v.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & ((text_lens[:, None] + indics0) < T1)
            mask1 = (text_lens[:, None] <= indics1) & (indics1 < (text_lens[:, None] + feat_lens[:, None]))
            pred_v_wotext[mask0] = pred_v[mask1]
            
            pred_v = pred_v_wotext

        pred_v = self.postnet(pred_v)

        if self.hp.use_unet_style_skip_connect:
            pred_v = pred_v + residual

        # cfg
        v_pred, v_pred_uncond = pred_v.chunk(2)

        text_cfg_w = text_cfg_w.unsqueeze(1).unsqueeze(2)
        v_pred = text_cfg_w * v_pred + (1 - text_cfg_w) * v_pred_uncond

        B = v_pred.shape[0]
        residual = residual[:B]
        alphas = alphas[:B]
        betas = betas[:B]

        if sampler == "ddim":
            x_pred = alphas * residual - betas * v_pred
            noise_pred = betas * residual + alphas * v_pred
            prev_x = prev_alphas * x_pred + prev_betas * noise_pred
        else:
            return NotImplementedError
        return prev_x

    def compute_condition(self, inputs, t):
        if self.hp.use_textprefix: 
            text_embed = self.frontend_embed(inputs["frontend"]) # [B, T, 1024]
            text_lens = inputs["text_lens"]
        else:
            text_embed, text_lens = None, None
        feat_lens = inputs["mel_lens"] if "mel_lens" in inputs else inputs["bn_lens"]

        B, device = inputs["token"].size(0), inputs["token"].device
        
        # token encoder to align frame-rate.
        token_embed = self.token_embedding(inputs["token"])
        token_embed = token_embed.transpose(1, 2)
        for layer in self.token_prenet:
            token_embed = layer(token_embed)
        token_embed = token_embed.transpose(1, 2) # B, T, C
        
        # speaker embedding
        if self.use_prompt:
            prompt_feature = f"prompt_{self.hp.prompt_feature}"
            spk_emb = self.prompt_encoder(inputs[prompt_feature])
            spk_emb = spk_emb.unsqueeze(1).expand(-1, token_embed.shape[1], -1)

            # local conditioning.
            ctx_feature = f"{self.hp.ctx_feature}_ctx"
            local_cond = torch.cat([
                token_embed, 
                inputs[ctx_feature], 
                spk_emb], dim=-1)
        else:
            local_cond = token_embed
        local_cond = self.local_cond_project(local_cond)

        # time embedding
        time_emb = self.time_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)

        return local_cond, text_embed, time_emb, feat_lens, text_lens

    def pred_x0(self, inputs, x_noisy, t):

        local_cond, text_embed, time_emb, feat_lens, text_lens = self.compute_condition(inputs, t)

        B, device = x_noisy.shape[0], x_noisy.device
        t_batch = extend_dim(t, dim=x_noisy.ndim)
        alphas, betas = self.get_alpha_beta(t_batch)

        residual = x_noisy

        # concat condition.
        x_noisy = self.x_prenet(x_noisy) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        # concat prefix-text.
        if self.hp.use_textprefix: 
            T = inputs[f"text_mel_mask"].shape[1]
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full([B, T, C], self.hp.x_padding_value, device=device, dtype=x_noisy.dtype)
            
            #x_noisy_wtext = alphas * x_noisy_wtext + betas * torch.randn_like(x_noisy_wtext)
            
            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = torch.arange(text_embed.shape[1], device=device)[None, :] < text_lens[:, None]
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)

            mask_x = (text_lens[:, None] <= indics_x) & (indics_x < (text_lens+feat_lens)[:, None]) & (indics_x - text_lens[:, None] < T_feat)
            mask_noisy = torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)
            
            encoder_input = x_noisy_wtext
            seq_mask = inputs["text_mel_mask"]
        else:
            encoder_input = x_noisy
            seq_mask = inputs[f"{self.hp.prompt_feature}_mask"]

        pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask)

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(B, x_noisy.shape[1], pred_v.shape[-1], device=device)
            
            T0 = x_noisy.shape[1]
            T1 = pred_v.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & ((text_lens[:, None] + indics0) < T1)
            mask1 = (text_lens[:, None] <= indics1) & (indics1 < (text_lens[:, None] + feat_lens[:, None]))
            pred_v_wotext[mask0] = pred_v[mask1].to(dtype=pred_v_wotext.dtype)

            pred_v = pred_v_wotext

        pred_v = self.postnet(pred_v)

        if self.hp.use_unet_style_skip_connect:
            pred_v = pred_v + residual

        pred_x0 = alphas * residual - betas * pred_v
        return pred_x0.transpose(1, 2)


class LlamaDiffusionLyric2Song(LlamaDiffusion):
    def __init__(self, hp):
        super().__init__(hp)
        self.style_embedder =  TagCategoricalEmbedder(
            embedding_dim=hp.encoder_dim,
            add_sos=True,
            dropout=0,
            vocab_type='SA',
        )

    def forward(self, inputs):
        # diffusion target
        x = inputs['bn']
        feat_lens = inputs['seqlen']
        B, device = x.size(0), x.device

        sigmas = self.sigma_distribution(num_samples=B, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        # time embedding
        time_emb = self.time_embedding(sigmas)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        noise = torch.randn_like(x)
        x_noisy = alphas * x + betas * noise
        residual = x_noisy
        if self.target_type == "velocity":
            target = alphas * noise - betas * x
        elif self.target_type == "x0":
            target = x
        elif self.target_type == "noise":
            target = noise

        # concat condition.
        x_noisy = self.x_prenet(torch.cat([x_noisy, inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)) \
                  + self.prenet(time_emb)

        # concat prefix-text.
        if self.hp.use_textprefix:
            text_embed = self.frontend_embed(inputs["text_tokens"])  # [B, T, 1024]
            style_embed = self.style_embedder.embed(batch=inputs['style_text'])
            text_embed = torch.cat([style_embed, text_embed], 1)
            text_lens = inputs["text_lens"] + style_embed.shape[1]

            T = (feat_lens + text_lens).max()
            C = x_noisy.shape[-1]
            x_noisy_wtext = torch.full([B, T, C], self.hp.x_padding_value, device=device, dtype=x_noisy.dtype)

            x_noisy_wtext = alphas * x_noisy_wtext + betas * torch.randn_like(x_noisy_wtext)

            T_text = text_embed.shape[1]
            T_feat = x_noisy.shape[1]
            indics_x = torch.arange(T, device=device)[None, :]
            mask_x = (indics_x < text_lens[:, None]) & (indics_x < T_text)
            mask_text = torch.arange(text_embed.shape[1], device=device)[None, :] < text_lens[:, None]
            x_noisy_wtext[mask_x] = text_embed[mask_text].to(dtype=x_noisy_wtext.dtype)

            mask_x = (text_lens[:, None] <= indics_x) & (indics_x < (text_lens + feat_lens)[:, None]) & (
                    indics_x - text_lens[:, None] < T_feat)
            mask_noisy = torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None]
            x_noisy_wtext[mask_x] = x_noisy[mask_noisy].to(dtype=x_noisy_wtext.dtype)

            encoder_input = x_noisy_wtext
            seq_mask = sequence_mask(feat_lens + text_lens, device=feat_lens.device)

        pred_v = self.encoder(encoder_input, encoder_input.shape[1], attention_mask=seq_mask)

        if self.hp.use_textprefix:
            pred_v_wotext = torch.zeros(B, x.shape[1], pred_v.shape[-1], device=device)

            T0 = x.shape[1]
            T1 = pred_v.shape[1]
            indics0 = torch.arange(T0, device=device)[None, :]
            indics1 = torch.arange(T1, device=device)[None, :]

            mask0 = (indics0 < feat_lens[:, None]) & ((text_lens[:, None] + indics0) < T1)
            mask1 = (text_lens[:, None] <= indics1) & (indics1 < (text_lens[:, None] + feat_lens[:, None]))
            pred_v_wotext[mask0] = pred_v[mask1].to(dtype=pred_v_wotext.dtype)

            pred_v = pred_v_wotext

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        elif self.target_type == "x0":
            pred = betas * residual + alphas * pred_v

        return pred.transpose(1, 2), target.transpose(1, 2)

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim",
                  text_cfg_w=1.0, mask_token=False,
                  **kwargs):
        text_embed = self.frontend_embed(inputs["text_tokens"])  # [B, T, 1024]
        style_embed = self.style_embedder.embed(batch=inputs['style_text'])
        text_embed = torch.cat([style_embed, text_embed], 1)
        local_cond = torch.cat([inputs['bn_ctx'], inputs['bn_ctx_mask']], -1)
        if sampler == "ddim":
            x = self.ddim_sample(inputs['bn'].shape[1], timesteps, local_cond, text_embed,
                                 text_cfg_w=text_cfg_w,
                                 **kwargs)
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(timesteps, local_cond, text_embed,
                                      text_cfg_w=text_cfg_w,
                                      **kwargs)
        elif sampler == "plms":
            x = self.plms_sample(timesteps, local_cond, text_embed,
                                 text_cfg_w=text_cfg_w,
                                 **kwargs)
        elif sampler == "consistency":
            x = self.consistency_sample(timesteps, local_cond, text_embed,
                                        text_cfg_w=text_cfg_w,
                                        **kwargs)
        else:
            raise NotImplementedError

        return x.transpose(1, 2)


if __name__ == "__main__":
    config = ModelArgs(text_prefix_type='leadsheet', use_textprefix=False, mask_token=False)
    inputs = {}

    frontend = {
        "phone": torch.randint(0, 1000, (32, 20)).to("cuda:0"),
        "tone": torch.randint(0, 30, (32, 20)).to("cuda:0"),
        "word_seg": torch.randint(0, 8, (32, 20)).to("cuda:0")
    }
    text_lens = torch.randint(0, 20, [32]).to("cuda:0")
    text_lens[0] = 20
    
    mel_ctx = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel_len = torch.randint(0, 100, [32]).to("cuda:0")
    mel_len[0] = 100
    token = torch.randint(0, 1000, (32, 100)).to("cuda:0")

    prompt_mel = torch.randn((32, 80, 50), dtype=torch.bfloat16).to("cuda:0")
    
    text_mel_len = text_lens + mel_len

    mel_mask = sequence_mask(mel_len, max_len=100, device="cuda:0")
    text_mel_mask = sequence_mask(text_mel_len, max_len=120, device="cuda:0")

    inputs = {}
    inputs["token"] = token
    inputs["leadsheet_token"] = token
    inputs["prompt_mel"] = prompt_mel
    inputs["mel"] = mel
    inputs["mel_mask"] = mel_mask
    inputs["mel_ctx"] = mel_ctx
    inputs["frontend"] = frontend
    inputs["text_mel_mask"] = text_mel_mask
    inputs["text_lens"] = text_lens
    inputs["mel_lens"] = mel_len
    inputs["flag_drop"] = [False] * 32
    from recipes.music_dit.modules.loss import MaskedMAELoss, MaskedMSELoss

    loss_funcs = MaskedMAELoss()
    with torch.autocast(device_type="cuda", enabled=True):
        model = LlamaDiffusion(config).to("cuda:0")
        pred_v, v_target = model(inputs)
        mask = torch.ones(32, 100).to("cuda:0")
        loss = loss_funcs(pred_v, v_target, mask)
        
