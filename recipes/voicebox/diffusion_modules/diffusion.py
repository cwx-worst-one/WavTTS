import torch
from torch import nn, Tensor
from torch.nn import functional as F
from collections import deque
import numpy as np
from functools import partial
from tqdm import tqdm
import math
from typing import Tuple, Optional
from einops import repeat

from .diff_unet import CondUNet, DualCondUNet
from .dpdnet import DPDNet
from .ECAPA_TDNN.model import ECAPA_TDNN
from .loss import MaskedMAELoss, MaskedMSELoss
from .dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


    
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.precompute_diffusion_parameters(hp)
        self.net = CondUNet(
                dim = hp.dim,
                in_channels = hp.in_channels,
                channels = hp.channels,
                factors = hp.factors,
                items = hp.items,
                attentions = hp.attentions,
                attention_features = hp.attention_features,
                attention_heads = hp.attention_heads,
                embedding_features = hp.embedding_features,
                spk_features = hp.spk_embed_dim,
                resnet_groups = hp.resnet_groups,
                modulation_features = hp.modulation_features,
                embedding_max_length = hp.embedding_max_length,
                out_channels = hp.out_channels
                )

        self.out_dim = hp.out_channels
        self.hp = hp
        self.diffusion_type = hp.diffusion_type 

    def precompute_diffusion_parameters(self, dec_cfg):
        if dec_cfg.schedule_type == "linear":
            betas = np.linspace(dec_cfg.min_beta, dec_cfg.max_beta, dec_cfg.timesteps)
        elif dec_cfg.schedule_type == "cosine":
            betas = betas_for_alpha_bar(dec_cfg.timesteps,
                    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.timesteps = len(betas)
        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 先算x_0, 方便进行clip
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.diff(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_normal(self, x, t, cond):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                t, x.shape) 
    
    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # normlize to -1 ~ 1
    def norm_spec(self, x):
        return x / 4.0
    
    def denorm_spec(self, x):
        return x * 4.0

    def clip_sample(self, cond, clip_denoised=True):
        t = self.timesteps
        batch_size = cond.size(0)
        device = cond.device
        x = torch.randn([batch_size, 1, self.out_dim, cond.size(-1)], device=device) 
        for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
            x = self.p_sample(x, torch.full((batch_size,), i, device=device, dtype=torch.long), cond, clip_denoised)
        return x


    def sample_velocity(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def forward(self, local_cond, 
            global_cond, 
            z0, 
            mask=None):
        # local_cond: [B, C1, T]
        # global_cond: [B, C1]
        # z0: [B, C2, T]
        # mask: [B, T]

        device = local_cond.device
        batch_size = local_cond.size(0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        norm_z0 = self.norm_spec(z0)
        noise = torch.randn_like(norm_z0)
        x_noisy = self.q_sample(norm_z0, t, noise)

        if self.hp.diffusion_type == "noise":
            x_recon = self.net(x_noisy, features=global_cond, time=t, embedding=local_cond)
            loss = ((noise - x_recon).abs() * mask.unsqueeze(1)).mean()
        elif self.hp.diffusion_type == "velocity":
            v_target = self.sample_velocity(norm_z0, t, noise)
            pred_v = self.net(x_noisy, features=global_cond, time=t, embedding=local_cond)
            loss = ((pred_v - v_target).abs() * mask.unsqueeze(1)).mean()
        else:
            raise NotImplementedError

        return loss
    
    
    @torch.no_grad()
    def plms_sample(self, local_cond, global_cond, speedup):

        def p_sample_plms(x, t, interval, local_cond, global_cond):

            def get_x_pred(x0, noise):
                x_pred = sqrt_alphas_cumprod_prev * x0 + \
                        sqrt_one_minus_alphas_cumprod_prev * noise
                return x_pred

            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
            alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
            alphas_cumprod_prev = extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)

            pred_list = self.pred_list
            pred = self.net(x, features=global_cond, time=t, embedding=local_cond)

            if self.diffusion_type == "noise":
                x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred) / sqrt_alphas_cumprod_t
                noise = pred
            elif self.diffusion_type == "velocity":
                x0 = sqrt_alphas_cumprod_t * x - \
                        sqrt_one_minus_alphas_cumprod_t * pred
                noise = sqrt_one_minus_alphas_cumprod_t * x + \
                        sqrt_alphas_cumprod_t * pred

            if len(pred_list) == 0:
                x_pred = get_x_pred(x0, noise)
                pred_prev = self.net(x_pred, features=global_cond, time=max(t-interval, 0), embedding=local_cond)
                pred_prime = (pred + pred_prev) / 2
            elif len(pred_list) == 1:
                pred_prime = (3 * pred - pred_list[-1]) / 2
            elif len(pred_list) == 2:
                pred_prime = (23 * pred - 16 * pred_list[-1] + 5 * pred_list[-2]) / 12
            elif len(pred_list) >= 3:
                pred_prime = (55 * pred - 59 * pred_list[-1] + 37 * pred_list[-2] - 9 * pred_list[-3]) / 24

            if self.diffusion_type == "noise":
                x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_prime) / sqrt_alphas_cumprod_t
                noise = pred_prime
            elif self.diffusion_type == "velocity":
                x0 = sqrt_alphas_cumprod_t * x - \
                        sqrt_one_minus_alphas_cumprod_t * pred_prime
                noise = sqrt_one_minus_alphas_cumprod_t * x + \
                        sqrt_alphas_cumprod_t * pred_prime

            x_prev = get_x_pred(x0, noise)
            pred_list.append(pred)

            return x_prev

        t = self.timesteps
        self.pred_list = deque(maxlen=4)
        batch_size = local_cond.size(0)
        device = local_cond.device
        frm_len = local_cond.size(-1)

        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)
        for i in tqdm(reversed(range(0, t, speedup))):
            x = p_sample_plms(x, 
                    torch.full((batch_size,), i, device=device, dtype=torch.long),
                    speedup,
                    local_cond,
                    global_cond)
        return x
     
    @torch.no_grad()
    def ddim_sample_bkp(self, local_cond, global_cond, speedup, eta=0.8,
            cfg_w_global=2.0, global_cond1=None):

        def p_sample_ddim(x, t, interval, local_cond, global_cond):
            def get_x_pred(x0, noise_t, t):
                sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_prev= extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_t= extract(self.alphas_cumprod, t, x.shape)

                if eta > 0:
                    sigma = eta * sqrt_one_minus_alphas_cumprod_prev / sqrt_one_minus_alphas_cumprod_t * torch.sqrt(1 - alphas_cumprod_t/alphas_cumprod_prev)

                    noise = torch.randn_like(noise_t)
                    x_pred = sqrt_alphas_cumprod_prev * x0 + \
                            torch.sqrt(1 - alphas_cumprod_prev - sigma**2) * noise_t + \
                            sigma * noise
                else:
                    x_pred = sqrt_alphas_cumprod_prev * x0 + \
                            torch.sqrt(1 - alphas_cumprod_prev) * noise_t
                return x_pred

            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            if self.diffusion_type == "noise":
                noise_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            elif self.diffusion_type == "velocity":
                if global_cond1 is None:
                    v_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                else:
                    v_pred_global_cond = self.net(x, features=global_cond, time=t, embedding=local_cond)
                    v_pred_uncond = self.net(x, features=global_cond1, time=t, embedding=local_cond)
                    v_pred = cfg_w_global * v_pred_global_cond + (1 - cfg_w_global) * v_pred_uncond

                noise_pred = sqrt_one_minus_alphas_cumprod_t * x + \
                        sqrt_alphas_cumprod_t * v_pred
                x0_pred = sqrt_alphas_cumprod_t * x - \
                        sqrt_one_minus_alphas_cumprod_t * v_pred

            elif self.diffusion_type == "x0":
                pass

            x_prev = get_x_pred(x0_pred, noise_pred, t)
            return x_prev
        
        t = self.timesteps
        batch_size = local_cond.size(0)
        device = local_cond.device
        frm_len = local_cond.size(-1)

        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)
        for i in tqdm(reversed(range(0, t, speedup))):
            x = p_sample_ddim(x, 
                    torch.full((batch_size,), i, device=device, dtype=torch.long),
                    speedup,
                    local_cond,
                    global_cond)
        return x

    @torch.no_grad()
    def max_variance_sample(self, local_cond, global_cond, speedup):

        def p_sample(x, t, interval, local_cond, global_cond):

            def get_x_pred(x0, t):
                sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_prev= extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_t= extract(self.alphas_cumprod, t, x.shape)

                noise = torch.randn_like(x0)
                x_pred = sqrt_alphas_cumprod_prev * x0 + \
                        torch.sqrt(1 - alphas_cumprod_prev) * noise
                return x_pred

            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            if self.diffusion_type == "noise":
                noise_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            elif self.diffusion_type == "velocity":
                v_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                x0_pred = sqrt_alphas_cumprod_t * x - \
                        sqrt_one_minus_alphas_cumprod_t * v_pred

            x_prev = get_x_pred(x0_pred, t)
            return x_prev
        
        t = self.timesteps
        batch_size = local_cond.size(0)
        device = local_cond.device
        frm_len = local_cond.size(-1)

        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)
        for i in tqdm(reversed(range(0, t, speedup))):
            x = p_sample(x, 
                    torch.full((batch_size,), i, device=device, dtype=torch.long),
                    speedup,
                    local_cond,
                    global_cond)
        return x

    @torch.no_grad()
    def ddim_sample(self, local_cond, global_cond, speedup, eta=0.8,
            cfg_w_global=2.0, global_cond1=None, inpaint=None, inpaint_cond=None):

        def p_sample_ddim(x, t, interval, local_cond, global_cond):
            def get_x_pred(x0, noise_t, t):
                sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_prev= extract(self.alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, torch.max(t-interval, torch.zeros_like(t)), x.shape)
                alphas_cumprod_t= extract(self.alphas_cumprod, t, x.shape)

                if eta > 0:
                    sigma = eta * sqrt_one_minus_alphas_cumprod_prev / sqrt_one_minus_alphas_cumprod_t * torch.sqrt(1 - alphas_cumprod_t/alphas_cumprod_prev)

                    noise = torch.randn_like(noise_t)
                    x_pred = sqrt_alphas_cumprod_prev * x0 + \
                            torch.sqrt(1 - alphas_cumprod_prev - sigma**2) * noise_t + \
                            sigma * noise
                else:
                    x_pred = sqrt_alphas_cumprod_prev * x0 + \
                            torch.sqrt(1 - alphas_cumprod_prev) * noise_t
                return x_pred

            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            if self.diffusion_type == "noise":
                noise_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            elif self.diffusion_type == "velocity":
                if global_cond1 is None:
                    v_pred = self.net(x, features=global_cond, time=t, embedding=local_cond)
                else:
                    v_pred_global_cond = self.net(x, features=global_cond, time=t, embedding=local_cond)
                    v_pred_uncond = self.net(x, features=global_cond1, time=t, embedding=local_cond)
                    v_pred = cfg_w_global * v_pred_global_cond + (1 - cfg_w_global) * v_pred_uncond

                noise_pred = sqrt_one_minus_alphas_cumprod_t * x + \
                        sqrt_alphas_cumprod_t * v_pred
                x0_pred = sqrt_alphas_cumprod_t * x - \
                        sqrt_one_minus_alphas_cumprod_t * v_pred

            elif self.diffusion_type == "x0":
                pass

            x_prev = get_x_pred(x0_pred, noise_pred, t)
            return x_prev

        t = self.timesteps
        batch_size = local_cond.size(0)
        device = local_cond.device
        frm_len = local_cond.size(-1)

        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)
        if inpaint is not None:
            inpaint = self.norm_spec(inpaint)
            inpaint_x = torch.randn_like(inpaint, device=device)
            #inpaint_x = self.q_sample(inpaint, 
            #        torch.full((batch_size,), t, device=device, dtype=torch.long),
            #        inpaint_x)
            x = torch.cat([inpaint_x, x], -1)
            local_cond = torch.cat([inpaint_cond, local_cond], -1)

        for i in tqdm(reversed(range(0, t, speedup))):
            x = p_sample_ddim(x, 
                    torch.full((batch_size,), i, device=device, dtype=torch.long),
                    speedup,
                    local_cond,
                    global_cond)

            if inpaint is not None:
                inpaint_noise = torch.randn_like(inpaint, device=device)
                inpaint_x = self.q_sample(inpaint, 
                        torch.full((batch_size,), i, device=device, dtype=torch.long),
                        inpaint_noise)
                x[:, :, :inpaint.size(-1)] = inpaint_x

        if inpaint is not None:
            x = x[:, :, -frm_len:]
        return x
    
    def inference(self, local_cond, global_cond, speedup, **kwargs):
        #x = self.plms_sample(local_cond, global_cond, speedup)
        x = self.ddim_sample(local_cond, global_cond, speedup, **kwargs)
        #x = self.max_variance_sample(local_cond, global_cond, speedup, **kwargs)
        x = self.denorm_spec(x.transpose(1, 2))
        return x


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

def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))
    
    
class VDiffusion(Diffusion):
    def __init__(self, hp):
        super().__init__(hp)

        net_name = hp.net_name if hasattr(hp, "net_name") else "CondUNet"
        print(net_name)
        print(hp.diffusion_use_ctiga)
        if net_name == "DualCondUNet" or net_name == "DualCondUNet2" \
                or net_name == "CondUNet":
            from .diff_unet import MODEL_MAP
            self.net = MODEL_MAP[net_name](
                    dim = hp.dim,
                    in_channels = hp.in_channels,
                    channels = hp.channels,
                    factors = hp.factors,
                    items = hp.items,
                    attentions = hp.attentions,
                    attention_features = hp.attention_features,
                    attention_heads = hp.attention_heads,
                    embedding_features = hp.embedding_features,
                    spk_features = hp.spk_embed_dim,
                    resnet_groups = hp.resnet_groups,
                    modulation_features = hp.modulation_features,
                    embedding_max_length = hp.embedding_max_length,
                    out_channels = hp.out_channels,
                    use_ctiga = hp.diffusion_use_ctiga,
                    use_positional_embedding=hp.use_positional_embedding
                    )
        elif net_name == "DPDNet":
            self.net = DPDNet(
                    input_dim=hp.out_channels,
                    feature_dim=hp.dpd_feature_dim,
                    cond_dim=hp.dpd_cond_dim,
                    cond_bn_dim=hp.dpd_cond_bn_dim,
                    num_blocks=hp.dpd_num_blocks,
                    segment_size=hp.dpd_segment_size,
                    segment_stride=hp.dpd_segment_stride,
                    dropout=hp.dpd_dropout,
                    intra_seq2seq=hp.dpd_intra_seq2seq,
                    inter_seq2seq=hp.dpd_inter_seq2seq
                    )
        else:
            raise NotImplementedError

        self.net_name = net_name

        self.sigma_distribution = UniformDistribution()
    
    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta
    
    def forward(self, local_cond, 
            global_cond, 
            z0, 
            local_cond2=None,
            mask=None,
            mask2=None):

        # local_cond: [B, C1, T]
        # global_cond: [B, C1]
        # z0: [B, C2, T]
        # mask: [B, T]

        batch_size, device = local_cond.size(0), local_cond.device

        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=z0.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        x = z0
        noise = torch.randn_like(x)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x

        if self.net_name == "CondUNet":
            pred_v = self.net(x_noisy, 
                    features=global_cond, 
                    time=sigmas, embedding=local_cond,
                    x_mask=mask)
        elif self.net_name == "DualCondUNet" or self.net_name == "DualCondUNet2":
            pred_v = self.net(x_noisy, 
                    features=global_cond, 
                    time=sigmas, embedding=local_cond,
                    aux_condition=local_cond2,
                    x_mask=mask,
                    aux_mask=mask2
                    )
        elif self.net_name == "DPDNet":
            pred_v = self.net(x_noisy,
                    sigmas, context=local_cond.transpose(1, 2), prompt=global_cond)
        else:
            raise NotImplementedError
        
        return pred_v, v_target

    @torch.no_grad()
    def ddim_sample(self, local_cond, global_cond, timesteps,
            cfg_w_global=2.0, inpaint_x=None, eta=0.0, 
            local_cond2=None):

        t = timesteps
        batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(-1)
        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)

        if t > 20:
            sigmas = torch.linspace(1.0, 0.0, t+1, device=device)
        else:
            sigmas = torch.linspace(1.0, 0.0, t+1, device=device)**2
        sigmas = repeat(sigmas, "i -> i b", b=batch_size)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)


        for i in tqdm(range(t)):
            v_pred = self.net(x, features=global_cond, time=sigmas[i], embedding=local_cond,
                aux_condition=local_cond2)

            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            x_pred = alphas[i] * x - betas[i] * v_pred
            noise_pred = betas[i] * x + alphas[i] * v_pred

            #if inpaint_x is not None:
            #    x_pred[:, :, :inpaint_x.shape[-1]] = inpaint_x
            #    noise_pred[:, :, :inpaint_x.shape[-1]] = x[:, :, :inpaint_x.shape[-1]] - inpaint_x

            if eta > 0:
                sigma = eta * betas[i + 1] / betas[i] * torch.sqrt(1 - (alphas[i]/alphas[i+1])**2)
                noise = torch.randn_like(noise_pred)
                x = alphas[i + 1] * x_pred + \
                        torch.sqrt(betas[i + 1]**2 - sigma**2) * noise_pred + \
                        sigma * noise
            else:
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x
    
    @torch.no_grad()
    def dpmsolver_sample(self, local_cond, global_cond, timesteps, local_cond2=None):
        batch_size, device, frm_len = local_cond.size(0), local_cond.device, local_cond.size(-1)
        x = torch.randn([batch_size, self.out_dim, frm_len], device=device)
        noise_schedule = NoiseScheduleVP(schedule="cosine")

        def my_wrapper(fn):
            def wrapped(x, t, **kwargs):
                out = fn(x, time=t, **kwargs)
                return out
            return wrapped

        model_fn = model_wrapper(
                my_wrapper(self.net),
                noise_schedule,
                model_type="v",
                model_kwargs={"features": global_cond, "embedding": local_cond, 
                    "aux_condition": local_cond2}
                )
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        x = dpm_solver.sample(
                x,
                t_end=0.008,
                steps=timesteps,
                order=2,
                skip_type="time_uniform",
                method="multistep")
        x = x.squeeze(1)
        return x

    
    def inference(self, local_cond, global_cond, timesteps, sampler, **kwargs):
        if sampler == "ddim":
            x = self.ddim_sample(local_cond, global_cond, timesteps, **kwargs)
        elif sampler == "dpmsolver":
            x = self.dpmsolver_sample(local_cond, global_cond, timesteps, **kwargs)
        else:
            raise NotImplementedError
        return x

