"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import rearrange

VOCODER_HZ = 125
SEMANTIC_HZ = 25

def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = extend_dim(scale, x.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

class Sampler(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        window_length: int, 
        vocoder_hz=None,
        semantic_hz=None,
        outpainting_overlap: int = 0,
    ):
        super().__init__()
        self.window_length = window_length
        self.in_channels = in_channels
        self.vocoder_hz = VOCODER_HZ if vocoder_hz is None else vocoder_hz
        self.semantic_hz = SEMANTIC_HZ if semantic_hz is None else semantic_hz
        self.outpainting_hop_ratio = (1 - outpainting_overlap)

    
    def set_device(self, device: torch.device):
        self.device = device

    def sample_loop(
        self, 
        model: nn.Module,
        current: Tensor, 
        semantic_context: torch.tensor,
        vc_context: Optional[torch.tensor] = None,
        vc_prefix: Optional[torch.tensor] = None,
        num_steps: int = 20, 
        bf16_portion: float = 0.,
        first: bool = False,
        show_progress: bool = False,
        angle_schedule: str = 'linear', 
        schdeule_slope: float = 2.0,
        classifier_free_guidance: int = 1,
    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        if angle_schedule == 'linear':
            angle_schedule = np.linspace(schdeule_slope, 1., num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == 'uniform':
            angle_schedule = [1 / num_steps,] * num_steps

        B, C, T = current.shape
        init_emb = current.clone()

        sigma = 1.
        for i in progress_bar:
    
            sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            if vc_context is not None and vc_prefix is not None:
                sigma_i[:, :, :vc_prefix.shape[-1]] = 0.0
                current[:, :, :vc_prefix.shape[-1]] = vc_prefix

            if not first:
                sigma_i[:, :, :625] = 0.0
                current[:, :, :625] = init_emb[:, :, :625]
     
            if i < int(num_steps*bf16_portion):
                enabled = True
            else:
                enabled = False

            if isinstance(model, dict): # ensemble inference
                if sigma_i[0,0,0] <= 0.25:
                    key = '2_0'
                elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                    key = '2_1'
                elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                    key = '2_2'
                elif sigma_i[0,0,0] > 0.75:
                    key = '2_3'
                diffusion_model: nn.Module = model[key]
            elif isinstance(model, nn.Module):
                diffusion_model = model

            # model prediction
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                v_pred = diffusion_model(
                    current, 
                    timesteps=sigma_i, 
                    semantic_context=semantic_context,
                    semantic_force_cfg=0,
                    vc_context=vc_context,
                    vc_force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = diffusion_model(
                            current, 
                            timesteps=sigma_i, 
                            semantic_context=semantic_context, 
                            semantic_force_cfg=1,
                            vc_context=vc_context,
                            vc_force_cfg=1,
                        )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        semantic_context: torch.tensor,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        bf16_portion: float,
        angle_schedule: str = 'linear',
        schdeule_slope: float = 2.0,
        classifier_free_guidance = 1,
        vc_context: Optional[torch.tensor] = None,
        vc_prefix: Optional[torch.tensor] = None,
    ) -> Tensor:

        # Sample initial chunks
        b, c, window_duration = num_items, self.in_channels, self.window_length

        # Calculate number of chunks
        total_duration = int(math.ceil(semantic_context.shape[-1] / self.semantic_hz))
        outpainting_duration = int(self.outpainting_hop_ratio*window_duration)
        num_outpainting_chunks = int(math.ceil((total_duration - window_duration) / outpainting_duration))
    
        # pad the semantic context with zeros
        outpainting_semantic_len = outpainting_duration*self.semantic_hz
        pad_len = outpainting_semantic_len - (semantic_context.shape[-1] - window_duration*self.semantic_hz) % outpainting_semantic_len
        semantic_context = torch.cat([semantic_context, torch.zeros(b, pad_len, device=self.device).long()], dim=-1)

        # Sample initial chunks
        vocoder_emb_len = int(window_duration*self.vocoder_hz)
        current_emb = torch.randn(b, c, vocoder_emb_len, device=self.device)
        semantic_hop_size = int(self.outpainting_hop_ratio*self.semantic_hz*self.window_length)
        diffusion_hop_size = int(self.outpainting_hop_ratio*self.vocoder_hz*self.window_length)
        tmp_emb = torch.zeros(b, c, vocoder_emb_len + diffusion_hop_size*(num_outpainting_chunks), device=self.device)
        avg_cnt = torch.zeros(b, c, vocoder_emb_len + diffusion_hop_size*(num_outpainting_chunks), device=self.device)
        prev_noise = current_emb

        for i in range(1 + num_outpainting_chunks):
            pred_emb = self.sample_loop(
                model=model,
                current=current_emb,
                semantic_context=semantic_context[:, 0 + (i*semantic_hop_size):int(window_duration*self.semantic_hz) + (i*semantic_hop_size)],
                vc_context=vc_context,
                vc_prefix=vc_prefix,
                num_steps=num_steps,
                bf16_portion=bf16_portion,
                angle_schedule=angle_schedule,
                schdeule_slope=schdeule_slope,
                classifier_free_guidance=classifier_free_guidance,
                first=(i==0),
            )

            tmp_emb[..., 0 + (i*diffusion_hop_size):vocoder_emb_len + (i*diffusion_hop_size)] += pred_emb
            avg_cnt[..., 0 + (i*diffusion_hop_size):vocoder_emb_len + (i*diffusion_hop_size)] += 1

            prev_emb = pred_emb[..., diffusion_hop_size:]

            new_noise = torch.randn(b, c, diffusion_hop_size, device=self.device)
            current_emb = torch.cat([prev_emb, new_noise], dim=-1)

        tmp_emb /= avg_cnt
        pred_emb = tmp_emb
        return pred_emb

def init_sampler(
    checkpoint_path,
    local_rank,
    cache_dir,
    params=None,
    sequence_length=None,   # for backward compat
):
    device = torch.device(f"cuda:{local_rank}")

    sampler = Sampler(
        in_channels=params['latent_dim'], 
        window_length=params['window_length'], 
        vocoder_hz=params['vocoder_hz'],
        semantic_hz=params['semantic_hz'],
        outpainting_overlap=params['outpainting_overlap'],
    )
    sampler.set_device(device)
    return { "sampler": sampler }

class LCMSampler(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        duration: int, 
        num_splits: int,
        vocoder_hz: int = VOCODER_HZ,
        semantic_hz: int = SEMANTIC_HZ,
    ):
        super().__init__()
        assert duration % num_splits == 0, "length must be divisible by num_splits"
        self.duration = duration
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.vocoder_hz = vocoder_hz
        self.semantic_hz = semantic_hz

    def set_device(self, device: torch.device):
        self.device = device

    def sample_loop(
        self, 
        model: nn.Module,
        current: Tensor, 
        semantic_context: torch.tensor,
        vc_context: Optional[torch.tensor] = None,
        vc_prefix: Optional[torch.tensor] = None,
        num_steps: int = 20, 
        bf16_portion: float = 0.,
        first: bool = False,
        show_progress: bool = False,
        angle_schedule: str = 'uniform', 
        schdeule_slope: float = 2.0,
        classifier_free_guidance: int = 1,
    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        if angle_schedule == 'linear':
            angle_schedule = np.linspace(schdeule_slope, 1., num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == 'uniform':
            angle_schedule = [1 / num_steps,] * num_steps

        B, C, T = current.shape
        init_emb = current.clone()

        guidance_scale = classifier_free_guidance*torch.ones(size=(B, 1, 1), device=self.device)

        sigma = 1.
        for i in progress_bar:
    
            sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            if vc_context is not None and vc_prefix is not None:
                sigma_i[:, :, :vc_prefix.shape[-1]] = 0.0
                current[:, :, :vc_prefix.shape[-1]] = vc_prefix

            # if not first:
            #     angles = math.pi /2. * sigma_i
            #     alphas, deltas = torch.cos(angles), torch.sin(angles)
            #     xt = alphas * init_emb + deltas * prev_noise
            #     current[:, :, :-625] = xt[:, :, :-625]
     
            if i < int(num_steps*bf16_portion):
                enabled = True
            else:
                enabled = False

            if isinstance(model, dict): # ensemble inference
                if sigma_i[0,0,0] <= 0.25:
                    key = '2_0'
                elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                    key = '2_1'
                elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                    key = '2_2'
                elif sigma_i[0,0,0] > 0.75:
                    key = '2_3'
                diffusion_model: nn.Module = model[key]
            elif isinstance(model, nn.Module):
                diffusion_model = model

            # model prediction
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                v_pred = diffusion_model(
                    current, 
                    timesteps=sigma_i, 
                    semantic_context=semantic_context,
                    semantic_force_cfg=0,
                    guidance_scale=guidance_scale
                )
            
            omega = math.pi / 2. * sigma
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            sigma = sigma - angle_schedule[i]
            # add noise if num_steps > 1
            if num_steps > 1:
                noise = torch.randn(v_pred.shape).to(v_pred.device)
                omega = math.pi / 2. * sigma
                current = np.cos(omega) * current + np.sin(omega) * noise
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        semantic_context: torch.tensor,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        bf16_portion: float,
        angle_schedule: str = 'uniform',
        schdeule_slope: float = 2.0,
        classifier_free_guidance = 1,
        vc_context: Optional[torch.tensor] = None,
        vc_prefix: Optional[torch.tensor] = None,
    ) -> Tensor:

        # Sample initial chunks
        b, c, duration = num_items, self.in_channels, self.duration
        
        # Sample initial chunks
        current_emb = torch.randn(b, c, duration*self.vocoder_hz, device=self.device)
        semantic_hop_size = 125
        diffusion_hop_size = 625
        tmp_emb = torch.zeros(b, c, duration*self.vocoder_hz + diffusion_hop_size *(num_chunks - 1), device=self.device)
        avg_cnt = torch.zeros(b, c, duration*self.vocoder_hz + diffusion_hop_size *(num_chunks - 1), device=self.device)
        for i in range(num_chunks):

            pred_emb = self.sample_loop(
                model=model,
                current=current_emb,
                semantic_context=semantic_context[:, 0 + (i*semantic_hop_size):duration*self.semantic_hz + (i*semantic_hop_size)],
                vc_context=vc_context,
                vc_prefix=vc_prefix,
                num_steps=num_steps,
                bf16_portion=bf16_portion,
                angle_schedule=angle_schedule,
                schdeule_slope=schdeule_slope,
                classifier_free_guidance=classifier_free_guidance,
                first=(i==0),
            )

            tmp_emb[..., 0 + (i*diffusion_hop_size):duration*self.vocoder_hz + (i*diffusion_hop_size)] += pred_emb
            avg_cnt[..., 0 + (i*diffusion_hop_size):duration*self.vocoder_hz + (i*diffusion_hop_size)] += 1

            prev_emb = pred_emb[..., -diffusion_hop_size:]

            new_noise = torch.randn(b, c, diffusion_hop_size, device=self.device)
            current_emb = torch.cat([prev_emb, new_noise], dim=-1)

        tmp_emb /= avg_cnt
        pred_emb = tmp_emb
        return pred_emb
