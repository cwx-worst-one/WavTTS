"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import enum
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Tuple
from einops import rearrange, repeat


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


class UniformDistribution:
    def __init__(self, vmin: float = 0., vmax: float = 1.):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class ARVSampler(nn.Module):
    def __init__(self, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
    
    def set_device(self, device: torch.device):
        self.device = device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * math.pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        #print(b, n, l, i, flush=True)
        n_half = n // 2
        sigmas = torch.linspace(1, 0, i * n_half)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        #return  torch.cat([torch.zeros_like(sigmas),] * (n - 1) + [sigmas,], dim=-1)
        if n % 2:
            return torch.cat([torch.zeros_like(sigmas), torch.zeros_like(sigmas[..., :self.split_length]), sigmas], dim=-1)
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
            self, 
            model: nn.Module,
            mulan_context: torch.tensor,
            current: Tensor, 
            num_steps: int = 20, 
            chunk_index: int = -1, 
            show_progress: bool = False,
            angle_schedule: str = 'linear', 
            classifier_free_guidance: int = 1,
            is_causal: bool = False,

    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        # cond_len = context.shape[-1]
        # cond_stride = cond_len // self.num_splits

        # TODO: revisit for longer sequences
        # if chunk_index == -1:
        #     context = context[..., :cond_len]
        # else:
        #     context = context[..., chunk_index * cond_stride + 3 : chunk_index * cond_stride + cond_len + 3]

        if angle_schedule == 'linear':
            angle_schedule = np.linspace(2., 1., num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == 'uniform':
            angle_schedule = [1 / num_steps,] * num_steps

        B, C, T = current.shape

        sigma = 1.
        for i in progress_bar:
            if chunk_index < 0:
                sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            else:
                # continuation
                sigma_i = torch.ones(B, 1, 2*self.split_length, device=self.device)
                sigma_i = torch.cat([torch.zeros(B, 1, T-2*self.split_length, device=self.device), sigma_i], -1)
                sigma_i = sigma_i * sigma

            # model prediction
            v_pred = model(
                current, 
                timesteps=sigma_i, 
                mulan_context=mulan_context, 
                mulan_force_cfg=0,
                is_causal=is_causal,
            )

            if classifier_free_guidance != 1:
                v_uncond = model(
                    current, 
                    timesteps=sigma_i, 
                    mulan_context=mulan_context, 
                    mulan_force_cfg=1,
                    is_causal=is_causal,
                )
                guidance_scale = classifier_free_guidance
                v_pred = guidance_scale * v_pred + (1 - guidance_scale) * v_uncond
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            #print(omega, flush=True)
            current[:, :32, :] = np.cos(omega) * current[:, :32, :] - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current[:, :32, :]

    def sample_start(self, 
        model, 
        start,
        num_items: int, 
        num_steps: int, 
        mulan_context: torch.tensor, 
        is_causal: bool = False,
        **kwargs
    ) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        if start is None:
            current = torch.randn(b, c, t, device=self.device)
        else:
            current = start
            assert current.shape == (b, c, t), "start shape doesn't match"
        
        return self.sample_loop(
            model=model,
            mulan_context=mulan_context,
            current=current,
            num_steps=num_steps, 
            chunk_index=-1, 
            is_causal=is_causal,
            **kwargs
        )

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        mulan_context: torch.tensor,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = 'linear',
        classifier_free_guidance = 1,
        is_causal: bool = False,
    ) -> Tensor:
        #assert_message = f"required at least {self.num_splits} chunks"
        #assert num_chunks >= self.num_splits, assert_message
        self.num_chunks = num_chunks

        # Sample initial chunks
        start = self.sample_start(
            model=model,
            start=start,
            num_items=num_items,
            num_steps=num_steps,
            angle_schedule=angle_schedule,
            classifier_free_guidance=classifier_free_guidance,
            mulan_context=mulan_context,
            is_causal=is_causal,
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

        # Get sigmas for autoregressive ladder
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        #sigmas = self.get_sigmas_ladder(
        #    num_items=b,
        #    num_steps_per_split=num_steps // self.num_splits,
        #)
        #alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        #print(start.shape, alphas.shape, flush=True)
        #sigma = torch.linspace(0, 1, num_steps, device=self.device).view(1, 1, num_steps).repeat(b, 1, 1)
        #alpha = torch.cos(sigma)
        #beta = torch.sin(sigma)
        #start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start.chunk(chunks=n, dim=-1))
        # Add fresh noise chunk
        shape = (b, self.in_channels, self.split_length)

        # Loop over ladder shifts
        progress_bar = tqdm(range(num_chunks), disable=not show_progress)
        last_delta = torch.linspace(0, math.pi/2, self.split_length, device=model.device).view(1, 1, -1).repeat(b, 1, 1)

        for j in progress_bar:
            #orig_chunk = chunks[-1]
            #chunks[-1] = torch.cos(last_delta) * orig_chunk + torch.sin(last_delta) * torch.randn_like(chunks[-1])
            chunks += [torch.randn(shape, device=model.device), torch.randn(shape, device=model.device)]
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                model=model,
                current=torch.cat(chunks[-n:], dim=-1),
                num_steps=num_steps // n * 2,
                chunk_index=j+2,
                angle_schedule=angle_schedule,
                classifier_free_guidance=classifier_free_guidance,
                context=context,
            )
            # Update chunks
            chunks = chunks[:-1]
            chunks[-n+1:] = list(updated.chunk(chunks=n, dim=-1))[:-1]
            if len(chunks) == self.num_chunks:
                break

        return torch.cat(chunks[:num_chunks], dim=-1)

if __name__ == '__main__':
    from recipes.diffusion.models.dualpath_net import DualPathDiffusionNetwork
    model = DualPathDiffusionNetwork(
        input_dim=16,
        feature_dim=768,
        num_blocks=1,
        num_chunks=4,
        context_dim=1024,
        segment_size=80,
        segment_stride=41,
        diffusion_steps=0,
        dropout=0,
        intra_seq2seq='gru',
        inter_seq2seq='roformer',
        end2end=False,
        use_checkpoint=True
    )
    xt = torch.randn(2, 16, 2500)
    t = torch.randn(2, 1, 2500)
    semantic = torch.randn(2, 250, 1024)
    out = model(xt, t, context=semantic, cfg=True)

    avr = ARVSampler(
        in_channels=16,
        length=5000,
        num_splits=4,
        device=torch.device('cuda')
    )
    test = avr(
        model=model, 
        context=semantic,
        num_items=2, # batch size: how many samples to generate
        num_chunks=4,
        num_steps=20, # diffusion steps?
        start=None,
        show_progress=True,
        angle_schedule='linear',
        classifier_free_guidance=1,
        
    )
    print(test.shape)
