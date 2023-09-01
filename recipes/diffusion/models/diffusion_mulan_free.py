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

    def sample_loop(
            self, 
            model: nn.Module,
            semantic_context: torch.tensor,
            current: Tensor, 
            num_steps: int = 20, 
            bf16_portion: float = 0.,
            chunk_index: int = -1, 
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

        sigma = 1.
        for i in progress_bar:
            if chunk_index < 0:
                sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            else:
                # continuation
                sigma_i = torch.ones(B, 1, 2*self.split_length, device=self.device)
                sigma_i = torch.cat([torch.zeros(B, 1, T-2*self.split_length, device=self.device), sigma_i], -1)
                sigma_i = sigma_i * sigma

            if i < int(num_steps*bf16_portion):
                enabled = True
            else:
                enabled = False

            if sigma_i[0,0,0] <= 0.25:
                key = '2_0'
            elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                key = '2_1'
            elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                key = '2_2'
            elif sigma_i[0,0,0] > 0.75:
                key = '2_3'

            # model prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                v_pred = model(
                    current, 
                    timesteps=sigma_i, 
                    semantic_context=semantic_context,
                    semantic_force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model(
                            current, 
                            timesteps=sigma_i, 
                            semantic_context=semantic_context, 
                            semantic_force_cfg=1,
                        )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    def sample_start(self, 
        model, 
        num_items: int, 
        num_steps: int, 
        bf16_portion: float,
        semantic_context: torch.tensor,
        **kwargs
    ) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Sample start
        return self.sample_loop(
            model=model,
            semantic_context=semantic_context,
            current=torch.randn(b, c, t, device=self.device,),
            num_steps=num_steps, 
            bf16_portion=bf16_portion,
            chunk_index=-1, 
            **kwargs
        )

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        semantic_context: torch.tensor,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        bf16_portion: float,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = 'linear',
        schdeule_slope: float = 2.0,
        classifier_free_guidance = 1,
    ) -> Tensor:
        #assert_message = f"required at least {self.num_splits} chunks"
        #assert num_chunks >= self.num_splits, assert_message
        self.num_chunks = num_chunks

        # Sample initial chunks
        start = self.sample_start(
            model=model,
            num_items=num_items,
            num_steps=num_steps,
            bf16_portion=bf16_portion,
            angle_schedule=angle_schedule,
            classifier_free_guidance=classifier_free_guidance,
            semantic_context=semantic_context,
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

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
