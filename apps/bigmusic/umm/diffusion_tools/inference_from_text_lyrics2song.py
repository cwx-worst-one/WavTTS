import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
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


class ARVSampler(nn.Module):
    def __init__(self, in_channels: int, duration: int, num_splits: int):
        super().__init__()
        assert duration % num_splits == 0, "length must be divisible by num_splits"
        self.duration = duration
        self.in_channels = in_channels
        self.num_splits = num_splits

    def set_device(self, device: torch.device):
        self.device = device

    def sample_loop(
        self,
        model: nn.Module,
        semantic_context: torch.tensor,
        current: Tensor,
        prev_noise: Optional[Tensor] = None,
        num_steps: int = 20,
        bf16_portion: float = 0.0,
        first: bool = False,
        show_progress: bool = False,
        angle_schedule: str = "linear",
        schdeule_slope: float = 2.0,
        classifier_free_guidance: int = 1,
    ) -> Tensor:

        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        if angle_schedule == "linear":
            angle_schedule = np.linspace(schdeule_slope, 1.0, num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == "uniform":
            angle_schedule = [1 / num_steps] * num_steps

        B, C, T = current.shape

        init_emb = current.clone()

        sigma = 1.0
        for i in progress_bar:

            sigma_i = torch.ones(B, 1, T, device=self.device) * sigma

            if not first:
                angles = math.pi / 2.0 * sigma_i
                alphas, deltas = torch.cos(angles), torch.sin(angles)
                xt = alphas * init_emb + deltas * prev_noise
                current[:, :, :-625] = xt[:, :, :-625]

            if i < int(num_steps * bf16_portion):
                enabled = True
            else:
                enabled = False

            if isinstance(model, dict):  # ensemble inference
                if sigma_i[0, 0, 0] <= 0.25:
                    key = "2_0"
                elif sigma_i[0, 0, 0] > 0.25 and sigma_i[0, 0, 0] <= 0.5:
                    key = "2_1"
                elif sigma_i[0, 0, 0] > 0.5 and sigma_i[0, 0, 0] <= 0.75:
                    key = "2_2"
                elif sigma_i[0, 0, 0] > 0.75:
                    key = "2_3"
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
                )

                if classifier_free_guidance != 1:
                    v_uncond = diffusion_model(
                        current,
                        timesteps=sigma_i,
                        semantic_context=semantic_context,
                        semantic_force_cfg=1,
                    )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
            omega = math.pi / 2.0 * angle_schedule[i]
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
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = "linear",
        schdeule_slope: float = 2.0,
        classifier_free_guidance=1,
    ) -> Tensor:

        # Sample initial chunks
        b, c, duration = num_items, self.in_channels, self.duration

        # Sample initial chunks
        current_emb = torch.randn(b, c, duration * VOCODER_HZ, device=self.device)
        semantic_hop_size = 125
        diffusion_hop_size = 625
        tmp_emb = torch.zeros(
            b,
            c,
            duration * VOCODER_HZ + diffusion_hop_size * (num_chunks - 1),
            device=self.device,
        )
        avg_cnt = torch.zeros(
            b,
            c,
            duration * VOCODER_HZ + diffusion_hop_size * (num_chunks - 1),
            device=self.device,
        )
        prev_noise = current_emb
        output_emb = []
        for i in range(num_chunks):

            pred_emb = self.sample_loop(
                model=model,
                semantic_context=semantic_context[
                    :,
                    0
                    + (i * semantic_hop_size) : duration * SEMANTIC_HZ
                    + (i * semantic_hop_size),
                ],
                current=current_emb,
                prev_noise=prev_noise[
                    ...,
                    0
                    + (i * diffusion_hop_size) : duration * VOCODER_HZ
                    + (i * diffusion_hop_size),
                ],
                num_steps=num_steps,
                bf16_portion=bf16_portion,
                angle_schedule=angle_schedule,
                classifier_free_guidance=classifier_free_guidance,
                first=(i == 0),
            )

            tmp_emb[
                ...,
                0
                + (i * diffusion_hop_size) : duration * VOCODER_HZ
                + (i * diffusion_hop_size),
            ] += pred_emb
            avg_cnt[
                ...,
                0
                + (i * diffusion_hop_size) : duration * VOCODER_HZ
                + (i * diffusion_hop_size),
            ] += 1

            prev_emb = pred_emb[..., -diffusion_hop_size:]

            new_noise = torch.randn(b, c, diffusion_hop_size, device=self.device)

            prev_noise = torch.cat([prev_noise, new_noise], dim=-1)
            current_emb = torch.cat([prev_emb, new_noise], dim=-1)

        tmp_emb /= avg_cnt
        pred_emb = tmp_emb
        return pred_emb
        # return torch.cat(output_emb, dim=-1)


def init_sampler(
    checkpoint_path,
    local_rank,
    cache_dir,
    duration=30,
    sequence_length=None,  # for backward compat
):
    device = torch.device(f"cuda:{local_rank}")
    sampler = ARVSampler(32, duration, 1)
    sampler.set_device(device)
    return {"sampler": sampler}
