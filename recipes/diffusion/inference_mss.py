import os
import glob
import argparse
import math
from tqdm import tqdm
from time import time
from pathlib import Path
from typing import Any, Optional, Tuple
from string import punctuation

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import torchaudio
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.diffusion.models.diffusion_model.utils import init_diffusio_mss, run_diffusion
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.diffusion.utils.mss_utils import pad_audio, enframe, deframe
torch.backends.cuda.matmul.allow_tf32 = True

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
            context: torch.tensor,
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

            # model prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                v_pred = model(
                    current, 
                    timesteps=sigma_i, 
                    context=context,
                    force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model(
                            current, 
                            timesteps=sigma_i, 
                            context=context, 
                            force_cfg=1,
                        )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current[:, :self.in_channels, :] = np.cos(omega) * current[:, :self.in_channels, :]  - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        context: torch.tensor,
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

        b, c, t = num_items, self.in_channels, self.length
        # Sample initial chunks
        start_emb = torch.cat([
            torch.randn(b, c, t, device=self.device,),
            context,
        ], dim=1)

        start = self.sample_loop(
            model=model,
            context=context.permute(0, 2, 1),
            current=start_emb,
            num_steps=num_steps, 
            bf16_portion=bf16_portion,
            chunk_index=-1, 
            show_progress=show_progress,
            angle_schedule=angle_schedule, 
            schdeule_slope=schdeule_slope,
            classifier_free_guidance=classifier_free_guidance,
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

def init_sampler(checkpoint_path, local_rank, cache_dir, sequence_length=3750):
    device = torch.device(f"cuda:{local_rank}")
    sampler = ARVSampler(128, sequence_length, 1)
    sampler.set_device(device)
    return { "sampler": sampler }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diffusion_steps',
        type=int,
        default=25
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--bf16_portion',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--schedule_slope',
        type=float,
        default=2.5
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=2.5
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='lyrics2song_test'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    parser.add_argument(
        '--asset_path',
        type=str,
        default='recipes/diffusion/assets'
    )
    parser.add_argument(
        '--diffusion_model_path',
        type=str,
        default='/opt/tiger/arnold_experiment/samantha/logs/diffusion/mss/checkpoints/diffusion-step=019999.ckpt'
    )
    parser.add_argument(
        '--vocoder_model_path',
        type=str,
        default='soundstream-step=153999-val_sdr=12.7618.ckpt'
    )

    args = parser.parse_args()

    # params
    os.makedirs(args.output_dir_path, exist_ok=True)
    device = torch.device(args.device)
    # download assets
    asset_path = args.asset_path

    REMOTE_PATHS = {
        'diffusion_model_path': args.diffusion_model_path,
        'vocoder_model_path': args.vocoder_model_path
    }

    local_rank = str(device).split(':')[-1]

    # diffusion
    diffusion_model = init_diffusio_mss(REMOTE_PATHS['diffusion_model_path'], local_rank, cache_dir=asset_path)['diffusion']
    
    requires = {
        'diffusion': diffusion_model,
        **init_sampler(None, local_rank, cache_dir=asset_path, sequence_length=1176),
        **init_vocoder(REMOTE_PATHS['vocoder_model_path'], local_rank, cache_dir=asset_path, sample_rate=44100),
    }

    # inference
    test_audio = torchaudio.load("/opt/tiger/arnold_experiment/test7_orig/林俊杰 - 暂时的记号.mp3")[0]

    paded_audio = pad_audio(test_audio, segment_samples=44100*8)
    audio_frames = enframe(paded_audio, segment_samples=44100*8).to(device)

    diffusion_params = vars(args)
    diffusion_start = time()
    diffusion_model = requires['diffusion']
    sampler = requires['sampler']
    vocoder = requires['vocoder']
    sample_rate = 44100
    num_chunks = args.num_chunks
    diffusion_steps = args.diffusion_steps
    schedule_slope = args.schedule_slope
    guidance_scale = args.guidance_scale
    bf16_portion = args.bf16_portion
    start_time = time()
    with torch.no_grad():
        output_wavs = []
        for batch in torch.split(audio_frames, args.batch_size):
            encoder_out = vocoder.encode(batch)
            mix_emb, _, _ = vocoder.sample(encoder_out,  deterministic=False)
            
            pred_emb = sampler(
                model=diffusion_model,
                context=mix_emb,
                num_items=mix_emb.shape[0],
                num_chunks=num_chunks,
                num_steps=diffusion_steps,
                bf16_portion=bf16_portion,
                start=None,
                show_progress=True,
                angle_schedule='linear',
                schdeule_slope=schedule_slope,
                classifier_free_guidance=guidance_scale,
            ).detach()

            pred_emb = pred_emb.float()
            # torch.interpolate causes OOM for large batch sizes > 24. chunking to batch of 8 instead.
            # If you see this error, lower batch size: "RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false."
            output_wavs.append(vocoder.decode(pred_emb[:, :128]).detach()) 
        
        output_wavs = torch.cat(output_wavs, dim=0)
        output_wav = deframe(output_wavs)

        torchaudio.save(
            f'test.wav',
            output_wav.cpu(),
            sample_rate,
        )
    # print(f'Inference RTF: {(time() - start_time)/(len(prompts)*10)}')
