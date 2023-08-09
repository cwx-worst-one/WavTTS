import os
import glob
import argparse
import math
from tqdm import tqdm
from time import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import torchaudio
from recipes.diffusion.modules.pl_module import DiffusionModule
from recipes.diffusion.models.tnt import TNTDiffusionNetwork
from recipes.soundstream.models.vqgan import VQGAN_KL_mix, VQGAN_KL_new
from recipes.soundstream.modules.pl_module_vae import VocoderModule
from recipes.musiclm.utils.dist import local_zero_first

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
    def __init__(self, in_channels: int, length: int, num_splits: int, diffusion_model_path):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits

        # diffusion
        diffusion_model = DiffusionModule.load_from_checkpoint(
            diffusion_model_path,
            diffusion_model=TNTDiffusionNetwork(
                input_dim=32,
                feature_dim=1024,
                context_dim=1,
                depth=16,
                segment_size=32,
                segment_stride=32,
                dropout=0,
                mulan_cfg_prob=0.10,
                semantic_cfg_prob=0.10,
                use_checkpoint=False
            ),
            target_dim=32,
            num_chunks=1,
            chunk_length=1250,
        )
        diffusion_model.eval()
        self.diffusion_model = diffusion_model

    
    def set_device(self, device: torch.device):
        self.device = device
        self.diffusion_model.to(device)
        self.diffusion_model.sampler.set_device(device)

    def sample_loop(
            self, 
            positive_mulan_context: torch.tensor, 
            negative_mulan_context: torch.tensor, 
            semantic_context: torch.tensor,
            current: Tensor, 
            num_steps: int = 20, 
            bf16_portion: float = 0.,
            chunk_index: int = -1, 
            show_progress: bool = False,
            angle_schedule: str = 'linear', 
            schdeule_slope: float = 2.0,
            classifier_free_guidance: int = 1,
            mulan_force_cfg: int = 0
    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        model = self.diffusion_model.model

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
                    mulan_context=positive_mulan_context, 
                    semantic_context=semantic_context,
                    mulan_force_cfg=mulan_force_cfg,
                    semantic_force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model(
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=positive_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=1,
                            semantic_force_cfg=1,
                        )
                    v_negative = v_uncond
                    if negative_mulan_context is not None:
                        v_negative = model(
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=negative_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=0,
                            semantic_force_cfg=1,
                        )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_negative)
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    def sample_start(self, 
        num_items: int, 
        num_steps: int, 
        bf16_portion: float,
        positive_mulan_context: torch.tensor, 
        negative_mulan_context: torch.tensor, 
        semantic_context: torch.tensor,
        **kwargs
    ) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Sample start
        return self.sample_loop(
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context,
            current=torch.randn(b, c, t, device=self.device),
            num_steps=num_steps, 
            bf16_portion=bf16_portion,
            chunk_index=-1, 
            **kwargs
        )

    @torch.no_grad()
    def forward(
        self,
        positive_mulan_context: torch.tensor,
        negative_mulan_context: torch.tensor,
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
        mulan_force_cfg: int = 1,
    ) -> Tensor:
        #assert_message = f"required at least {self.num_splits} chunks"
        #assert num_chunks >= self.num_splits, assert_message
        self.num_chunks = num_chunks

        # Sample initial chunks
        start = self.sample_start(
            num_items=num_items,
            num_steps=num_steps,
            bf16_portion=bf16_portion,
            angle_schedule=angle_schedule,
            classifier_free_guidance=classifier_free_guidance,
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context,
            mulan_force_cfg=mulan_force_cfg
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

def download_checkpoint(checkpoint_path, cache_dir):
    local_path = Path(f'{cache_dir}/{str(Path(checkpoint_path).stem)}.ckpt')
    if not os.path.exists(local_path):
        print(f'Downloading {checkpoint_path}')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # get the folder path
        folder_path = '/'.join(local_path.parts[:-1])
        if '/home/' in checkpoint_path:
            os.system(f'hdfs dfs -get {checkpoint_path} {folder_path}')
        elif '/mnt/' in checkpoint_path:
            os.system(f'cp {checkpoint_path} {folder_path}')
    return local_path

def init_diffusion(checkpoint_path, local_rank, cache_dir):
    with local_zero_first():
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        sampler = ARVSampler(32, 1250, 1, local_path)
        sampler.set_device(device)
        return { "diffusion": sampler }

def init_vocoder(checkpoint_path, local_rank, cache_dir):
    with local_zero_first():
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        # vocoder model
        vocoder_model_pl = VocoderModule.load_from_checkpoint(
            local_path,
            generator=VQGAN_KL_new(
                latent_dim=32,
                downsample_rates=[2, 3, 4, 8],
                upsample_rates=[8, 4, 3, 2],
                encoder_base_dim=96,
                decoder_base_dim=2560,
            ),
            discriminator=None,
            strict=False
            )
        generator = vocoder_model_pl.generator.eval().to(device)
        generator.decoder.eval().to(device)
        return { "vocoder": generator }

@torch.no_grad()
def run_diffusion(requires, positive_mulan_ids, samples, diffusion_steps=25, schedule_slope=2.5, guidance_scale=3, bf16_portion=1.0, mulan_force_cfg=0):
    sampler = requires['diffusion']
    vocoder = requires['vocoder']
    pred_emb = sampler(
        positive_mulan_context=positive_mulan_ids,
        negative_mulan_context=None,
        semantic_context=samples,
        num_items=samples.shape[0],
        num_chunks=1,
        num_steps=diffusion_steps,
        bf16_portion=bf16_portion,
        start=None,
        show_progress=True,
        angle_schedule='linear',
        schdeule_slope=schedule_slope,
        classifier_free_guidance=guidance_scale,
        mulan_force_cfg=mulan_force_cfg
    ).detach()
    pred_emb = pred_emb.float()
    # torch.interpolate causes OOM for large batch sizes > 24. chunking to batch of 16 instead.
    wavs_g = torch.cat([vocoder.decode(c).detach() for c in torch.split(pred_emb, 16)])
    return wavs_g



# if __name__ == '__main__':
#     diffusion_steps = 25
#     schedule_slope = 2.5
#     guidance_scale = 2.5
#     bf16_portion = 0

#     device = 'cuda:0'
#     device = torch.device(device)
#     asset_path = '/mnt/bn/ashaw-us/repos/samantha/recipes/diffusion/assets'

#     diffusion_ckpt_path = '/home/byte_speech_sv/weitsung.lu/diffusion/diffusion-step=346999.ckpt'
#     vocoder_ckpt_path = '/home/byte_speech_sv/weitsung.lu/soundstream/dac_vae_125hz_24k/checkpoints/soundstream-step=592799-val_sdr=12.1354.ckpt'

#     sampler = init_diffusion(diffusion_ckpt_path, asset_path)
#     vocoder = init_vocoder(vocoder_ckpt_path, asset_path)
#     requires = { **sampler, **vocoder}
#     requires['diffusion'].to(device)
#     requires['vocoder'].to(device)

#     from recipes.musiclm.datasets.mcc import MCC40MDataset
#     import IPython.display as ipd
#     dataset = MCC40MDataset(
#         url2index="/mnt/bn/audio-diffusion/data/genre_balanced_mcc/vocal_mcc_hdfs_paths_1506.tar_to_index.tsv",
#         sample_rate=24000,
#         duration=10.0,
#         max_num_crops=1,
#         aed_filtered=False,
#         use_pipe=True
#     )
#     it = iter(dataset)


#     item = next(it)

#     x = item['audio']

#     mulan_ids = get_mulan_tokens(requires, x.cuda())

#     wav2vec_ids = get_wav2vec_tokens(requires, x.cuda())

#     # print(f'Inference RTF: {(time() - start_time)/(len(intput_path_list)*10)}')