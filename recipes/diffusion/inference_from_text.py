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
from recipes.musiclm.lightning.modules import SemanticModule
from recipes.diffusion.modules.pl_module import load_ema_checkpoint
from recipes.diffusion.models.tnt import TNTDiffusionNetwork
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
    mulan_rvq_indexs,
)
from recipes.soundstream.models.vqgan import VQGAN_KL_mix, VQGAN_KL_new
from recipes.soundstream.modules.pl_module_vae import VocoderModule
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
                key = '2-0'
            elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                key = '2-1'
            elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                key = '2-2'
            elif sigma_i[0,0,0] > 0.75:
                key = '2-3'

            # model prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                v_pred = model[key](
                    current, 
                    timesteps=sigma_i, 
                    mulan_context=positive_mulan_context, 
                    semantic_context=semantic_context,
                    mulan_force_cfg=0,
                    semantic_force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model[key](
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=positive_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=1,
                            semantic_force_cfg=1,
                        )
                    v_negative = v_uncond
                    if negative_mulan_context is not None:
                        v_negative = model[key](
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
        model, 
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
            model=model,
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
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
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diffusion_steps',
        type=int,
        default=25
    )
    parser.add_argument(
        '--bf16_portion',
        type=float,
        default=1.0
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
        default=16
    )
    parser.add_argument(
        '--input_text_path', 
        type=str, 
        default='../prompts.txt'
    )
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='google_outs'
    )
    parser.add_argument(
        '--input_source',
        type=str,
        default='semantic'
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
        '--mulan_model_path',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/v1.3_g4mix_127/checkpoints/mulan-step=036000-median_rank_0=127-kaggle.ckpt'
    )
    parser.add_argument(
        '--mulan_center_path',
        type=str,
        default='/home/byte_speech_sv/dongguo/mulan/codebook/kmeans_minibatch_codebook-mulan1b_g4_mix_127-1024x12.npy'
    )
    parser.add_argument(
        '--semantic_model_path',
        type=str,
        default='/mnt/bn/audio-diffusion/ducle/logs/semantic_flash_llama/mcc40m_conservative_filtered_vad2/checkpoints/step=091000-tr_loss=2.4243-val_loss_0=2.8712.ckpt'
    )
    parser.add_argument(
        '--semantic_center_path',
        type=str,
        default='/home/byte_speech_sv/zongyu.yin/ckpts/w2v/2.1/centroids_epoch_10.npy',
    )
    parser.add_argument(
        '--diffusion_model_path_2_0',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/diffusion/model_11_MCCVAD_ensemble_1.0/checkpoints/diffusion-step=153999.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_1',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/diffusion/model_11_MCCVAD_ensemble_1.0/checkpoints/diffusion-step=153999.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_2',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/diffusion/model_11_MCCVAD_ensemble_1.1/checkpoints/diffusion-step=049999.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_3',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/diffusion/model_11_MCCVAD_ensemble_2.3/checkpoints/diffusion-step=024999.ckpt'
    )
    parser.add_argument(
        '--vocoder_model_path',
        type=str,
        default='/home/byte_speech_sv/weitsung.lu/soundstream/dac_vae_125hz_24k/checkpoints/soundstream-step=592799-val_sdr=12.1354.ckpt'
    )

    args = parser.parse_args()

    # params
    os.makedirs(args.output_dir_path, exist_ok=True)
    device = torch.device(args.device)
    # download assets
    asset_path = args.asset_path
    os.makedirs(f'{asset_path}/diffusion_2.0', exist_ok=True)
    os.makedirs(f'{asset_path}/diffusion_2.1', exist_ok=True)
    os.makedirs(f'{asset_path}/diffusion_2.2', exist_ok=True)
    os.makedirs(f'{asset_path}/diffusion_2.3', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

    REMOTE_PATHS = {
        'mulan_model_path': args.mulan_model_path,
        'mulan_center_path': args.mulan_center_path,
        'semantic_model_path': args.semantic_model_path,
        'semantic_center_path': args.semantic_center_path,
        'diffusion_model_path_2_0': args.diffusion_model_path_2_0,
        'diffusion_model_path_2_1': args.diffusion_model_path_2_1,
        'diffusion_model_path_2_2': args.diffusion_model_path_2_2,
        'diffusion_model_path_2_3': args.diffusion_model_path_2_3,
        'vocoder_model_path': args.vocoder_model_path,
        'vocoder_model_path_2': '/home/byte_speech_sv/weitsung.lu/soundstream/yongye/1000k_ckpt.pyt'
    }

    LOCAL_PATHS = {
        'mulan_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["mulan_model_path"]).stem)}.ckpt',
        'mulan_center_path': f'{asset_path}/{str(Path(REMOTE_PATHS["mulan_center_path"]).stem)}.npy',
        'semantic_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["semantic_model_path"]).stem)}.ckpt',
        'semantic_center_path': f'{asset_path}/{str(Path(REMOTE_PATHS["semantic_center_path"]).stem)}.npy',
        'diffusion_model_path_2_0': f'{asset_path}/diffusion_2.0/{str(Path(REMOTE_PATHS["diffusion_model_path_2_0"]).stem)}.ckpt',
        'diffusion_model_path_2_1': f'{asset_path}/diffusion_2.1/{str(Path(REMOTE_PATHS["diffusion_model_path_2_1"]).stem)}.ckpt',
        'diffusion_model_path_2_2': f'{asset_path}/diffusion_2.2/{str(Path(REMOTE_PATHS["diffusion_model_path_2_2"]).stem)}.ckpt',
        'diffusion_model_path_2_3': f'{asset_path}/diffusion_2.3/{str(Path(REMOTE_PATHS["diffusion_model_path_2_3"]).stem)}.ckpt',
        'vocoder_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["vocoder_model_path"]).stem)}.ckpt',
        'vocoder_model_path_2': 'assets/1000k_ckpt.pyt' # different path due to soundstream code
    }

    for k, v in LOCAL_PATHS.items():
        if not os.path.exists(v):
            print(f'Downloading {REMOTE_PATHS[k]}')
            # get the folder path
            folder_path = '/'.join(v.split('/')[:-1])
            if k == 'vocoder_model_path_2':
                os.system(f'hdfs dfs -get {REMOTE_PATHS[k]} assets')
            else:
                if '/home/' in REMOTE_PATHS[k]:
                    os.system(f'hdfs dfs -get {REMOTE_PATHS[k]} {folder_path}')
                elif '/mnt/' in REMOTE_PATHS[k]:
                    os.system(f'cp {REMOTE_PATHS[k]} {folder_path}')

    # Mulan
    mulan_model = create_mulan_model(LOCAL_PATHS['mulan_model_path'], device=device)
    mulan_centers = np.load(LOCAL_PATHS['mulan_center_path'])
    mulan_centers = torch.from_numpy(mulan_centers).float().to(device)

     # Semantic model
    class ExtraParams:
        sample_rate = 24000
        duration = 10
        num_rounds = 3
        semantic_duration = 10
        coarse_duration = 10
        fine_duration = 4
        semantic_stride = 5
        coarse_stride = 5
        fine_stride = 3
        wav2vec_codebook_size = 1024
        mulan_codebook_size = 1024
        mulan_num_rvq = 12
        soundstream_codebook_size = 1024
        wav2vec_frame_rate = 25
        soundstream_frame_rate = 50
        num_coarse = 4
        num_fine = 8
        semantic_temperature = 1.0
        coarse_temperature = 0.9
        fine_temperature = 0.8
        sample_mode = "gumbel"


    semantic_module = SemanticModule.load_from_checkpoint(LOCAL_PATHS['semantic_model_path']).to(device).eval()
    semantic_centers = np.load(LOCAL_PATHS['semantic_center_path'])
    semantic_centers = torch.from_numpy(semantic_centers).float().to(device)

    # diffusion
    diffusion_model = {}
    diffusion_model_path = {
        '2-0': LOCAL_PATHS['diffusion_model_path_2_0'],
        '2-1': LOCAL_PATHS['diffusion_model_path_2_1'],
        '2-2': LOCAL_PATHS['diffusion_model_path_2_2'],
        '2-3': LOCAL_PATHS['diffusion_model_path_2_3'],
    }
    for key in diffusion_model_path:
        diffusion_model[key] = load_ema_checkpoint(
            diffusion_model_path[key],
            TNTDiffusionNetwork(
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
        )
        diffusion_model[key].eval()
        diffusion_model[key].to(device)
        diffusion_model[key]

    sampler = ARVSampler(32, 1250, 1)
    sampler.set_device(device)

    # vocoder model
    vocoder_model_pl = VocoderModule.load_from_checkpoint(
        LOCAL_PATHS['vocoder_model_path'],
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
    vocoder_model = vocoder_model_pl.generator.eval().to(device)

    # get the prompts
    with open(args.input_text_path, 'r') as f:
        prompt_list = [prompt.replace('\n', '') for prompt in f.readlines()][:32]
    # inference
    start_time = time()
    with torch.no_grad():
        for i in range(0, len(prompt_list), args.batch_size):
            print(f"generating {i} to {i + args.batch_size}")
            texts = prompt_list[i:(i + args.batch_size)]
            mulan_emb = mulan_inference(mulan_model, text=texts, device=device)
            postive_mulan_ids, ds = mulan_rvq_indexs(mulan_emb, mulan_centers)

            semantic_samples = semantic_module.predict(postive_mulan_ids, ExtraParams()) 

            diffusion_start = time()
            pred_emb = sampler(
                model=diffusion_model,
                positive_mulan_context=postive_mulan_ids,
                negative_mulan_context=None,
                semantic_context=semantic_samples,
                num_items=semantic_samples.shape[0],
                num_chunks=1,
                num_steps=args.diffusion_steps,
                bf16_portion=args.bf16_portion,
                start=None,
                show_progress=True,
                angle_schedule='linear',
                schdeule_slope=args.schedule_slope,
                classifier_free_guidance=args.guidance_scale,
            ).detach()
            print('d ', time() - diffusion_start)
            wavs_g = vocoder_model.decode(pred_emb.float()).detach()

            for wav_g, text in zip(wavs_g, texts):
                torchaudio.save(
                    f'{args.output_dir_path}/{text[:100]}.wav',
                    wav_g.cpu(),
                    24000,
                )

    print(f'Inference RTF: {(time() - start_time)/(len(prompt_list)*10)}')
