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
from recipes.bigmusic.lightning.semantic_modules import SemanticRLModule, process_eos_indexes, truncate_wav_to_eos
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
)
from recipes.diffusion.models.diffusion_model.utils import init_diffusion, run_diffusion
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.bigmusic.callbacks.save_outputs import save_batch_outputs, save_video
from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt
from samantha.utils.hparams import DotDict
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

            if not first:
                angles = math.pi /2. * sigma_i
                alphas, deltas = torch.cos(angles), torch.sin(angles)
                xt = alphas * init_emb + deltas * prev_noise
                current[:, :, :-625] = xt[:, :, :-625]

     
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
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = 'linear',
        schdeule_slope: float = 2.0,
        classifier_free_guidance = 1,
    ) -> Tensor:

        # Sample initial chunks
        b, c, duration = num_items, self.in_channels, self.duration

        # Sample initial chunks
        current_emb = torch.randn(b, c, duration*VOCODER_HZ, device=self.device)
        semantic_hop_size = 125
        diffusion_hop_size = 625
        tmp_emb = torch.zeros(b, c, duration*VOCODER_HZ + diffusion_hop_size *(num_chunks - 1), device=self.device)
        avg_cnt = torch.zeros(b, c, duration*VOCODER_HZ + diffusion_hop_size *(num_chunks - 1), device=self.device)
        prev_noise = current_emb
        output_emb = []
        for i in range(num_chunks):

            pred_emb = self.sample_loop(
                model=model,
                semantic_context=semantic_context[:, 0 + (i*semantic_hop_size):duration*SEMANTIC_HZ + (i*semantic_hop_size)],
                current=current_emb,
                prev_noise=prev_noise[..., 0 + (i*diffusion_hop_size):duration*VOCODER_HZ + (i*diffusion_hop_size)],
                num_steps=num_steps,
                bf16_portion=bf16_portion,
                angle_schedule=angle_schedule,
                classifier_free_guidance=classifier_free_guidance,
                first=(i==0),
            )

            tmp_emb[..., 0 + (i*diffusion_hop_size):duration*VOCODER_HZ + (i*diffusion_hop_size)] += pred_emb
            avg_cnt[..., 0 + (i*diffusion_hop_size):duration*VOCODER_HZ + (i*diffusion_hop_size)] += 1

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
    sequence_length=None,   # for backward compat
):
    device = torch.device(f"cuda:{local_rank}")
    sampler = ARVSampler(32, duration, 1)
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
        default=3,
    )
    parser.add_argument(
        '--samples_per_prompt',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--input_prompt_path', 
        type=str, 
        default="/mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv",
    )
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='offline_recons'
    )
    parser.add_argument(
        '--input_lang',
        type=str,
        default='en', # [en, zh_phone, zh_wp]
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
        default='/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=014000-median_rank_1=160-kaggle.ckpt'
    )
    parser.add_argument(
        '--lyrics_max_seq_len',
        type=int,
        default=400,
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--semantic_model_path',
        type=str,
        default='/mnt/bn/audio-diffusion/ducle/logs/semantic_model_mulan_rlhf/30s_4x2_billboard_sft_text+loudness+qualitativepair+genrechordv2-weight4_sim_v3/checkpoints/step=003500.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_0',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_1',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_2',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_3',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--vocoder_model_path',
        type=str,
        default='/mnt/bn/audio-diffusion/ducle/recipes/diffusion/assets/soundstream-step=374999-val_sdr=12.9557.ckpt'
    )

    args = parser.parse_args()

    # params
    os.makedirs(args.output_dir_path, exist_ok=True)
    device = torch.device(args.device)
    # download assets
    asset_path = args.asset_path

    REMOTE_PATHS = {
        'mulan_model_path': args.mulan_model_path,
        'semantic_model_path': args.semantic_model_path,
        'diffusion_model_path_2_0': args.diffusion_model_path_2_0,
        'diffusion_model_path_2_1': args.diffusion_model_path_2_1,
        'diffusion_model_path_2_2': args.diffusion_model_path_2_2,
        'diffusion_model_path_2_3': args.diffusion_model_path_2_3,
        'vocoder_model_path': args.vocoder_model_path
    }

    local_rank = str(device).split(':')[-1]

    # Mulan
    mulan_model_path = download_checkpoint(REMOTE_PATHS['mulan_model_path'], cache_dir=asset_path)
    mulan_model = create_mulan_model(mulan_model_path, device=device)

    # Semantic model
    semantic_model_path = download_checkpoint(REMOTE_PATHS['semantic_model_path'], cache_dir=asset_path)
    semantic_module = SemanticRLModule.load_from_checkpoint(semantic_model_path).to(device).eval()
    semantic_module.requires = { "mulan_infer_fn": mulan_inference, "mulan": mulan_model }

    # diffusion
    diffusion_model = {}
    diffusion_checkpoint_paths = {}
    diffusion_model_path = {
        '2_0': REMOTE_PATHS['diffusion_model_path_2_0'],
        '2_1': REMOTE_PATHS['diffusion_model_path_2_1'],
        '2_2': REMOTE_PATHS['diffusion_model_path_2_2'],
        '2_3': REMOTE_PATHS['diffusion_model_path_2_3'],
    }
    for key, remote_path in diffusion_model_path.items():
        ckpt_name = Path(remote_path).name
        if ckpt_name not in diffusion_checkpoint_paths:
            is_zh_token= True if args.input_lang=='zh_phone' else False
            diffusion_checkpoint_paths[ckpt_name] = init_diffusion(
                diffusion_model_path[key], local_rank, cache_dir=asset_path,
                is_zh_token=is_zh_token)['diffusion']
        diffusion_model[key] = diffusion_checkpoint_paths[ckpt_name]

    
    batch_size = args.batch_size
    duration = args.duration
    sample_rate = 24000
    requires = {
        'diffusion': diffusion_model,
        **init_sampler(None, local_rank, cache_dir=asset_path, duration=duration),
        **init_vocoder(REMOTE_PATHS['vocoder_model_path'], local_rank, cache_dir=asset_path)
    }
    lyrics_max_seq_len = semantic_module.extra_params.get("lyrics_max_seq_len", args.lyrics_max_seq_len)

    prompt_path = args.input_prompt_path
    if prompt_path == "":
        # [Example] Input prompt use case 
        index = ["s0001", "s0002"]
        category = ["none", "none"]
        style_text = ["大陆流行 开心 男声", "摇滚 平静 男声"]
        lyrics = [
            "阳光彩虹小白马 滴滴哒滴滴哒",            
            "没有什么能够阻挡 你对自由地向往"
            ]

        prompt_path = { 'index': index, 'category': category, 'style_text': style_text, 'lyrics': lyrics }
    inference_dataset = inference_dataset_from_prompt(
        prompt_path,
        conditions="style_text,lyrics_tokens",
        batch_size=batch_size,
        lyrics_max_seq_len=lyrics_max_seq_len,
        lang=args.input_lang,
        max_items=16)

    # inference
    start_time = time()
    diffusion_params = vars(args)  
    total_items = 0
    # get all files in a folder
    sample_paths = sorted(glob.glob(f'offline/*.npy'))
    import soundfile as sf
    with torch.no_grad():
        for batch_idx, batch in enumerate(sample_paths[:10]):
            print(f"generating {batch_idx}")
            semantic_samples = torch.from_numpy(np.load(sample_paths[batch_idx])).to(device)

            # Process the input lyrics and style prompt
            # hp = DotDict({ "duration": duration, "semantic_temperature": 1 , 'sample_mode': "gumble"})
            
            # semantic_samples = semantic_module.predict(batch, hp)
            # inputs_embeds = semantic_module.prepare_inputs_embeddings(
            #     batch={
            #         'conditions': "style_text,lyrics_tokens", 
            #         'lyrics_tokens': batch['lyrics_tokens'],
            #         'style_text': batch['style_text']})
            # inputs_embeds = repeat(inputs_embeds, 'b n d -> (b s) n d', s=args.samples_per_prompt)
            # semantic_samples = semantic_module.super_predict(inputs_embeds, 750, 1.0)
            # torch.save(semantic_samples, f'semantic_samples/{batch_idx}.pt')

            # semantic_samples, eos_index_list = process_eos_indexes(semantic_samples, semantic_module, sample_rate=sample_rate)
           
            diffusion_start = time()
            wavs_g = run_diffusion(requires, semantic_samples, params=diffusion_params)
            for idx, wav in enumerate(wavs_g):
                base_name = os.path.basename(sample_paths[batch_idx])
                sf.write(f'{args.output_dir_path}/{base_name}_{idx}.wav', wav.cpu().numpy().T, sample_rate)
            print('d ', time() - diffusion_start)
            # wavs = truncate_wav_to_eos(wavs_g, eos_index_list)

            # outputs = { "generated_audio": wavs }
            # output_dir = args.output_dir_path
            # save_batch_outputs(
            #     outputs, batch, output_dir=output_dir, sample_rate=sample_rate, 
            #     sample_round=args.samples_per_prompt, index_offset=total_items)
            # total_items += len(wavs)    
    save_video(output_dir, output_dir)

    print(f'Inference RTF: {(time() - start_time)/(len(prompts)*duration)}')
