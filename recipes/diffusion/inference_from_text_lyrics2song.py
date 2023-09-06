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
# from recipes.bigmusic.lightning.semantic_modules import SemanticModule
# TODO: (AS) switch this to unified model once training is complete
from recipes.bigmusic.dev.v0.lightning.semantic_modules_v0 import SemanticModule
from recipes.bigmusic.dev.qq.lightning.semantic_modules_qq import MixSemanticModule
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
)
from recipes.diffusion.models.diffusion_model.utils import init_diffusion
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform
torch.backends.cuda.matmul.allow_tf32 = True

def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")    
    return text.translate(str.maketrans("", "", nlp_punctuation))

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
        et = torch.randn_like(init_emb)

        sigma = 1.
        for i in progress_bar:
            
            sigma_i = torch.ones(B, 1, T, device=self.device) * sigma

            if not first:
                angles = math.pi /2. * sigma_i
                alphas, deltas = torch.cos(angles), torch.sin(angles)
                xt = alphas * init_emb + deltas * et
                current[:, :, :625] = xt[:, :, :625]

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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
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
        b, c, t = num_items, self.in_channels, self.length
        pred_embs = []

        # Sample initial chunks
        current_emb = torch.randn(b, c, t, device=self.device)
        semantic_hop_size = 125
        overlap_size= 625
        for i in range(num_chunks):

            pred_emb = self.sample_loop(
                model=model,
                semantic_context=semantic_context[:, 0 + (i*semantic_hop_size):250 + (i*semantic_hop_size)],
                current=current_emb,
                num_steps=num_steps,
                bf16_portion=bf16_portion,
                angle_schedule=angle_schedule,
                classifier_free_guidance=classifier_free_guidance,
                first=(i==0),
            )
            pred_embs.append(pred_emb)
            current_emb = torch.cat([pred_emb[:, :, overlap_size:], torch.randn(b, c, overlap_size, device=self.device)], dim=-1)

        tmp_emb = torch.zeros(b, c, 1250 + overlap_size *(num_chunks - 1), device=self.device)

        for i, emb in enumerate(pred_embs):
            tmp_emb[..., 0 + (i*overlap_size):1250 + (i*overlap_size)] += emb

        tmp_emb[..., overlap_size:-overlap_size] /= 2
        pred_emb = tmp_emb
        return pred_emb

def init_sampler(checkpoint_path, local_rank, cache_dir):
    device = torch.device(f"cuda:{local_rank}")
    sampler = ARVSampler(32, 1250, 1)
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
        default=1
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
        default=5,
    )
    parser.add_argument(
        '--input_text_path', 
        type=str, 
        default='../prompts.txt'
    )
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='lyrics2song_test'
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
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/lyrics2song_30s_ckpts/mulan-step=014000-median_rank_1=160-kaggle.ckpt'
    )
    parser.add_argument(
        '--semantic_model_path',
        type=str,
        default='/mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/semantic_model_mulan/semantic_chroma_text_from_scratch_artist_gender_l24_h16_b8_g32_2em4/checkpoints/step=053000-val_accu_0=7.79.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_0',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_part2/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_1',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_part2/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_2',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_part2/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--diffusion_model_path_2_3',
        type=str,
        default='/mnt/bn/audio-diffusion/wtl/diffusion/model_14_part2/checkpoints/last.ckpt'
    )
    parser.add_argument(
        '--vocoder_model_path',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/weitsung.lu/soundstream/dac_vae_125hz_24k_v2_part2/checkpoints/soundstream-step=374999-val_sdr=12.9557.ckpt'
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

    # 10s
    if args.num_chunks == 1:
        semantic_module_cls = SemanticModule
    # 30s
    else:
        semantic_module_cls = MixSemanticModule
    
    # Semantic model
    semantic_model_path = download_checkpoint(REMOTE_PATHS['semantic_model_path'], cache_dir=asset_path)
    semantic_module = semantic_module_cls.load_from_checkpoint(semantic_model_path).to(device).eval()
    semantic_module.requires = { "mulan_infer_fn": mulan_inference, "mulan": mulan_model }

    # lyrics tokenizer
    lyrics_max_seq_len = semantic_module.extra_params.lyrics_max_seq_len
    lyrics_tokenizer = LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len=lyrics_max_seq_len)

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
            diffusion_checkpoint_paths[ckpt_name] = init_diffusion(diffusion_model_path[key], local_rank, cache_dir=asset_path)['diffusion']
        diffusion_model[key] = diffusion_checkpoint_paths[ckpt_name]

    
    sampler = init_sampler(None, local_rank, cache_dir=asset_path)['sampler']
    vocoder_model = init_vocoder(REMOTE_PATHS['vocoder_model_path'], local_rank, cache_dir=asset_path)['vocoder']

    # input text. 
    # TODO: move to a separate file
    mood_prompts = ['', '', '']
    genre_prompts = ['dream pop', 'hiphop jazz', 'acoustic country']
    
    prompts = [" ".join([m, g]) for m, g in zip(mood_prompts, genre_prompts)]
    
    # lyrics = [
    #     "won't you talk to me texas, let me hear them drawl, i spent my last five dollars on this one long distance call won't you talk to me texas i got these homesick blues tell me i can come on home to you",
    #     "it may be factual it may be cool ungain love everybody plays the fool how can you help it when the music starts to play and your ability to reason is swept away oh heaven",
    #     " i see the crystal raindrops fall and the beauty of it all is when the sun comes shining through to make those rainbows in my mind when i think of you sometime and i wanna spend some time with you"
    #     ] 
    lyrics = [
        "Hey Jude, don't make it bad. Take a sad song and make it better."
    ] * 3
    lyrics = [{'lyrics': l} for l in lyrics]
    lyrics_tokens = [lyrics_tokenizer(lyric)['lyrics_tokens'] for lyric in lyrics]
    lyrics_tokens = torch.stack(lyrics_tokens).to(device)

    # inference
    start_time = time()
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            print(f"generating {i} to {i + args.batch_size}")

            # Process the input lyrics and style prompt
            _lyrics_tokens = lyrics_tokens[i:(i + args.batch_size)]
            _prompts = prompts[i:(i + args.batch_size)]

            inputs_embeds = semantic_module.prepare_inputs_embeddings(batch={'conditions': "style_text,lyrics_tokens", 'lyrics_tokens': _lyrics_tokens, 'style_text': _prompts})
            semantic_samples = semantic_module.super_predict(inputs_embeds, 250 + 125*(args.num_chunks - 1), 1.0)
            if args.num_chunks > 1:
                eos_id = semantic_module.target_embedder.eos_id
                if eos_id is not None:
                    eos_index = torch.cumsum(semantic_samples == eos_id, 1) > 0
                    semantic_samples[eos_index] = 0   

            diffusion_start = time()
            pred_emb = sampler(
                model=diffusion_model,
                semantic_context=semantic_samples,
                num_items=semantic_samples.shape[0],
                num_chunks=args.num_chunks,
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

            if args.num_chunks > 1:
                for wav_g, text, eos in zip(wavs_g, _prompts, eos_index):
                    # find first True in eos
                    try:
                        eos = torch.nonzero(eos)[0][0]
                        wav_g[..., eos*960:] = 0
                    except:
                        pass

                    torchaudio.save(
                        f'{args.output_dir_path}/{text[:100]}.wav',
                        wav_g.cpu(),
                        24000,
                    )
            else:
                for wav_g, text in zip(wavs_g, _prompts):

                    torchaudio.save(
                        f'{args.output_dir_path}/{text[:100]}.wav',
                        wav_g.cpu(),
                        24000,
                    )

    print(f'Inference RTF: {(time() - start_time)/(len(prompts)*10)}')
