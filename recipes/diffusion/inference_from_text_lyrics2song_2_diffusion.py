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
from recipes.bigmusic.lightning.semantic_modules import SemanticModule, process_eos_indexes
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
)
from recipes.diffusion.modules.pl_module_semantic import DiffusionModule
from recipes.diffusion.models.tnt_v2 import TNTDiffusionNetwork
from recipes.diffusion.models.diffusion_sampler import Sampler
from recipes.diffusion.models.diffusion_model.utils import init_diffusion, run_diffusion
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.bigmusic.callbacks.save_outputs import save_batch_outputs, save_video
from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt
from samantha.utils.hparams import DotDict
torch.backends.cuda.matmul.allow_tf32 = True
VOCODER_HZ = 125
SEMANTIC_HZ = 25

def init_sampler(checkpoint_path, local_rank, cache_dir, duration=30, vocoder_hz=125, semantic_hz=25):
    device = torch.device(f"cuda:{local_rank}")
    sampler = Sampler(32, duration, 1, vocoder_hz, semantic_hz)
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
        default=2,
    )
    parser.add_argument(
        '--samples_per_prompt',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--input_prompt_path', 
        type=str, 
        default='/mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv'
    )
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='lyrics2song_2_diffusion_output'
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
        default='/mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/last.ckpt'
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

    # get umm vq
    from recipes.umm.requires.model_initializer import init_stage3
    emb_table = init_stage3(
        'hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt',
        local_rank,
        cache_dir=asset_path
    )['Stage3'].model.vq.embedding.weight

    # Semantic model
    semantic_diffusion_path = '/mnt/bn/audio-diffusion/wtl/semantic/model_0/checkpoints/last-EMA.ckpt'
    semantic_diffusion_module = DiffusionModule.load_from_checkpoint(
        semantic_diffusion_path,
        diffusion_model=TNTDiffusionNetwork(
                input_dim=32,
                feature_dim=1024,
                context_dim=1024,
                depth=16,
                segment_size=4,
                segment_stride=4,
                dropout=0,
                semantic_cfg_prob=0.10,
                use_checkpoint=False
        )
        )
    semantic_diffusion_module.requires = { "mulan_infer_fn": mulan_inference, "mulan": mulan_model }
    semantic_diffusion_module.to(device)
    semantic_sampler = init_sampler(None, local_rank, cache_dir=asset_path, duration=30, vocoder_hz=25, semantic_hz=100)
    
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

    batch_size = args.batch_size
    duration = args.duration
    sample_rate = 24000
    requires = {
        'diffusion': diffusion_model,
        **init_sampler(None, local_rank, cache_dir=asset_path, duration=duration),
        **init_vocoder(REMOTE_PATHS['vocoder_model_path'], local_rank, cache_dir=asset_path)
    }
    lyrics_max_seq_len = args.lyrics_max_seq_len

    prompt_path = args.input_prompt_path
    if prompt_path is None:
        # [Example] Input prompt use case 
        style_prompt_metadata = [
            {
                "final_mood": 'Chill',
                "final_genre": 'Dream Pop',
                'merge_aed': 'Female'
            },
            {
                "final_mood": 'relax',
                "final_genre": 'HipPop',
                'merge_aed': 'Male'
            },
            {
                "final_mood": '',
                "final_genre": 'Acoustic country',
                'merge_aed': 'Male'
            },
        ]
        lyrics = [
            "won't you talk to me texas, let me hear them drawl, i spent my last five dollars on this one long distance call won't you talk to me texas i got these homesick blues tell me i can come on home to you",
            "it may be factual it may be cool ungain love everybody plays the fool how can you help it when the music starts to play and your ability to reason is swept away oh heaven",
            " i see the crystal raindrops fall and the beauty of it all is when the sun comes shining through to make those rainbows in my mind when i think of you sometime and i wanna spend some time with you"
            ]
        prompts = { 'metadata': style_prompt_metadata, 'lyrics': lyrics }
        inference_dataset = inference_dataset_from_prompt(prompts, conditions="style_text,lyrics_tokens", batch_size=batch_size, lyrics_max_seq_len=lyrics_max_seq_len)
    else:
        inference_dataset = inference_dataset_from_prompt(prompt_path, conditions="style_text,lyrics_tokens", batch_size=batch_size, lyrics_max_seq_len=lyrics_max_seq_len)

    # inference
    start_time = time()
    diffusion_params = vars(args)
    total_items = 0
    with torch.no_grad():
         for batch_idx, batch in enumerate(inference_dataset):
            print(f"generating {batch_idx}")

            inputs_embeds = semantic_diffusion_module.prepare_inputs_embeddings(batch={'conditions': "style_text,lyrics_tokens", 'lyrics_tokens': batch['lyrics_tokens'].to(device), 'style_text': batch['style_text']})
            diffusion_start = time()
            semantic_samples= semantic_sampler["sampler"](
                model=semantic_diffusion_module.model,
                semantic_context=inputs_embeds,
                num_items=inputs_embeds.shape[0],
                num_chunks=args.num_chunks,
                num_steps=25,
                bf16_portion=args.bf16_portion,
                angle_schedule='linear',
                schdeule_slope=args.schedule_slope,
                classifier_free_guidance=8,
            ).detach()
            print('s ', time() - diffusion_start)

            # get the closet index in emb_table through cosine similarity
            _cache = []
            for i, _semantic_samples in enumerate(semantic_samples):
                sim = torch.matmul(torch.nn.functional.normalize(emb_table, dim=-1), torch.nn.functional.normalize(_semantic_samples, dim=0)) # 32768 32, 32 750
                am = torch.argmax(sim.T, dim=1)
                _cache.append(am)
            semantic_samples = torch.stack(_cache)
            # semantic_samples = semantic_module.super_predict(inputs_embeds, 750 + 125*(args.num_chunks - 1), 1.0)

            # semantic_samples, eos_index_list = process_eos_indexes(semantic_samples, semantic_module, sample_rate=sample_rate)
            diffusion_start = time()
            wavs_g = run_diffusion(requires, semantic_samples, params=diffusion_params)
            print('d ', time() - diffusion_start)

            for wav_g, ly, text in zip(wavs_g, batch['lyrics_normalized_text'], batch['style_text']):
                if wav_g.dim() == 1:
                    wav_g = wav_g.unsqueeze(0)
                # if eos is not None:
                #     wav_g = wav_g[:, :eos]

                torchaudio.save(
                    f'{args.output_dir_path}/{text[:50]}_{ly[:200]}.wav',
                    wav_g.cpu(),
                    sample_rate,
                )
