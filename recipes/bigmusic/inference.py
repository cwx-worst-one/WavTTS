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
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
    mulan_rvq_indexs,
)
from recipes.musiclm.requires.model_initializer import init_soundstream_decoder
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, AddConditionsTransform, AddMulanVocalTagTransform
from recipes.bigmusic.datasets.lyrics import transform_dataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from torch.utils.data import DataLoader
from recipes.bigmusic.lightning.semantic_modules import MixSemanticModule, SemanticModule, SemanticT5Module
from recipes.bigmusic.lightning.acoustic_modules import CoarseModule
from recipes.musiclm.lightning.modules import FineModule
from recipes.musiclm.inference.utils import slugify, save_wav
from collections import namedtuple

SAMPLE_RATE = 24000

def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")    
    return text.translate(str.maketrans("", "", nlp_punctuation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        '--output_dir_path', 
        type=str, 
        default='google_outs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    parser.add_argument(
        '--asset_path',
        type=str,
        default='/mnt/bn/audio-diffusion/qq/asset'
    )

    # Required module
    parser.add_argument(
        '--semantic_model_path',
        type=str,
        default='/mnt/bn/audio-diffusion/qq/logs/semantic_flash_llama/chroma_vocal/semantic_model_mix/varlen_20_30_bs24_03B/checkpoints/step=147000-val_accu_0=19.17.ckpt'
    )
    parser.add_argument(
        '--coarse_model_path',
        type=str,
        default='/mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse_unified/coarse_flash_llama_wav2vec/retrained_params/checkpoints/step=065000-val_accu_0=15.83.ckpt'
    )
    parser.add_argument(
        '--fine_model_path',
        type=str,
        default='/mnt/bn/lyrics-to-song/ashaw/logs/l2s/fine/fine_flash_llama_mcc_vocals/vocal_finetune_60m/checkpoints/last.ckpt',
    )
    parser.add_argument(
        '--soundstream_model_path',
        type=str,
        default='hdfs://harunava/home/byte_speech_sv/zongyu.yin/ckpts/soundstream/190k',
    )    
    
    args = parser.parse_args()

    # params
    os.makedirs(args.output_dir_path, exist_ok=True)    
    device = torch.device(args.device)
    # download assets
    asset_path = args.asset_path
    os.makedirs(asset_path, exist_ok=True)

    REMOTE_PATHS = {
        'semantic_model_path': args.semantic_model_path,
        'coarse_model_path': args.coarse_model_path,
        'fine_model_path': args.fine_model_path,
        'soundstream_model_path': args.soundstream_model_path,
    }

    LOCAL_PATHS = {
        'semantic_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["semantic_model_path"]).stem)}.ckpt',
        'coarse_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["coarse_model_path"]).stem)}.ckpt',
        'fine_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["fine_model_path"]).stem)}.ckpt',
        'soundstream_model_path': f'{asset_path}/{str(Path(REMOTE_PATHS["soundstream_model_path"]).stem)}',
    }

    for k, v in LOCAL_PATHS.items():
        if not os.path.exists(v):
            print(f'Downloading {REMOTE_PATHS[k]}')
            # get the folder path
            folder_path = '/'.join(v.split('/')[:-1])
            if '/home/' in REMOTE_PATHS[k]:
                print("execute", f'hdfs dfs -get {REMOTE_PATHS[k]} {folder_path}')
                os.system(f'hdfs dfs -get {REMOTE_PATHS[k]} {folder_path}')
            elif '/mnt/' in REMOTE_PATHS[k]:
                os.system(f'cp {REMOTE_PATHS[k]} {folder_path}')
                print("execute", f'cp {REMOTE_PATHS[k]} {folder_path}')

    # Initialize modules    
    semantic_module = MixSemanticModule.load_from_checkpoint(LOCAL_PATHS['semantic_model_path']).to(device).eval()
    semantic_module.load_required_modules()
    coarse_module = CoarseModule.load_from_checkpoint(LOCAL_PATHS['coarse_model_path']).to(device).eval()
    coarse_module.load_required_modules()
    fine_module = FineModule.load_from_checkpoint(LOCAL_PATHS['fine_model_path']).to(device).eval()    
    soundstrem_module = init_soundstream_decoder(
        hpath=args.soundstream_model_path,
        local_rank=0,
        cache_dir='/mnt/bn/audio-diffusion/qq/asset/190k'
    )['ss_dec']    
    
    # Get the prompts
    genre = "pop"
    mood = "happy"
    lyrics = "Five little monkeys jumping on the bed!"

    conditions = "style_text,lyrics_tokens"
    lyrics_norm = normalize_text(lyrics)
    items = [{"style_text": " ".join([mood, genre]),
              "lyrics": lyrics_norm
            }]
    segment_transforms = [LyricsTokenTransform.init_espeak_tokenizer(
        lyrics_max_seq_len=250), AddMulanVocalTagTransform()]
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    dataset = transform_dataset(dataset, segment_transforms=segment_transforms, batch_transforms=batch_transforms, batch_size=1)
    
    # inference
    start_time = time()
    hp = {
        "semantic_temperature": 1.0,
        "coarse_temperature": 0.9,
        "fine_temperature": 0.8,
        "sample_mode": "gumbel",
        "sample_rate": 24000,
        "duration": 30,
        "coarse_duration": 10,
        "fine_duration": 4,
        "coarse_stride": 5,
        "fine_stride": 3,
        "semantic_frame_rate": 25,
        "soundstream_codebook_size": 1024,
        "soundstream_frame_rate": 50,
        "num_coarse": 4,
        "num_fine": 8,             
    }
    HP = namedtuple('HP', hp)
    extra_params = HP(**hp)

    with torch.no_grad():
        batch = next(iter(dataset))        
        batch['lyrics_tokens'] = batch['lyrics_tokens'].to(device)        
        semantic_samples = semantic_module.predict(batch, extra_params)
        eos_index = torch.cumsum(semantic_samples == self.semantic_module.target_embedder.eos_id, 1) > 0
        semantic_samples[eos_index] = 0        
        
        coarse_samples = coarse_module.predict(semantic_samples, extra_params)
        fine_samples = fine_module.predict(coarse_samples, extra_params)
        
        bs = coarse_samples.size(0)
        coarse_samples = coarse_samples.view([bs, -1, extra_params.num_coarse])
        fine_samples = fine_samples.view([bs, -1, extra_params.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(extra_params.num_coarse + extra_params.num_fine, device=coarse_samples.device)
            * extra_params.soundstream_codebook_size
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = soundstrem_module(vqgan_inputs).squeeze(1)
        
        for i, (wav, eos) in enumerate(wavs, eos_index):
            # wav = wav[: eos_index/25]
            file_name = ""
            if 'lyrics_tokens' in conditions:
                file_name += slugify(batch[i]['lyrics'])[:128]
            if 'style_text' in conditions:
                file_name += '--' + slugify(batch[i]['style_text'])[:128]
            if file_name: 
                file_name += f'.{round}-{i}'
            else:
                file_name += f'{round}-{i}'        
            wav_fp = os.path.join(args.output_dir_path, f"{file_name}.wav")            
            save_wav(wav.cpu().float(), wav_fp, sr=SAMPLE_RATE)

    print(f'Inference RTF: {(time() - start_time)}')
