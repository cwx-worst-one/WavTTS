import sys
from time import perf_counter
from typing import List, Optional

import torch
import torchaudio
import webdataset as wds

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from recipes.soundstorm.lightning.soundstorm import SoundStorm, SoundStormInference


@torch.no_grad()
def generate(
    soundstorm: SoundStorm,
    semantic_tokens: torch.Tensor,
    max_seq_len: int,
    iterations: List[int],
    score_strategies: List[str],
    sampled_t: Optional[int] = None,
    temperature: float = 1.0,
):
    batch_size = semantic_tokens.shape[0]
    pred_tokens = torch.empty(
        batch_size,
        soundstorm.n_quantizers,
        0,
        device=soundstorm.device,
        dtype=torch.long,
    )
    ratio = soundstorm.audio_model.frame_rate // soundstorm.semantic_model.frame_rate
    samples_to_generate = semantic_tokens.shape[1] * ratio
    semantic_window_size = max_seq_len // ratio
    prefix_size = max_seq_len // 2
    while pred_tokens.shape[2] < samples_to_generate:
        if pred_tokens.shape[2] == 0:
            st = 0
            st_semantic = 0
            en_semantic = st_semantic + semantic_window_size
            this_prefix = None
            this_prefix_size = 0
        elif samples_to_generate - pred_tokens.shape[2] >= max_seq_len:
            st = pred_tokens.shape[2] - prefix_size
            st_semantic = st // ratio
            en_semantic = st_semantic + semantic_window_size
            this_prefix = pred_tokens[..., st : st + prefix_size]
            this_prefix_size = prefix_size
        else:
            this_prefix_size = max_seq_len - (
                samples_to_generate - pred_tokens.shape[2]
            )
            st = pred_tokens.shape[2] - this_prefix_size
            st_semantic = st // ratio
            en_semantic = st_semantic + semantic_window_size
            this_prefix = pred_tokens[..., st : st + this_prefix_size]
        this_pred_tokens, _ = soundstorm.iterative_decoding(
            semantic_tokens[..., st_semantic:en_semantic],
            max_seq_len=max_seq_len,
            iterations=iterations,
            score_strategies=score_strategies,
            temperatures=temperature,
            sampled_t=sampled_t,
            prefix_tokens=this_prefix,
        )
        pred_tokens = torch.cat(
            (pred_tokens, this_pred_tokens[..., this_prefix_size:]), dim=2
        )
    return soundstorm.audio_model.decode(pred_tokens)


def load_model(ckpt_path: str, device: str, **kwargs) -> SoundStorm:
    return SoundStorm.load_from_checkpoint(ckpt_path, **kwargs).to(device)


def load_model2(
    soundstorm_path: str, semantic_ckpt_path: str, device: str, **kwargs
) -> SoundStormInference:
    print(f"Loading from {soundstorm_path} / {semantic_ckpt_path}...")
    return SoundStormInference(soundstorm_path, semantic_ckpt_path, **kwargs).to(device)
