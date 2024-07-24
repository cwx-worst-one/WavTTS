import os
from typing import Optional

import torch

from recipes.research.audio_codec.scripts.compile import get_latest_model_from_commit
from recipes.research.audio_codec.zoo import AudioCodec_f81b3fa_64l, AudioCodec_7c355ea_64l
from recipes.research.diff import DiffInstrumental
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


def load_diffusion_model(
    commit_hash: str,
    device: torch.device,
    cache: bool = False,
    ckpt_path: Optional[str] = None,
):
    if ckpt_path is None:
        ckpt_path = get_latest_model_from_commit("diff/default", commit_hash)

    model = DiffInstrumental.load_from_checkpoint(
        ckpt_path, cache=cache, map_location=device
    )
    model = model.eval()
    model.freeze()
    model.setup(device=device)
    # model = model.load_audio_codec(AudioCodec_f81b3fa_64l())
    model = model.load_audio_codec(AudioCodec_7c355ea_64l(), device)
    model.commit_hash = commit_hash
    model.commit_step = os.path.basename(ckpt_path)
    return model


def sample(
    model: DiffInstrumental,
    text_prompt: str,
    seconds_start: float,
    seconds_total: float,
    t: int,
    cfg_weight: float,
    schedule_tau: float,
    batch_size: int = 1,
    negative_prompt: Optional[str] = None,
    init_noise: Optional[torch.Tensor] = None,
    solver: str = "dpmpp-3m-sde",
    duration: float = 60,
    device: torch.device = "cuda",
):

    logger.info(
        f"Solver: {solver}\nInference parameters:\nPrompt: {text_prompt}\nSeconds start: {seconds_start}\nSeconds total: {seconds_total}\nNegative prompt: {negative_prompt}\nTimesteps: {t}\nCFG weight: {cfg_weight}\nTau: {schedule_tau}\nDuration: {duration}"
    )
    batch_text = [text_prompt] * batch_size
    batch_seconds_start = [seconds_start] * batch_size
    batch_seconds_total = [seconds_total] * batch_size
    with torch.no_grad():
        if negative_prompt is None:
            batch_negative_prompt = None
        else:
            batch_negative_prompt = [negative_prompt] * batch_size

        print("Denoising text -> z_audio...")
        pred_z_audio = model.sample_from_text(
            batch_text,
            batch_seconds_start,
            batch_seconds_total,
            t=t,
            cfg_weight=cfg_weight,
            schedule_tau=schedule_tau,
            negative_text=batch_negative_prompt,
            init_noise=init_noise,
            solver=solver,
            duration=duration,
            device=device,
        )
        pred_audio = model.decode_audio(pred_z_audio)

    return pred_audio, model.config.sample_rate
