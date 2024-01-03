import librosa
import torch
from torchaudio.transforms import Resample

from scripts.data_processing.audio.ecapa_tdnn import ECAPA_TDNN_SMALL

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# Keep track of 10 different messages and then warn again
@lru_cache(10)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


def preprocess_audio(audio_bin, sample_rate, resampler, device, *_, **__):
    
    sample_rate = 16000
    wav, sr = librosa.load(audio_bin, sr=None)
    if wav.size == 0:
        return None, None
    audio_dur = wav.shape[0] / float(sr)
    wav = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    if sr != sample_rate:
        if sr not in resampler:
            resampler[sr] = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        warn_once(logger, f"resample audio from {sr=} to {sample_rate=}")
        wav = resampler[sr](wav)
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, *_, **__):
    for wav in batch:
        embed = model(wav).cpu().squeeze().numpy()
        yield embed


def load_model(device, model_path, *_, **__):
    checkpoint = f"{model_path}/wavlm_large_finetune.pth"
    model = init_model(checkpoint, model_path).to(device).eval()
    return model


def init_model(checkpoint, cache_dir):
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024, feat_type="wavlm_large", config_path=None, cache_dir=cache_dir
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    return model


# spk_emb
