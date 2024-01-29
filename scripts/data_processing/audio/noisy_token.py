import math
import os
import random
from tarfile import ExtractError

import librosa
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from torchaudio.functional import add_noise
from torchaudio.transforms import Resample

from samantha.utils.distributed import rank_zero_first
from samantha.utils.hdfs_helper import _run_command
from scripts.data_processing.audio.umm_tokenizer import process_batch as _umm_encode
from scripts.data_processing.audio.wvae_mel_utils import process_batch as _wvae_encode

snr_range = [0, 20]


def preprocess_audio(audio_bin, sample_rate, resampler, device, noise_lst, *_, **__):
    def _resample(_wav, _sr):
        if _sr != sample_rate:
            if _sr not in resampler:
                resampler[_sr] = Resample(orig_freq=_sr, new_freq=sample_rate).to(
                    device
                )
            return resampler[_sr](_wav)
        return wav

    wav, sr = librosa.load(audio_bin, sr=None)
    if wav.size == 0:
        return None, None
    audio_dur = wav.shape[-1] / float(sr)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]

    wav = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    wav = _resample(wav, sr)
    audio_len = wav.shape[-1]

    snr = random.uniform(*snr_range)
    noisy_wav = None

    while True:
        noise_addr = random.choice(noise_lst)
        noise, noise_sr = librosa.load(noise_addr, sr=None)
        if len(noise.shape) == 2 and noise.shape[-1] == 2:
            noise = noise[:, 0]
        noise = torch.as_tensor(noise, dtype=torch.float32, device=device).unsqueeze(0)
        noise = _resample(noise, noise_sr)
        noise_len = noise.shape[-1]

        if noise_len >= audio_len:
            shift = random.randint(0, noise_len - audio_len)
            noise_chunk = noise[:, shift : shift + audio_len]
        else:
            noise_chunk = torch.zeros_like(wav)
            shift = random.randint(0, audio_len - noise_len)
            noise_chunk[:, shift : shift + noise_len] = noise

        noisy_wav = add_noise(wav, noise_chunk, torch.as_tensor([snr], device=device))

        if not torch.isnan(noisy_wav).any():
            break

    scale = max(0.001, noisy_wav.abs().max())
    noisy_wav = (noisy_wav / scale * 0.95).view(1, -1)
    return (noisy_wav, snr, noise_addr, shift), audio_dur


def padding(wav):
    sample_rate = 24000
    umm_freq, wvae_freq = 25, 40
    pad_mod = math.lcm(sample_rate // umm_freq, sample_rate // wvae_freq)
    wav_len = wav.shape[-1]
    pad_len = wav_len % pad_mod
    if pad_len != 0:
        wav = F.pad(wav, (0, pad_mod - pad_len), "constant", 0)
    return wav


@torch.no_grad()
def process_batch(model, batch, device, sample_rate, *_, **__):
    if not batch:
        yield from batch

    batch_wav = [padding(b[0]) for b in batch]
    batch_wvae = _wvae_encode(
        model["wvae"], [wav.unsqueeze(0) for wav in batch_wav], device, sample_rate
    )
    batch_umm = _umm_encode(model["umm"], batch_wav, device)

    for (wav, snr, noise_addr, noise_shift), wvae, umm in zip(
        batch, batch_wvae, batch_umm
    ):
        yield (wav.squeeze().cpu().numpy() * 32767).astype(np.int16), wvae, umm, {
            "umm_version": "0.6.2",
            "wvae_version": "3.1",
            "noise_addr": noise_addr,
            "noise_shift": noise_shift,
            "snr": snr,
            "snr_range": snr_range,
        }


def load_model(device, model_path, *_, **__):
    rank = int(device.split(":")[-1])
    umm_ckpt_path = f"{model_path}/umm/umm_tokenizer_%d.pt"
    wvae_encoder_path = f"{model_path}/wvae/wavevae_encoder_%d.pt"
    umm = torch.jit.load(umm_ckpt_path % rank).eval().to(device)
    wvae_encoder = torch.jit.load(wvae_encoder_path % rank).eval().to(device)
    return {"umm": umm, "wvae": wvae_encoder}


def process_after_downloading(model_path):
    with rank_zero_first(is_global=False):
        noise_dir = f"{model_path}/noise"
        if not os.path.exists(noise_dir):
            noise_tar = f"{model_path}/noise.tar"
            assert os.path.exists(noise_tar), f"{noise_tar} not exists"
            cmd = f"tar -xf {noise_tar} -C {model_path}"
            ret = _run_command(cmd=cmd)
            if ret.exit != 0:
                raise ExtractError(f"failed to extract {noise_tar}")
            os.remove(noise_tar)

        noise_lst = [
            os.path.join(noise_dir, item)
            for item in os.listdir(noise_dir)
            if item.endswith(".wav")
        ]
    return {"noise_lst": noise_lst}
