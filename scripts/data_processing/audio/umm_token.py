import librosa
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from recipes.umm.modules.lit_module import Stage3


def preprocess_audio(audio_bin, sample_rate, resampler, device, *_, **__):
    wav, sr = librosa.load(audio_bin, sr=None)
    if wav.size == 0:
        return None, None
    audio_dur = wav.shape[-1] / float(sr)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]

    scale = max(0.001, np.max(np.abs(wav)))
    wav = wav / scale * 0.95

    wav = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)

    if sr != sample_rate:
        if sr not in resampler:
            resampler[sr] = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resampler[sr](wav)

    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, device, *_, **__):
    if not batch:
        yield from batch

    for wav in batch:
        yield model.wav2token(wav).cpu().squeeze().numpy()


def load_model(device, model_path, *_, **__):
    return Stage3.load_from_checkpoint(model_path).eval().to(device)
