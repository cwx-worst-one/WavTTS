import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from scripts.data_processing.audio.wvae_mel_utils import (
    mel_spectrogram_torch,
    spectrogram_torch,
)


def preprocess_audio(audio_bin, sample_rate, resampler, device, freq=40, *_, **__):
    wav, sr = librosa.load(audio_bin, sr=None)
    audio_dur = wav.shape[0] / float(sr)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]
    wav = torch.as_tensor(wav, dtype=torch.float32, device=device)
    if sr != sample_rate:
        if sr not in resampler:
            resampler[sr] = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resampler[sr](wav)
    wav = wav.cpu().numpy()
    wav *= 1.0 / max(0.01, np.max(np.abs(wav)))
    wav = torch.from_numpy(wav).float()
    wav = torch.stack([wav]).unsqueeze(1).float()
    pad_mod = sample_rate // freq
    wav = F.pad(
        wav,
        (0, (wav.size(-1) // pad_mod + 1) * pad_mod - wav.size(-1), 0, 0, 0, 0),
        value=0.0,
    )
    return wav, audio_dur


def process_batch(
    model,
    batch,
    device,
    sample_rate,
    n_fft=2048,
    hop_length=300,
    win_length=1200,
    freq=40,
):
    if not batch:
        yield from batch
    length = [e.shape[-1] for e in batch]
    max_length = max(length)
    padding_length = np.arange(4, 64, 4)
    for candidate in padding_length:
        if max_length <= candidate:
            max_length = candidate
            break
    batch = [
        F.pad(e, (0, max_length - length[i], 0, 0, 0, 0), value=0.0)
        for i, e in enumerate(batch)
    ]
    batch_wav = torch.cat(batch, dim=0).to(device)
    batch_spec = spectrogram_torch(
        batch_wav.squeeze(1), n_fft, sample_rate, hop_length, win_length
    )
    batch_mel = mel_spectrogram_torch(
        batch_wav.squeeze(1), n_fft, 80, sample_rate, hop_length, win_length, 0.0, None
    )

    batch_output = model(
        batch_wav.to(device), batch_spec.to(device), batch_mel.to(device)
    )

    pad_mod = sample_rate // freq
    for i, ilen in enumerate(length):
        yield batch_output[i, : ilen // pad_mod].cpu().numpy()


def model_path_patten(feature_version):
    return f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_mel_token_{feature_version}/wavevae_encoder_%d.pt"  # noqa
