import json
import pickle

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from samantha.dataio.webdataset.writer import Writer


def preprocess_audio(audio_bin, sample_rate, *_, **__):
    wav, sr = librosa.load(audio_bin, sr=None)
    audio_dur = wav.shape[0] / float(sr)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]
    wav *= 0.95 / max(0.01, np.max(np.abs(wav)))
    if sr != sample_rate:
        print("convert sr ...")
        wav = librosa.core.resample(wav, sr, sample_rate)
    wav = torch.from_numpy(trim_silence(wav)).float()
    pad_len = wav.size(-1) % 480
    wav = torch.nn.functional.pad(wav, (0, 480 - pad_len)).view(1, -1)
    return wav, audio_dur


def trim_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset.

    wav = np.pad(wav, (5400, 5400))

    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    # num_sil_samples = int(8 * 300)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - 3200, 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + 5400, len(wav))

    trimmed = wav[start_idx:stop_idx]
    return trimmed


def process_batch(model, batch, device, *_, **__):

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
        F.pad(e, (0, max_length - length[i]), value=0.0) for i, e in enumerate(batch)
    ]
    batch_wav = torch.cat(batch, dim=0)
    quant_indexes = model(batch_wav.to(device))
    for i, ilen in enumerate(length):
        ss = quant_indexes[i, : ilen // 480, :].cpu().numpy()
        yield ss


def model_path_patten(feature_version):
    return f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/ss_{feature_version}/ss_encoder_%d.pt"  # noqa
