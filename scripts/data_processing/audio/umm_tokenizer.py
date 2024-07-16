import logging

import librosa
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from torchaudio.transforms import Resample

logger = logging.getLogger(__name__)


def preprocess_audio(audio_bin, sample_rate, resampler, device, *_, **__):
    hop_length, frame_rate = 150, 40
    rate = int(sample_rate / frame_rate)

    try:
        wav, sr = librosa.load(audio_bin, sr=None)
        if wav.size == 0:
            return None, None
    except Exception as e:
        logger.error("read audio error", exc_info=True)
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

    pad_len = wav.shape[-1] % (hop_length * 4)
    if pad_len != 0:
        pad_len = hop_length * 4 - pad_len
    wav = F.pad(wav, (0, pad_len))
    if wav.size(-1) % rate > 0:
        wav = F.pad(wav, (0, rate - (wav.size(-1) % rate)), "constant", 0)
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, device, *_, **__):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not batch:
        yield from batch
    one_min = 60 * 24000
    for wav in batch:
        try:
            token = []
            for start in range(0, wav.shape[-1], one_min):
                sub_wav = wav[:, start : start + one_min]
                token.append(model(sub_wav.to(device)).squeeze().cpu().numpy())
            yield np.hstack(token)
        except Exception as e:
            logger.error(f"process wav error", exc_info=e)
            yield None


def model_path_patten(feature_version):
    return f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/umm_tokenizer_{feature_version}/umm_tokenizer_%d.pt"  # noqa
