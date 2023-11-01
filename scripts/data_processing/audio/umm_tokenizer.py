import librosa
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F


def preprocess_audio(audio_bin, sample_rate, *_, **__):
    hop_length, frame_rate = 150, 40
    rate = int(sample_rate / frame_rate)

    wav, sr = librosa.load(audio_bin, sr=None)
    audio_dur = wav.shape[-1] / float(sr)
    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav[:, 0]
    if sr != sample_rate:
        print("convert sr ...")
        wav = librosa.core.resample(wav, sr, sample_rate)

    scale = max(0.001, np.max(np.abs(wav)))
    wav = wav / scale * 0.95
    pad_len = wav.shape[-1] % (hop_length * 4)
    if pad_len != 0:
        pad_len = hop_length * 4 - pad_len
    wav = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
    wav = torch.nn.functional.pad(wav, (0, pad_len))
    if wav.size(-1) % rate > 0:
        wav = F.pad(wav, (0, rate - (wav.size(-1) % rate)), "constant", 0)
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, device, *_, **__):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not batch:
        yield from batch

    for wav in batch:
        yield model(wav.to(device)).squeeze().cpu().numpy()


def model_path_patten(feature_version):
    return f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/umm_tokenizer_{feature_version}/umm_tokenizer_%d.pt"  # noqa
