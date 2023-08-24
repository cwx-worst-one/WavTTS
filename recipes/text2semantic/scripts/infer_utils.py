import os
import random

import librosa
import numpy as np
import torch

import samantha.utils.hdfs_helper as hh
from samantha.utils.distributed import rank_zero_first
from scipy.io.wavfile import write


def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device


def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 2)
    return


hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def trim_prompt_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset
    wav = np.pad(wav, (5400, 5400))
    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    start_idx = max(index[0] - 3200, 0)
    stop_idx =  index[1] + 2400 # avoid trim voiced segment.
    trimmed = wav[start_idx:stop_idx]
    trimmed = np.pad(trimmed, (0, 2400)) # pad silience (0.0) 100ms.
    return trimmed


def trim_silence(wav):
    wav = np.pad(wav, (5400, 5400))
    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    start_idx = max(index[0] - 3200, 0)
    stop_idx =  index[1] + 5400 # avoid trim voiced segment.
    trimmed = wav[start_idx:stop_idx]
    return trimmed


def load_torch_script(model_path, rank, cache_dir):
    device = f"cuda:{rank}"
    if hh.ishdfs(model_path):
        model_path = model_path % rank
        os.makedirs(cache_dir, exist_ok=True)
        fn = os.path.basename(model_path)
        local_path = os.path.join(cache_dir, fn)
        if not os.path.exists(local_path):
            success = hh.get(model_path, local_path)
            if not success:
                raise ConnectionError(
                    f"failed to retrieve {model_path} to {local_path}"
                )
        return torch.jit.load(local_path).to(device).eval()
    else:
        return torch.jit.load(model_path).to(device).eval()