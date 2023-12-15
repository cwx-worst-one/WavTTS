import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torchaudio.transforms import Resample


def preprocess_audio(
    audio_bin, sample_rate, resampler, device, trim=True, freq=40, *_, **__
):
    wav, sr = librosa.load(audio_bin, sr=None)
    if wav.size == 0:
        return None, None
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
    if trim:
        wav = trim_silence(wav)
    wav = torch.from_numpy(wav).float()
    wav = torch.stack([wav]).unsqueeze(1).float()
    pad_mod = sample_rate // freq
    wav = F.pad(
        wav,
        (0, (wav.size(-1) // pad_mod + 1) * pad_mod - wav.size(-1), 0, 0, 0, 0),
        value=0.0,
    )
    return wav, audio_dur


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=torch.hann_window(win_size).to(dtype=y.dtype, device=y.device),
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # torch 1.8 compatable. stft could support half type
    old_type = y.dtype
    y = y.float()
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = spec.to(old_type)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec


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

    _, batch_m, batch_logs = model(
        batch_wav.to(device), batch_spec.to(device), batch_mel.to(device)
    )
    pad_mod = sample_rate // freq
    for i, ilen in enumerate(length):
        m, logs = (
            batch_m[i : i + 1, :, : ilen // pad_mod],
            batch_logs[i : i + 1, :, : ilen // pad_mod],
        )
        m = m[0].permute(1, 0)
        logs = logs[0].permute(1, 0)
        bn = torch.cat([m, logs], -1)
        bn = bn.cpu().numpy()
        yield bn


def model_path_patten(feature_version):
    return f"hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_mel_{feature_version}/wavevae_encoder_%d.pt"  # noqa
