import torch
from torch import nn

from librosa.filters import mel as librosa_mel_fn
from distutils.version import LooseVersion


MAX_WAV_VALUE = 32768.0
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    '''
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    '''
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    '''
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    name = "sr_{}_nfft_{}_win_{}_hop_{}_fmin_{}_fmax_{}_device_{}".format(sampling_rate, n_fft, win_size, hop_size, fmin, fmax, dtype_device)
    if name not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[name] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if name not in hann_window:
        hann_window[name] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # torch 1.8 compatable. stft could support half type
    old_type = y.dtype
    y = y.float()
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[name],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = spec.to(old_type)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[name], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def log_mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    name = "sr_{}_nfft_{}_win_{}_hop_{}_fmin_{}_fmax_{}_device_{}".format(sampling_rate, n_fft, win_size, hop_size, fmin, fmax, y.device)
    if name not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[name] = torch.from_numpy(mel).float().to(y.device)
        hann_window[name] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect'
    )
    y = y.squeeze(1)

    if is_pytorch_17plus:
        spec = torch.stft(y,
                          n_fft,
                          hop_length=hop_size,
                          win_length=win_size,
                          # window=hann_window[str(fmax) + '_' + str(y.device)],
                          window=hann_window[name],
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=False)
    else:
        spec = torch.stft(y,
                          n_fft,
                          hop_length=hop_size,
                          win_length=win_size,
                          # window=hann_window[str(fmax) + '_' + str(y.device)],
                          window=hann_window[name],
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  # linear spec
    mel = torch.matmul(mel_basis[name], spec)
    log_mel = mel.clamp(1e-5).log10()
    return log_mel
