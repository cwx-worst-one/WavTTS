import librosa
import numpy as np
import torch
from scipy.signal import get_window


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


mel_basis_cache = {}


def torch_wav2spec(wav,
                   num_mels=160,
                   fmin=0,
                   fmax=12000,
                   mel_basis=None,
                   sample_rate=24000,
                   fft_size=1200,
                   hop_size=240,
                   win_length=1200,
                   eps=1e-6):
    fft_window = get_window('hann', win_length, fftbins=True)
    fft_window = torch.FloatTensor(fft_window).to(wav.device)
    if mel_basis is None:
        mel_basis = mel_basis_cache.get('mel_basis')
        if mel_basis is None:
            mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis_cache['mel_basis'] = mel_basis
    mel_basis = torch.FloatTensor(mel_basis).to(wav.device)
    x_stft = torch.stft(wav, fft_size, hop_size, win_length, fft_window,
                        center=True, pad_mode='constant', normalized=False, onesided=True, return_complex=True)
    linear_spc = torch.abs(x_stft)
    mel = mel_basis @ linear_spc
    mel = torch.log10(torch.clamp_min(mel, eps))  # (n_mel_bins, T)
    return mel.transpose(1, 2)
