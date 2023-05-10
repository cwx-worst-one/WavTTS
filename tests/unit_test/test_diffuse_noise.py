"""
unit test of producing diffuse noise
"""
import numpy as np
import torch
from core.dataset.preprocess.se.diffuse_noise import gen_diffuse, DiffuseNoise, gen_mix_matrix
from core.utils import get_local_rank


def stft(x, nfft, hop, wintype):
    """
    stft transform only used in diffuse noise module
    """
    if wintype == 'hanning':
        win = np.hanning(nfft)
        win = win * np.sqrt(hop / np.sum(win**2))
    else:
        win = np.hamming(nfft)
        win = win * np.sqrt(hop / np.sum(win**2))

    # L = x.shape[0]
    delay1 = nfft - 1
    delay2 = nfft - 1
    d1 = np.zeros([delay1, 1])
    d2 = np.zeros([delay2, 1])
    x = np.append(d1, x)
    x = np.append(x, d2)
    nx = x.shape[0]
    col = int((nx - nfft) / hop + 1)
    x_spec = np.zeros([col, int(nfft / 2 + 1)], dtype='complex128')
    for n in range(col):
        x_win = x[hop * n : hop * n + nfft] * win
        x_spec[n, :] = np.fft.rfft(x_win, axis=0)
    return x_spec


def istft(x_spec, nfft, hop, wintype):
    """
    istft transform only used in diffuse noise module
    """
    if wintype == 'hanning':
        win = np.hanning(nfft)
        win = win * np.sqrt(hop / np.sum(win**2))
    else:
        win = np.hamming(nfft)
        win = win * np.sqrt(hop / np.sum(win**2))
    hop = int(hop)
    frame_len = x_spec.shape[0]
    x = np.fft.irfft(x_spec, axis=1)
    ny = (frame_len - 1) * hop + nfft
    y = np.zeros([ny, 1])
    for l in range(frame_len):
        x[l, :] = x[l, :] * win
    start = 0
    for l in range(frame_len):
        y[start : start + nfft, 0] = y[start : start + nfft, 0] + x[l, :]
        start += hop
    delay1 = nfft - 1
    return y[delay1 + 1 : -1, :]


def mix_signals(x_in, diffuse_matrix, method='cholesky'):
    """
    mix input siganl with diffuse matirx
    """
    [sample_len, mic_num] = x_in.shape
    fft_len = (diffuse_matrix.shape[2] - 1) * 2
    col = int((sample_len + fft_len - 1 + fft_len - 1 - fft_len) / (fft_len / 4) + 1)
    x_spec = np.zeros([mic_num, col, int(fft_len / 2 + 1)], dtype='complex128')
    mix_matrix = np.zeros(diffuse_matrix.shape)
    mix_spec = np.zeros(x_spec.shape, dtype='complex128')
    mix = np.zeros([sample_len, mic_num])

    for m in range(mic_num):
        x_spec[m, :, :] = stft(x_in[:, m], fft_len, int(fft_len / 4), 'hanning')

    mix_spec[:, :, 0] = x_spec[0, 0, 0]
    for k in range(1, int(fft_len / 2 + 1)):
        if method == 'cholesky':
            mix_matrix[:, :, k] = np.linalg.cholesky(diffuse_matrix[:, :, k])
        elif method == 'eig':
            values, vectors = np.linalg.eig(diffuse_matrix[:, :, k])
            idx = values.argsort()[::1]
            values = values[idx]
            vectors = vectors[:, idx]
            values = np.diag(values)
            mix_matrix[:, :, k] = np.dot(np.sqrt(values), vectors).T
        else:
            print('no such method')
        mix_spec[:, :, k] = np.dot(mix_matrix[:, :, k], x_spec[:, :, k])
    for m in range(mic_num):
        mix[:, m] = istft(mix_spec[m, :, :], fft_len, fft_len / 4, 'hanning')[0:sample_len, 0]
    return mix


def gen_fake_audio(bsz, mic_num, min_length, max_length):
    '''gen fake batch audio datas'''
    lengths = np.random.randint(min_length, max_length, bsz)
    audio_datas = []
    max_length = max(lengths)
    audio_tensor = torch.zeros(bsz, mic_num, max_length)
    mask_tensor = torch.zeros(bsz, max_length)
    for bid in range(bsz):
        length = lengths[bid]
        audio = np.random.randn(mic_num, length)
        audio_datas.append(audio)
        audio_tensor[bid, :, :length] = torch.from_numpy(audio)
        mask_tensor[bid, :length] = 1
    return audio_datas, audio_tensor, mask_tensor


def gen_fake_rir(bsz=3, mic_num=2):
    '''gen fake rir'''
    fs = 16000
    array_type = 'linear'
    mic_distance = 0.027
    radius = 0.03
    rir = gen_diffuse(fs, mic_num, array_type, mic_distance, radius)
    batch_rir = gen_mix_matrix(rir, 'eig')
    batch_rir = [batch_rir] * bsz
    rir_tensor = torch.from_numpy(np.stack(batch_rir, axis=0))
    return rir, rir_tensor


# pylint: disable='invalid-name'
def test_mix_signals_without_padding(bsz=3, mic_num=2, length=12300):
    '''test_mix_signals_without_padding'''
    rir, rir_tensor = gen_fake_rir(bsz, mic_num)
    diffuse_module = DiffuseNoise(cuda_version=True)
    audio_datas, audio_tensor, mask_tensor = gen_fake_audio(bsz, mic_num, length, length + 1)
    base_out = []
    for bid in range(bsz):
        audio = audio_datas[bid]
        audio = audio.transpose(1, 0)
        base_out.append(mix_signals(audio, rir, 'eig'))
    base_out = np.stack(base_out, axis=0).transpose(0, 2, 1)
    base_out = torch.from_numpy(base_out)

    new_out = diffuse_module.mix_signals(
        audio_tensor.cuda(get_local_rank()), rir_tensor.cuda(get_local_rank())
    ).double()
    new_out = new_out.cpu()
    new_out *= mask_tensor.unsqueeze(1)
    assert torch.allclose(base_out, new_out, atol=3e-7)


def test_mix_signals_with_padding(bsz=3, mic_num=2, min_length=12300, max_length=12800):
    '''test_mix_signals_with_padding'''
    rir, rir_tensor = gen_fake_rir(bsz, mic_num)
    diffuse_module = DiffuseNoise(cuda_version=True)
    audio_datas, audio_tensor, mask_tensor = gen_fake_audio(bsz, mic_num, min_length, max_length)
    base_out = torch.zeros_like(audio_tensor)
    for bid in range(bsz):
        audio = audio_datas[bid]
        length = audio.shape[-1]
        audio = audio.transpose(1, 0)
        audio = mix_signals(audio, rir, 'eig')
        audio = audio.transpose(1, 0)
        base_out[bid, :, :length] = torch.from_numpy(audio)

    new_out = diffuse_module.mix_signals(
        audio_tensor.cuda(get_local_rank()), rir_tensor.cuda(get_local_rank())
    )
    new_out = new_out.cpu()
    new_out *= mask_tensor.unsqueeze(1)
    assert torch.allclose(base_out, new_out, atol=3e-7)
