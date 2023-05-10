"""
producing diffuse noise
"""
import numpy as np


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


def gen_linear_diffuse(x, fs, mic_num, distance):
    """
    generate linear diffuse
    """
    c = 340
    fft_len = 256
    d = distance
    ww = 2 * np.pi * fs * np.arange(0, int(fft_len / 2 + 1)) / fft_len
    diffuse_matrix = np.zeros([mic_num, mic_num, int(fft_len / 2 + 1)])
    for p in range(mic_num):
        for q in range(mic_num):
            if p == q:
                diffuse_matrix[p, q, :] = np.ones([1, 1, int(fft_len / 2 + 1)])
            else:
                diffuse_matrix[p, q, :] = np.sinc(ww * np.abs(p - q) * d / c / np.pi)
    out = mix_signals(x, diffuse_matrix, 'eig')
    return out


def gen_circular_diffuse(x, fs, mic_num, radius):
    """
    generate circular diffuse
    """
    theta_step = np.pi * 2 / mic_num
    c = 340
    fft_len = 256
    d = radius * 2
    ww = 2 * np.pi * fs * np.arange(0, int(fft_len / 2 + 1)) / fft_len
    diffuse_matrix = np.zeros([mic_num, mic_num, int(fft_len / 2 + 1)])
    for p in range(mic_num):
        for q in range(mic_num):
            if p == q:
                diffuse_matrix[p, q, :] = np.ones([1, 1, int(fft_len / 2 + 1)])
            else:
                tmp = np.sin(np.abs(p - q) * theta_step / 2)
                diffuse_matrix[p, q, :] = np.sinc(ww * d * tmp / c / np.pi)
    out = mix_signals(x, diffuse_matrix, 'eig')
    return out


def gen_diffuse(x, fs, mic_num, array_type, mic_distance, radius):
    """
    generate diffuse noise with given parameters
    """
    if array_type == "circular":
        assert radius is not None
        return gen_circular_diffuse(x, fs, mic_num, radius)
    if array_type == "linear":
        assert mic_distance is not None
        return gen_linear_diffuse(x, fs, mic_num, mic_distance)
    raise Exception("gen_diffuse: array_type error!")
