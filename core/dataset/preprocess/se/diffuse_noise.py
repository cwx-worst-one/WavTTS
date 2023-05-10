"""
producing diffuse noise
"""
import numpy as np
import torch
from core.utils import get_local_rank


class DiffuseNoise:
    '''
    class diffuse Noise
    '''

    def __init__(self, cuda_version=True):
        '''
        init
        '''
        NFFT = 256
        HOP = int(NFFT / 4)
        hanning_win = np.hanning(NFFT)
        hanning_win = hanning_win * np.sqrt(HOP / np.sum(hanning_win**2))

        hamming_win = np.hamming(NFFT)
        hamming_win = hamming_win * np.sqrt(HOP / np.sum(hamming_win**2))
        hanning_win = torch.from_numpy(hanning_win)
        hamming_win = torch.from_numpy(hamming_win)
        if cuda_version:
            self.hanning_win = hanning_win.cuda(get_local_rank())
            self.hamming_win = hamming_win.cuda(get_local_rank())
        else:
            self.hanning_win = hanning_win
            self.hamming_win = hamming_win

    def stft(self, in_x, nfft, hop, wintype):
        '''stft transform only used in diffuse noise module'''
        if wintype == 'hanning':
            win = self.hanning_win
        else:
            win = self.hamming_win
        delay1 = nfft - 1
        delay2 = nfft - 1
        bsz = in_x.shape[0]  # bsz * mic_num
        x = torch.zeros([bsz, delay1 + delay2 + in_x.shape[-1]], device=in_x.device)
        x[:, delay1 : delay1 + in_x.shape[-1]] = in_x
        nx = x.shape[-1]
        col = int((nx - nfft) / hop + 1)
        x_spec = torch.zeros([bsz, col, int(nfft / 2 + 1)], dtype=torch.complex128, device=x.device)
        # x : bsz * num_samples
        # new_x: bsz * col, nfft
        new_x = x.as_strided([bsz, col, nfft], [x.shape[-1], hop, 1])
        x_win = new_x * win
        x_spec = torch.fft.rfft(x_win, dim=-1)
        return x_spec

    def istft(self, x_spec, fft, hop, wintype):
        """
        istft transform only used in diffuse noise module
        """
        if wintype == 'hanning':
            win = self.hanning_win
        else:
            win = self.hamming_win
        hop = int(hop)
        col = x_spec.shape[1]
        x = torch.fft.irfft(x_spec, dim=2)
        bsz = x_spec.shape[0]
        nfft = int(fft / hop)
        # y: hop, bsz, col + nfft - 1
        y = torch.zeros(hop, bsz, col + nfft - 1, device=x_spec.device)
        x = x * win
        x = x.reshape(bsz, col, nfft, hop)
        x = x.permute(3, 0, 1, 2)  # hop, bsz, col, off
        # col, col, hop
        kernel = torch.eye(col, device=x.device).reshape(col, col, 1) * torch.ones(
            1, hop, device=x.device
        )
        kernel = kernel.permute(2, 0, 1)
        kernel = kernel.double()
        for off in range(nfft):
            y[:, :, off : off + col] += torch.bmm(x[:, :, :, off], kernel)
        y = y.permute(1, 2, 0)
        y = y.reshape(bsz, (col + nfft - 1) * hop, 1)
        delay1 = fft - 1
        return y[:, delay1 + 1 : -1, :]

    def mix_signals(self, x_in, mix_matrix, split_size=16):
        """
        mix input siganl with diffuse matirx
        """
        bsz, mic_num, sample_len = x_in.shape
        fft_len = (mix_matrix.shape[-1] - 1) * 2
        col = int((sample_len + fft_len - 2) / (fft_len / 4) + 1)
        x_spec = torch.zeros(
            [bsz, mic_num, col, int(fft_len / 2 + 1)], dtype=torch.complex128, device=x_in.device
        )
        mix_spec = torch.zeros(x_spec.shape, dtype=torch.complex128, device=x_in.device)
        x_spec = self.stft(
            x_in.reshape(-1, sample_len), fft_len, int(fft_len / 4), 'hanning'
        ).reshape(bsz, mic_num, col, int(fft_len / 2 + 1))
        zero_dim_values = (
            x_spec[:, 0, 0, 0].expand(mic_num, col, bsz).reshape(bsz, mic_num, col).clone()
        )
        shapes = mix_spec.permute(0, 3, 1, 2).shape
        zero_dim_values = mix_spec[:, :, :, 0].clone()
        # b * m * m * k -> b * k * m * m -> B * m * m
        mix_matrix = mix_matrix.permute(0, 3, 1, 2)
        mix_matrix = mix_matrix.reshape(-1, mix_matrix.shape[-2], mix_matrix.shape[-1])
        # b * m * n * k -> b * k * m * n -> B * m * n
        x_spec = x_spec.permute(0, 3, 1, 2)
        x_spec = x_spec.reshape(-1, x_spec.shape[-2], x_spec.shape[-1])
        mix_spec = (
            torch.complex(torch.bmm(mix_matrix, x_spec.real), torch.bmm(mix_matrix, x_spec.imag))
            .reshape(shapes)
            .permute(0, 2, 3, 1)
        )
        mix_spec[:, :, :, 0] = zero_dim_values
        mix = torch.zeros([bsz, mic_num, sample_len], device=mix_spec.device)
        mix = mix.reshape(-1, sample_len)
        mix_spec = mix_spec.reshape(-1, mix_spec.shape[-2], mix_spec.shape[-1])
        # TODO(litianyu.y): remove this split to speed up
        num = (mix_spec.shape[0] + split_size - 1) // split_size  # split size 16
        new_mix_specs = mix_spec.chunk(num, dim=0)
        off = 0
        for new_mix_spec in new_mix_specs:
            mix[off : off + new_mix_spec.shape[0], ...] = self.istft(
                new_mix_spec, fft_len, fft_len / 4, 'hanning'
            )[:, 0:sample_len, 0]
            off += new_mix_spec.shape[0]
        mix = mix.reshape(bsz, mic_num, sample_len)
        return mix


def gen_linear_diffuse(fs, mic_num, distance):
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
    # out = mix_signals(x, diffuse_matrix, 'eig')
    return diffuse_matrix


def gen_circular_diffuse(fs, mic_num, radius):
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
    # out = mix_signals(x, diffuse_matrix, 'eig')
    return diffuse_matrix


def gen_mix_matrix(diffuse_matrix, method='eig'):
    '''
    calucate mix_matrix
    '''
    fft_len = (diffuse_matrix.shape[2] - 1) * 2
    mix_matrix = np.zeros(diffuse_matrix.shape)
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
    return mix_matrix


def gen_diffuse(fs, mic_num, array_type, mic_distance, radius):
    """
    generate diffuse noise with given parameters
    """
    if array_type == "circular":
        assert radius is not None
        return gen_circular_diffuse(fs, mic_num, radius)
    if array_type == "linear":
        assert mic_distance is not None
        return gen_linear_diffuse(fs, mic_num, mic_distance)
    raise Exception("gen_diffuse: array_type error!")
