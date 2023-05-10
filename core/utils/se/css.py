''' css algorithm layer of mic arrays.
'''
import torch
import numpy as np


def cal_mic_corr(mic0_subband, mic1_subband):
    '''Calculate coherence of mic0 and mic1
    Inputs are:
    mic0_subband: input first mic subband signal, [B, T, F, 2]
    mic1_subband: input second mic subband signal, [B, T, F, 2]
    Outpus are:
    out_xcor_re: real part of cross corr [B, T, F]
    out_xcor_im: image part of cross corr [B, T, F]
    '''
    with torch.no_grad():
        lamda = 0.7165
        mic_xcor_re = (
            mic0_subband[:, :, :, 0] * mic1_subband[:, :, :, 0]
            + mic0_subband[:, :, :, 1] * mic1_subband[:, :, :, 1]
        )
        mic_xcor_im = (
            mic0_subband[:, :, :, 1] * mic1_subband[:, :, :, 0]
            - mic0_subband[:, :, :, 0] * mic1_subband[:, :, :, 1]
        )
        mic0_auto_cor = (
            mic0_subband[:, :, :, 0] * mic0_subband[:, :, :, 0]
            + mic0_subband[:, :, :, 1] * mic0_subband[:, :, :, 1]
        )
        mic1_auto_cor = (
            mic1_subband[:, :, :, 0] * mic1_subband[:, :, :, 0]
            + mic1_subband[:, :, :, 1] * mic1_subband[:, :, :, 1]
        )

        lamda1 = 1.0 - lamda
        for i in range(mic0_subband.shape[1] - 1):
            mic0_auto_cor[:, i + 1, :] = torch.add(
                lamda * mic0_auto_cor[:, i, :], lamda1 * mic0_auto_cor[:, i + 1, :]
            )
            mic1_auto_cor[:, i + 1, :] = torch.add(
                lamda * mic1_auto_cor[:, i, :], lamda1 * mic1_auto_cor[:, i + 1, :]
            )
            mic_xcor_re[:, i + 1, :] = torch.add(
                lamda * mic_xcor_re[:, i, :], lamda1 * mic_xcor_re[:, i + 1, :]
            )
            mic_xcor_im[:, i + 1, :] = torch.add(
                lamda * mic_xcor_im[:, i, :], lamda1 * mic_xcor_im[:, i + 1, :]
            )
        auto_cor = torch.sqrt(mic0_auto_cor * mic1_auto_cor) + 1e-10
        out_xcor_re = mic_xcor_re / auto_cor
        out_xcor_im = mic_xcor_im / auto_cor
        return out_xcor_re, out_xcor_im


def cal_tmp_corr(mic0_subband, mic1_subband):
    '''Calculate the temporance corr of two mic signals.
    Inputs are:
    mic0_subband: input first mic subband signal, [B, T, F, 2]
    mic1_subband: input second mic subband signal, [B, T, F, 2]
    Outputs are:
    out_xcor_re: real part of cross corr [B, T, F]
    out_xcor_im: image part of cross corr [B, T, F]
    '''
    with torch.no_grad():
        mic_xcor_re = (
            mic0_subband[:, :, :, 0] * mic1_subband[:, :, :, 0]
            + mic0_subband[:, :, :, 1] * mic1_subband[:, :, :, 1]
        )
        mic_xcor_im = (
            mic0_subband[:, :, :, 1] * mic1_subband[:, :, :, 0]
            - mic0_subband[:, :, :, 0] * mic1_subband[:, :, :, 1]
        )
        mic0_auto_cor = (
            mic0_subband[:, :, :, 0] * mic0_subband[:, :, :, 0]
            + mic0_subband[:, :, :, 1] * mic0_subband[:, :, :, 1]
        )
        mic1_auto_cor = (
            mic1_subband[:, :, :, 0] * mic1_subband[:, :, :, 0]
            + mic1_subband[:, :, :, 1] * mic1_subband[:, :, :, 1]
        )

        auto_cor = torch.sqrt(mic0_auto_cor * mic1_auto_cor) + 1e-10
        out_xcor_re = mic_xcor_re / auto_cor
        out_xcor_im = mic_xcor_im / auto_cor
        return out_xcor_re, out_xcor_im


def cal_target_corr(xcor_re, direction, mic_space):
    '''Calculate the mic correhence of target direction.
    Inputs are:
    xcor_re: real part of cross corr of mics [B, T, F]
    xcor_im: image part of cross corr of mics [B, T, F]
    direction: target signal direction [B]
    mic_space: mic distance of two mics(/m).
    Outpus are:
    target_cor_re: real part of corr of target direction [B, T, F]
    target_cor_im: image part of corr of target direction [B, T, F]
    '''
    with torch.no_grad():
        batch_size, n_frames, n_f = xcor_re.size()
        nfreq = (n_f - 1.0) * 2.0
        tdoa_np = 16000.0 * 2.0 * np.pi * mic_space * torch.cos(direction / 180.0 * np.pi) / 340.0
        tdoa = tdoa_np.to(xcor_re.device)
        freq_idx = torch.arange(0, n_f, 1).to(tdoa.device)
        freq_idx_float = freq_idx.float() + 1.0
        freq_idx_float = freq_idx_float / nfreq
        target_cor_re = torch.ones(batch_size, n_frames, n_f).to(tdoa.device)
        target_cor_im = torch.ones(batch_size, n_frames, n_f).to(tdoa.device)
        for bidx in range(batch_size):
            tdoa_freq = tdoa[bidx] * freq_idx_float
            cos_beta = torch.cos(tdoa_freq)
            sin_beta = torch.sin(tdoa_freq)
            target_cor_re[bidx, :, :] = cos_beta
            target_cor_im[bidx, :, :] = sin_beta
        return target_cor_re, target_cor_im


def cal_css_gain(xcor_re, xcor_im, tdoa):
    '''Calculate css gain.
    xcor_re: eal part of cross corr of mics [B, T, F].
    xcor_im: image part of cross corr of mics [B, T, F]
    tdoa: target direction of arrival [B]
    '''
    with torch.no_grad():
        batch_size, n_frames, n_f = xcor_re.size()
        nfreq = (n_f - 1.0) * 2.0
        freq_idx = torch.arange(0, n_f, 1).to(tdoa.device)
        freq_idx_float = freq_idx.float() + 1.0
        freq_idx_float = freq_idx_float / nfreq
        gain_out = torch.ones(batch_size, n_frames, n_f).to(tdoa.device)
        xcor_abs2 = xcor_re**2 + xcor_im**2
        for bidx in range(batch_size):
            tdoa_freq = tdoa[bidx] * freq_idx_float
            cos_beta = torch.cos(tdoa_freq)
            sin_beta = torch.sin(tdoa_freq)
            numerator = 0.5 * (xcor_abs2[bidx, :, :] - 1.0) + 1e-10
            denominator = xcor_re[bidx, :, :] * cos_beta + xcor_im[bidx, :, :] * sin_beta
            denominator = denominator - 1.0
            denominator = torch.abs(denominator) + 1e-10
            numerator = torch.abs(numerator)
            gain_out[bidx, :, :] = numerator / denominator
        gain_out = torch.clamp(gain_out, min=0, max=1.0)
        gain_out = torch.sqrt(gain_out)
        return gain_out


def css_gain_mul(mic0_subband, css_gain):
    '''Apply the gain to mic subband.
    Inputs are:
    mic0_subband: subband signal of first mic [B, T, F, 2]
    css_gain: css gain calculated [B, T, F]
    Outpus are:
    mic_gain_out: gained subband signal of first mic [B, T, F, 2]
    '''
    with torch.no_grad():
        mic_re = mic0_subband[:, :, :, 0]
        mic_im = mic0_subband[:, :, :, 1]
        mic_re_gain = mic_re * css_gain
        mic_im_gain = mic_im * css_gain
        mic_re_gain = torch.unsqueeze(mic_re_gain, 3)
        mic_im_gain = torch.unsqueeze(mic_im_gain, 3)
        mic_gain_out = torch.cat([mic_re_gain, mic_im_gain], 3)
        return mic_gain_out


def css(mic0_subband, mic1_subband, direction, mic_space):
    '''main processing of css.
    Inputs are:
    mic0_subband: input first mic subband signal, [B, T, F, 2]
    mic1_subband: input second mic subband signal, [B, T, F, 2]
    direction: target direction, [B]
    mic_space: mic distance of two mics(/m).
    Output is:
    css_gain: css gain for target directions [B, T, F].
    '''
    with torch.no_grad():
        device = mic0_subband.device
        out_xcor_re, out_xcor_im = cal_mic_corr(mic0_subband, mic1_subband)
        tdoa_np = 16000.0 * 2.0 * np.pi * mic_space * torch.cos(direction / 180.0 * np.pi) / 340.0
        tdoa = tdoa_np.to(device)
        css_gain = cal_css_gain(out_xcor_re, out_xcor_im, tdoa)
        return css_gain
