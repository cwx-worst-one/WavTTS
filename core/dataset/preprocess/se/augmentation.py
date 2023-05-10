import os
from pickle import TRUE
import sys
import argparse
import logging
import random
import torch
import torch.nn as nn
import scipy
import math
import numpy as np
import torchaudio
from ctypes import *
from .colornoise import powerlaw_psd_gaussian


def generate_zeros(s, length):
    '''generate zeros'''
    return np.zeros([s.shape[0], length])


def fix_length_zero(data, target_length):
    '''length fix'''
    mask = np.ones(target_length)
    ori_length = data.shape[-1]
    if ori_length < target_length:
        pad_left = int((target_len - ori_length) * random.random())
        pad_right = target_len - ori_length - pad_left
        lefts = generate_zeros(data, pad_left)
        rights = generate_zeros(data, pad_right)
        mask[:pad_left] = 0
        mask[pad_right:] = 0
        out = np.concatenate([lefts, data, rights], axis=-1)
    else:
        start = int(random.random() * (ori_length - target_length))
        out = data[..., start : start + target_length]
    return out, mask


def fix_length_repeat(data, target_length):
    '''pad length with repeat'''
    origin_length = data.shape[-1]
    if origin_length < target_length:
        repeat_times = target_length // origin_length + 1
        data = np.repeat(data, repeat_times)
    origin_length = data.shape[-1]
    start = int(random.random() * (origin_length - target_length))
    out = data[..., start : start + target_length]
    return out


def fix_lengths_zeros(datas, target_length):
    '''length fix'''
    outs = []
    ori_length = datas[0].shape[-1]
    if ori_length <= target_length:
        pad_left = random.randint(0, target_length - ori_length)

        pad_right = target_length - ori_length - pad_left
        lefts = generate_zeros(datas[0], pad_left)
        rights = generate_zeros(datas[0], pad_right)
        for data in datas:
            out = np.concatenate([lefts, data, rights], axis=-1)
            outs.append(out)
    else:
        start = random.randint(0, ori_length - target_length - 1)
        for data in datas:
            out = data[..., start : start + target_length]
            outs.append(out)
    return outs


def mix(speech, noise, snr, speech_flag, frame_max, gen_cfg):
    '''mix func for aec training'''
    frame_size = gen_cfg.frame_size
    if not speech_flag:
        bsz = noise.shape[0]
        random_tensor = torch.rand((bsz, 1, 1), device=noise.device)
        noise = noise * (random_tensor * 0.8 + 0.1)
        return noise
    bsz = speech.shape[0]
    max_length = speech.shape[-1]
    speech_t = speech[:, 0, ...]
    noise_t = noise[:, 0, ...]
    speech_t = speech_t - torch.mean(speech_t, dim=-1, keepdim=True)
    noise_t = noise_t - torch.mean(noise_t, dim=-1, keepdim=True)

    n_frames = int(max_length / frame_size)
    if frame_max:
        speech_framed = speech_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)
        noise_framed = noise_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)
        speech_power = torch.max((speech_framed**2).sum(-1), dim=-1).values
        noise_power = torch.max((noise_framed**2).sum(-1), dim=-1).values
    else:
        speech_power = torch.mean((speech_t**2), dim=-1)
        noise_power = torch.mean((noise_t**2), dim=-1)
    noise_scale = torch.sqrt(
        speech_power / (noise_power + gen_cfg.min_limit) / torch.pow(10.0, snr / 10.0)
    ).reshape(-1, 1, 1)
    noise = noise * noise_scale
    return noise


def nlp_simulation_custom(inp):
    '''nlp simulation custom'''
    simu_type = random.choice([0, 1, 2, 3])
    if simu_type == 0:
        ref = inp / 32768.0
        max_val = np.max(np.abs((ref)))
        tgt_val = np.random.uniform(1, 3)
        tgt_ratio = tgt_val / (max_val + 1e-5)
        bias_val = np.random.uniform(-max_val * 0.2, max_val * 0.2)
        echo = np.arctan((ref - bias_val) * tgt_ratio) / tgt_ratio + bias_val
        echo *= 32768
    elif simu_type == 1:
        ref = inp / 32768.0
        epsilon = np.random.uniform(2, 5)
        a = 5 / epsilon
        echo = a * ref / np.sqrt(a**2 + ref**2)
        echo *= 32768
    elif simu_type == 2:
        ref = inp / 32768.0
        epsilon = np.random.uniform(2, 5)
        a = epsilon / 10
        echo = 1 - np.exp(-a * ref)
        echo *= 32768
    elif simu_type == 3:
        ref = inp / 32768.0
        epsilon = np.random.uniform(2, 5)
        a = np.log10(5 / epsilon) + 0.1
        echo = 2 * a * ref + a * (ref**2) + ref**3
        echo *= 32768
    return echo


def iir_aug(x):
    '''iir aug'''
    x = torch.from_numpy(x).reshape(1, -1)
    r = -3 / 8 + np.random.rand(4).astype(np.float32) * (6 / 8)
    a = torch.tensor([1, r[0], r[1]])
    b = torch.tensor([1, r[2], r[3]])
    x = torchaudio.functional.lfilter(x, a, b)
    x = x.numpy()
    return x


def fft4gccphat(x, frame_len, frame_shift):
    time_steps = x.shape[-1]
    discard_right = (time_steps - frame_len) % frame_shift
    x_pad = x[: time_steps - discard_right]
    x_stft = torch.stft(
        x_pad, frame_len, hop_length=frame_shift, center=False, return_complex=False
    )
    return x_stft


def ifft4gccphat(X):
    device = X.device
    frame_length = (X.shape[-3] - 1) * 2
    n_frames = X.shape[-2]

    y = torch.zeros(n_frames, frame_length, device=device)
    for i in range(n_frames):
        spec = X[:, i, :]
        spec_cmplx = torch.complex(spec[..., 0], spec[..., 1])
        iffted = torch.fft.irfft(spec_cmplx)

        y[i] = iffted
    return y


def multiply_complex(in1, in2):
    assert in1.ndim == in2.ndim
    out_real = in1[..., 0] * in2[..., 0] - in1[..., 1] * in2[..., 1]
    out_imag = in1[..., 1] * in2[..., 0] + in1[..., 0] * in2[..., 1]
    return torch.stack([out_real, out_imag], -1)


def tde(
    mic,
    ref,
    frame_len=4096,
    hop=512,
    thresh=0.1,
    tolerance=128,
    significance=5,
    whole_utterance_tde=True,
    hangover_limit=5,
    return_tde=False,
):
    '''
    Args:
        mic [time_steps]
        ref [time_steps]
    if lag>0 then mic is slower than ref else mic is faster
    '''
    if mic.ndim == 1:
        SINGLE_CH = True
        mic = torch.from_numpy(mic).float()
        ref = torch.from_numpy(ref).float()
    else:
        SINGLE_CH = False
        mic = torch.from_numpy(mic[0]).float()
        ref = torch.from_numpy(ref[0]).float()
    delta = tolerance // 2
    assert mic.shape == ref.shape
    if mic.shape[-1] < frame_len:
        raise RuntimeError
    mic_stft = fft4gccphat(mic, frame_len, hop)
    ref_stft = fft4gccphat(ref, frame_len, hop)

    ref_stft_conj = ref_stft.clone()
    ref_stft_conj[..., 1] = -ref_stft_conj[..., 1]

    R = multiply_complex(mic_stft, ref_stft_conj)
    c = R / (R.pow(2).sum(-1).sqrt().clamp(min=1e-12).unsqueeze(-1))  # [F,T,2]
    c_ifft = ifft4gccphat(c)  # [n_frames,frame_len]
    n_frames = c_ifft.shape[0]

    if whole_utterance_tde:
        c_avg = c_ifft.mean(0)
        d_avg = torch.cat([c_avg[frame_len // 2 + 1 :], c_avg[: frame_len // 2 + 1]], dim=0)
        d_avg_abs = d_avg.abs()
        max_idx = torch.argmax(d_avg_abs)
        lag = max_idx - (frame_len // 2) + 1

        if d_avg_abs[max_idx] > thresh and d_avg_abs[max_idx] > d_avg_abs.mean() * significance:
            delay_detected = True
            if lag > tolerance:
                actual_mv = lag - delta
                ref_out = ref[:-actual_mv]
                ref_out = torch.cat([ref.new_zeros(actual_mv), ref_out])
            elif lag < -tolerance:
                actual_mv = -(lag + delta)
                ref_out = ref[actual_mv:]
                ref_out = torch.cat([ref_out, ref_out.new_zeros(actual_mv)])
            else:
                ref_out = ref
        else:
            delay_detected = False
            ref_out = ref
    else:
        d = torch.cat([c_ifft[:, frame_len // 2 + 1 :], c_ifft[:, : frame_len // 2 + 1]], dim=1)
        d_abs = d.abs()
        d_max_idx = torch.argmax(d.abs(), dim=1)
        d_smooth_idx = d_max_idx.new_ones(d_max_idx.shape) * ((frame_len // 2) + 1)
        detected = d_max_idx.new_zeros(d_max_idx.shape)
        hangover = 0
        undetected_cnt = 0
        for i in range(1, n_frames):
            if (
                d_abs[i][d_max_idx[i]] > thresh
                and d_abs[i][d_max_idx[i]] > d_abs[i].mean() * significance
            ):
                delay_detected = True
                undetected_cnt = 0
                detected[i] = 1
            else:
                delay_detected = False
                undetected_cnt += 1

            if undetected_cnt > hangover_limit:
                d_smooth_idx[i] = (frame_len // 2) + 1
            else:
                if delay_detected:
                    if d_max_idx[i] != d_smooth_idx[i - 1]:
                        hangover += 1
                    else:
                        hangover -= 1
                        hangover = max(hangover, 0)

                if hangover > hangover_limit:
                    d_smooth_idx[i] = d_max_idx[i]
                    hangover = 0
                else:
                    d_smooth_idx[i] = d_smooth_idx[i - 1]
        lags = d_smooth_idx - (frame_len // 2) + 1
        ref_out = ref.clone()
        time_delay = np.zeros(ref.shape)
        for i in range(n_frames):
            start_idx = frame_len + (i - 1) * hop
            end_idx = frame_len + i * hop
            ref_out[start_idx:end_idx] = lags[i]
            if lags[i] > tolerance:
                actual_mv = lags[i] - delta
                ref_start_idx = start_idx - actual_mv
                if ref_start_idx < 0:
                    ref_start_idx = 0
                    start_idx += actual_mv - start_idx
                ref_end_idx = end_idx - actual_mv
                ref_out[start_idx:end_idx] = ref[ref_start_idx:ref_end_idx]
            elif lags[i] < 0:
                actual_mv = -(lags[i] - delta)
                ref_start_idx = start_idx + actual_mv
                ref_end_idx = end_idx + actual_mv
                if ref_end_idx > ref.shape[-1]:
                    end_idx -= ref_end_idx - ref.shape[-1]
                    ref_end_idx = ref.shape[-1]
                ref_out[start_idx:end_idx] = ref[ref_start_idx:ref_end_idx]
            else:
                ref_out[start_idx:end_idx] = ref[start_idx:end_idx]

    if SINGLE_CH:
        pass
    else:
        ref_out = ref_out.unsqueeze(0)

    if return_tde:
        return time_delay
    else:
        return ref_out.numpy()


def parallel_aec_rlsyz_cpu_dsp_no_sub_16k(mic, ref, frame_size=128, fft_size=256):
    '''parallel_aec_rlsyz_cpu_dsp_no_sub_16k'''
    device = mic.device

    mic = mic.cpu().numpy()
    ref = ref.cpu().numpy()

    ref_tde = ref.copy()
    batch_size, data_len = mic.shape
    frames = data_len // frame_size
    subband_num = int(fft_size / 2 + 1)
    aec = np.zeros_like(mic)
    c_fun = cdll.LoadLibrary(
        './clibs/linear_aec_subband_multiproc_rlsm_dsp_erl_tde_nod_16k_no_sub/build/alg/libbdSPILAudioProc.so'
    )
    mic_ptr = mic.ctypes.data_as(POINTER(c_float))
    ref_ptr = ref_tde.ctypes.data_as(POINTER(c_float))
    aec_ptr = aec.ctypes.data_as(POINTER(c_float))

    c_fun.parallel_aec(
        mic_ptr, ref_ptr, aec_ptr, batch_size, data_len, frames * subband_num, frames
    )
    aec = torch.from_numpy(aec).to(device)
    ref_tde = torch.from_numpy(ref_tde).to(device)
    return aec, ref_tde
