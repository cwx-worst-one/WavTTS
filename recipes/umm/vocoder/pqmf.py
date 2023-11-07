# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import os
import numpy as np
import torch
import torch.nn.functional as F

import scipy.optimize as optimize
from scipy.signal import kaiser
from scipy.io.wavfile import read, write
import librosa


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
            / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super(PQMF, self).__init__()

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)

    def forward(self, x):
        return self.analysis(x)


def _objective(cutoff_ratio):
    h_proto = design_prototype_filter(num_taps, cutoff_ratio, beta)
    conv_h_proto = np.convolve(h_proto, h_proto[::-1], mode='full')
    length_conv_h = conv_h_proto.shape[0]
    half_length = length_conv_h // 2

    check_steps = np.arange((half_length) // (2 * num_subbands)) * 2 * num_subbands
    _phi_new = conv_h_proto[half_length:][check_steps]
    phi_new = np.abs(_phi_new[1:]).max()
    # Since phi_new is not convex, This value should also be considered.
    diff_zero_coef = np.abs(_phi_new[0] - 1 / (2 * num_subbands))

    return phi_new + diff_zero_coef


if __name__ == "__main__":
    num_subbands = 4
    num_taps = 62
    beta = 9.0

    ret = optimize.minimize(_objective, np.array([0.01]),
                            bounds=optimize.Bounds(0.01, 0.99))
    opt_cutoff_ratio = ret.x[0]
    print(f"optimized cutoff ratio = {opt_cutoff_ratio:.08f}")

    pqmf = PQMF(subbands=num_subbands,
                taps=num_taps,
                cutoff_ratio=opt_cutoff_ratio,
                beta=beta)
    pqmfs = [
        PQMF(subbands=2, taps=62, cutoff_ratio=0.26699457, beta=9.0),
        PQMF(subbands=3, taps=62, cutoff_ratio=0.18366124, beta=9.0),
        PQMF(subbands=5, taps=72, cutoff_ratio=0.11463421, beta=9.0),
        PQMF(subbands=7, taps=82, cutoff_ratio=0.08427813, beta=9.0),
        PQMF(subbands=11, taps=92, cutoff_ratio=0.05690741, beta=9.0),
    ]
    for i, pqmf in enumerate(pqmfs):
        data, sr = librosa.load("000001.wav", sr=None)
        data = torch.stack([torch.from_numpy(data).unsqueeze(0)]).float()
        result = pqmf.analysis(data)
        wav = pqmf.synthesis(result)
        wav = wav[0][0].numpy()
        print(np.max(np.abs(wav)), data.squeeze().shape, wav.shape)
        wav = wav / np.max(np.abs(wav)) * 32767.0
        write("test_{}.wav".format(i), sr, wav.astype(np.int16))
