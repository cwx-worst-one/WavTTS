''' fix beamforming algorithm for mic arrays. '''
import numpy as np
import torch


def generate_steer_vector_linear(mic_space, direction, mic_subband):
    '''Calculate steering vector of target direction w.r.t linear array.
    mic_space: mic distance of two mics(/m)
    direction: target signal direction [B]
    mic_subband: subband signal of first mic [B, nmic, T, F, 2].
    '''
    mic_num = mic_subband.shape[1]
    subband_num = mic_subband.shape[3]
    device = mic_subband.device
    nfreq = (subband_num - 1.0) * 2.0

    tar_r = 5.0
    sou_x = tar_r * torch.cos(direction / 180.0 * np.pi)  # B
    sou_y = tar_r * torch.sin(direction / 180.0 * np.pi)

    freq_idx = torch.arange(0, subband_num, 1.0, dtype=torch.float32).to(device)
    freq_idx_float = freq_idx / nfreq

    array_len = (mic_num - 1) * mic_space
    mic_x = torch.linspace(
        array_len / 2, -array_len / 2, steps=mic_num, device=sou_x.device
    )  # nmic
    mic_y = torch.zeros_like(mic_x)
    distance = (
        torch.sqrt(
            (mic_y[None, :] - sou_y[:, None]) ** 2 + (mic_x[None, :] - sou_x[:, None]) ** 2
        ).to(device)
        - tar_r
    )  # B, nmic
    sig_tao = distance * 16000.0 / 340.0
    temp = 2.0 * np.pi * sig_tao[:, None, :] * freq_idx_float[None, :, None]  # B F nmic
    steer_re = torch.cos(temp)
    steer_im = torch.sin(temp)
    return steer_re.unsqueeze(1), steer_im.unsqueeze(1)  # B 1 F nmic


def generate_steer_vector_circular(radius, direction, mic_subband):
    '''Calculate steering vector of target direction w.r.t circular array.
    radius: radius of the circular array(/m)
    direction: target signal direction [B]
    mic_subband: subband signal of first mic [B, nmic, T, F, 2].
    '''
    mic_num = mic_subband.shape[1]
    subband_num = mic_subband.shape[3]
    device = mic_subband.device
    nfreq = (subband_num - 1.0) * 2.0

    tar_r = 5.0
    sou_x = tar_r * torch.cos(direction / 180.0 * np.pi)  # B
    sou_y = tar_r * torch.sin(direction / 180.0 * np.pi)

    freq_idx = torch.arange(0, subband_num, 1.0, dtype=torch.float32).to(device)
    freq_idx_float = freq_idx / nfreq

    mic_angle = torch.linspace(0, np.pi * 2, steps=mic_num + 1)[:-1].to(device)
    mic_x = radius * torch.cos(mic_angle)
    mic_y = radius * torch.sin(mic_angle)  # nmic

    distance = (
        torch.sqrt(
            (mic_y[None, :] - sou_y[:, None]) ** 2 + (mic_x[None, :] - sou_x[:, None]) ** 2
        ).to(device)
        - tar_r
    )  # B, nmic
    sig_tao = distance * 16000.0 / 340.0
    temp = 2.0 * np.pi * sig_tao[:, None, :] * freq_idx_float[None, :, None]  # B F nmic
    steer_re = torch.cos(temp)
    steer_im = torch.sin(temp)
    return steer_re.unsqueeze(1), steer_im.unsqueeze(1)  # B 1 F nmic


def fixbeam(mic_subband, direction, array_type, mic_space):
    '''fix beamforming algo in subband domain for two mics.
    Inputs are:
    mic_subband: [B, nmic, T, F, 2]
    direction: target signal direction [B]
    mic_space: mic distance of two mics(/m)
    Outpus are:
    steer_mic_out: signal after ds bf [B, nmic, T, F, 2]
    fixbeam_out: fixbeam out of target direction [B, T, F, 2]
    '''
    with torch.no_grad():
        if array_type == 'linear':
            steer_re, steer_im = generate_steer_vector_linear(
                mic_space, direction, mic_subband
            )  # B 1 F nmic
        elif array_type == 'circular':
            steer_re, steer_im = generate_steer_vector_circular(
                mic_space, direction, mic_subband
            )  # B 1 F nmic
        fixbeam_out = []
        for j in range(mic_subband.shape[1]):
            fixbeam_out_re = (
                mic_subband[:, j, :, :, 0] * steer_re[..., j]
                - mic_subband[:, j, :, :, 1] * steer_im[..., j]
            )
            fixbeam_out_im = (
                mic_subband[:, j, :, :, 0] * steer_im[..., j]
                + mic_subband[:, j, :, :, 1] * steer_re[..., j]
            )
            fixbeam_out.append(torch.stack((fixbeam_out_re, fixbeam_out_im), dim=-1))  # B T F 2
        fixbeam_out = torch.stack(fixbeam_out, dim=1)  # B nmic T F 2
        return fixbeam_out.float(), torch.mean(fixbeam_out, dim=1).float()
