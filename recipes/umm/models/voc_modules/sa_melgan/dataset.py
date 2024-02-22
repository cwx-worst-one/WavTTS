import torch
import torch.utils.data
import math


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def get_pitch_condition_torch(mels, pitch, frame_shift, sampling_rate, mode = 'singgan', onnx_export = False, noise_index = 1.0):
    # Different from get_pitch_condition, mels is batch, pitch is also batch.
    # mels: batch * timestep * 80
    # pitch: batch * 1 * timestep
    '''
    guided by paper https://arxiv.org/pdf/1810.11946.pdf
    '''

    mels_normed = mels + 4.0
    if mode == 'refinegan_1':
        raise 'Error, not implemented'

    elif mode == 'singgan_torch':
        pitch_s = torch.nn.functional.interpolate(pitch, scale_factor=frame_shift, mode='linear')
        tmp = torch.cumsum(pitch_s / sampling_rate, dim = -1)

        if not onnx_export:
            phase = tmp * 2 * math.pi + (torch.rand(pitch_s.shape[0], pitch_s.shape[1], 1).to(pitch.device) * 2 - 1) * math.pi
        else:
            phase = tmp * 2 * math.pi
        excitation = torch.sin(phase)
        unvoiced_mask = (pitch_s <= 20).type(torch.float32)
        unvoice_noise = (1 - unvoiced_mask) * 0.003 + unvoiced_mask * 0.1 / 3
        unvoice_noise = unvoice_noise * torch.randn_like(excitation)
        unvoice_noise = noise_index * unvoice_noise

        norm = excitation * (1 - unvoiced_mask) + unvoice_noise
        return norm
    elif mode == 'gt':
        return pitch
