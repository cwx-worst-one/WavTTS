'''
Unit test of some aec simulator ops
'''

import random
import numpy as np
import torch


def base_add_noise(speech, noise, snrs, speech_flag=True, frame_max=False, frame_size=160):
    '''base add noise func'''
    if speech_flag > 0:
        speech_center = speech - np.mean(speech)
        noise_center = noise - np.mean(noise)
        _max_length = speech.shape[0]
        # snrs = snr_min + (snr_max-snr_min) * random.random()
        if frame_max:
            num_frames = speech.shape[-1] // frame_size
            speech_center_frame = speech_center[: num_frames * frame_size].reshape(
                num_frames, frame_size
            )
            noise_center_frame = noise_center[: num_frames * frame_size].reshape(
                num_frames, frame_size
            )
            speech_power = np.max((speech_center_frame**2).sum(-1))
            noise_power = np.max((noise_center_frame**2).sum(-1))
        else:
            speech_power = (speech_center**2).mean()
            noise_power = (noise_center**2).mean()
        noise_scale = np.sqrt(speech_power / (noise_power + 1e-5) / np.power(10.0, snrs / 10.0))
        noise = noise * noise_scale
    else:
        noise = noise * (random.random() * 0.8 + 0.1)
    return noise


def new_add_noise(speech, noise, snr, speech_flag=True, frame_max=False, frame_size=160):
    '''new add noise func'''
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
        speech_power / (noise_power + 1e-5) / torch.pow(10.0, snr / 10.0)
    ).reshape(-1, 1, 1)
    noise = noise * noise_scale
    return noise


def gen_fake_audio(bsz, mic_num, min_length, max_length):
    '''gen fake batch audio datas'''
    lengths = np.random.randint(min_length, max_length, bsz)
    audio_datas = []
    noise_datas = []
    max_length = max(lengths)
    audio_tensor = torch.zeros(bsz, mic_num, max_length)
    noise_tensor = torch.zeros(bsz, mic_num, max_length)
    for bid in range(bsz):
        length = lengths[bid]
        audio = np.random.randn(mic_num, length)
        noise = np.random.randn(mic_num, length)
        audio_datas.append(audio)
        noise_datas.append(noise)
        audio_tensor[bid, :, :length] = torch.from_numpy(audio)
        noise_tensor[bid, :, :length] = torch.from_numpy(noise)
    return audio_datas, audio_tensor, noise_datas, noise_tensor


def test_add_noise(bsz=3, mic_num=1, length=128000, snr_min=15, snr_max=30):
    '''test add noise'''
    speech_datas, speech_tensor, noise_datas, noise_tensor = gen_fake_audio(
        bsz, mic_num, length, length + 1
    )
    max_length = speech_tensor.shape[-1]

    base_speech_out0 = torch.zeros(bsz, max_length)
    base_speech_out1 = torch.zeros(bsz, max_length)

    snr_tensor = []

    for bid in range(bsz):
        snr = np.random.uniform(snr_min, snr_max)
        snr_tensor.append(snr)
        speech = base_add_noise(speech_datas[bid][0], noise_datas[bid][0], snr, frame_max=False)
        base_speech_out0[bid, : speech.shape[-1]] = torch.from_numpy(speech)
        speech = base_add_noise(speech_datas[bid][0], noise_datas[bid][0], snr, frame_max=True)
        base_speech_out1[bid, : speech.shape[-1]] = torch.from_numpy(speech)

    snr = torch.tensor(snr_tensor)
    snr = snr.cuda()
    speech_tensor = speech_tensor.cuda()
    noise_tensor = noise_tensor.cuda()

    new_speech_out0 = new_add_noise(speech_tensor, noise_tensor, snr, frame_max=False)
    new_speech_out1 = new_add_noise(speech_tensor, noise_tensor, snr, frame_max=True)

    new_speech_out0 = new_speech_out0.cpu()
    new_speech_out1 = new_speech_out1.cpu()
    assert torch.allclose(base_speech_out0, new_speech_out0[:, 0, :], atol=5e-7)
    assert torch.allclose(base_speech_out1, new_speech_out1[:, 0, :], atol=5e-7)


def base_agc_fn(mic, agc):
    '''base agc fn'''
    ratio = 1 / (np.max(np.abs(mic)) + 1e-3) * agc
    mic *= ratio
    return mic


def rand_agc_ratio(data):
    '''generate random agc ratio
    used in aec task
    '''
    gen_cfg = dict(
        agc_min=1000,
        agc_max=32767,
        min_limit=1e-5,
    )
    bsz = data.shape[0]
    random_data = torch.rand(bsz, device=data.device)
    agc_value = random_data * (gen_cfg['agc_max'] - gen_cfg['agc_min']) + gen_cfg['agc_min']
    agc_ratio = (
        1 / (torch.max(torch.max(torch.abs(data), dim=-1).values, dim=-1).values + 1e-3) * agc_value
    )
    agc_ratio = agc_ratio.reshape(-1, 1, 1)
    data *= agc_ratio
    return data, agc_value


def test_agc_fn(bsz=3, length=128000):
    '''new agc dn'''
    speech_datas, speech_tensor, _, _ = gen_fake_audio(bsz, 1, length, length + 1)
    new_speech_out, agc_value = rand_agc_ratio(speech_tensor.cuda())
    base_speech_out = []

    for idx, speech in enumerate(speech_datas):
        base_speech_out.append(base_agc_fn(speech[0], agc_value[idx].item()))

    base_speech_out = torch.from_numpy(np.stack(base_speech_out, axis=0))
    new_speech_out = new_speech_out.cpu()
    assert torch.allclose(base_speech_out, new_speech_out[:, 0, :].double(), atol=5e-7)
