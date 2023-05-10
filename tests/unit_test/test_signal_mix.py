"""
test mix func in simulator module
"""
import torch
import numpy as np


def mix(speech, noise, snr, frame_size=160, min_limit=1e-5):
    """Mix noise into speech.

    Args:
        sppech: speech to be mixed.
        noise: noise to be mixed.
        snr_min: minimum value of snr/sir.
        snr_max: maximum value of snr/sir.

    Return:
        noise: scaled noise.
        noisy: product after mixing.
        snr: snr
    """
    bsz = speech.shape[0]
    speech_t = speech[:, 0, ...]
    noise_t = noise[:, 0, ...]
    max_length = speech.shape[-1]
    n_frames = int(max_length / frame_size)
    speech_framed = speech_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)
    noise_framed = noise_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)
    speech_power = torch.max(torch.mean((speech_framed**2), dim=-1), dim=-1).values
    noise_power = torch.max(torch.mean((noise_framed**2), dim=-1), dim=-1).values
    noise_scale = torch.sqrt(
        speech_power / (noise_power + min_limit) / torch.pow(10.0, snr / 10.0)
    ).reshape(-1, 1, 1)
    noise = noise * noise_scale
    noisy = speech + noise
    return noise, noisy, snr


def base_mix(speech, noise, snr, frame_size=160, min_limit=1e-5):
    """Mix noise into speech.

    Args:
        sppech: speech to be mixed.
        noise: noise to be mixed.
        snr_min: minimum value of snr/sir.
        snr_max: maximum value of snr/sir.

    Return:
        noise: scaled noise.
        noisy: product after mixing.
        snr: snr
    """
    speech = speech.astype(np.float32)
    noise = noise.astype(np.float32)
    speech_t = speech[0]
    noise_t = noise[0]
    max_length = speech.shape[1]
    n_frames = int(max_length / frame_size)
    speech_framed = speech_t[: n_frames * frame_size].reshape(n_frames, frame_size)
    noise_framed = noise_t[: n_frames * frame_size].reshape(n_frames, frame_size)
    speech_power = np.max(np.mean((speech_framed**2), axis=1), axis=0)
    noise_power = np.max(np.mean((noise_framed**2), axis=1), axis=0)
    noise_scale = np.sqrt(speech_power / (noise_power + min_limit) / np.power(10.0, snr / 10.0))
    noise = noise * noise_scale
    noisy = speech + noise
    return noise, noisy, snr


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


def test_mix_with_padding(
    bsz=3, mic_num=2, min_length=12300, max_length=12800, snr_min=0, snr_max=15
):
    '''test mix with padding'''
    speech_datas, speech_tensor, noise_datas, noise_tensor = gen_fake_audio(
        bsz, mic_num, min_length, max_length
    )
    max_length = speech_tensor.shape[-1]

    base_noise_out = torch.zeros(bsz, mic_num, max_length)
    base_speech_out = torch.zeros(bsz, mic_num, max_length)

    snr_tensor = []

    for bid in range(bsz):
        snr = np.random.uniform(snr_min, snr_max)
        snr_tensor.append(snr)
        noise, speech, _ = base_mix(speech_datas[bid], noise_datas[bid], snr)
        base_noise_out[bid, :, : noise.shape[-1]] = torch.from_numpy(noise)
        base_speech_out[bid, :, : speech.shape[-1]] = torch.from_numpy(speech)

    snr = torch.tensor(snr_tensor)
    snr = snr.cuda()
    speech_tensor = speech_tensor.cuda()
    noise_tensor = noise_tensor.cuda()

    new_noise_out, new_speech_out, _ = mix(speech_tensor, noise_tensor, snr)

    new_noise_out = new_noise_out.cpu()
    new_speech_out = new_speech_out.cpu()

    assert torch.allclose(base_noise_out, new_noise_out, atol=5e-7)
    assert torch.allclose(base_speech_out, new_speech_out, atol=5e-7)


def test_mix_without_padding(bsz=3, mic_num=2, length=12800, snr_min=0, snr_max=15):
    '''test mix with padding'''
    speech_datas, speech_tensor, noise_datas, noise_tensor = gen_fake_audio(
        bsz, mic_num, length, length + 1
    )
    max_length = speech_tensor.shape[-1]

    base_noise_out = torch.zeros(bsz, mic_num, max_length)
    base_speech_out = torch.zeros(bsz, mic_num, max_length)

    snr_tensor = []

    for bid in range(bsz):
        snr = np.random.uniform(snr_min, snr_max)
        snr_tensor.append(snr)
        noise, speech, _ = base_mix(speech_datas[bid], noise_datas[bid], snr)
        base_noise_out[bid, :, : noise.shape[-1]] = torch.from_numpy(noise)
        base_speech_out[bid, :, : speech.shape[-1]] = torch.from_numpy(speech)

    snr = torch.tensor(snr_tensor)
    snr = snr.cuda()
    speech_tensor = speech_tensor.cuda()
    noise_tensor = noise_tensor.cuda()

    new_noise_out, new_speech_out, _ = mix(speech_tensor, noise_tensor, snr)

    new_noise_out = new_noise_out.cpu()
    new_speech_out = new_speech_out.cpu()

    assert torch.allclose(base_noise_out, new_noise_out, atol=5e-7)
    assert torch.allclose(base_speech_out, new_speech_out, atol=5e-7)


def test_mix_with_zero_tensor(bsz=3, mic_num=2, length=12800, snr=-1):
    '''test mix with zero tensor'''

    _, speech_tensor, _, _ = gen_fake_audio(bsz, mic_num, length, length + 1)
    max_length = speech_tensor.shape[-1]

    base_noise_out = torch.zeros(bsz, mic_num, max_length)
    base_speech_out = speech_tensor.clone()

    snr_tensor = [snr] * bsz
    snr = torch.tensor(snr_tensor)
    snr = snr.cuda()
    speech_tensor = speech_tensor.cuda()
    noise_tensor = base_noise_out.cuda()

    new_noise_out, new_speech_out, _ = mix(speech_tensor, noise_tensor, snr)
    new_noise_out = new_noise_out.cpu()
    new_speech_out = new_speech_out.cpu()

    assert torch.allclose(base_noise_out, new_noise_out)
    assert torch.allclose(base_speech_out, new_speech_out)
