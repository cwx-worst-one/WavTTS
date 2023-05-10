'''
test fftconvolve
'''
import numpy as np
import torch
from scipy.signal import fftconvolve as fftconv
from core.utils import get_local_rank
from core.dataset.preprocess.se.fftconvolve import fftconvolve


def gen_random_data(mic_num, length, rir_length):
    '''gen random data'''
    audio_data = np.random.randn(mic_num, length)
    rir_data = np.random.randn(mic_num, rir_length)
    return audio_data, rir_data


def get_random_batch(bsz=3, mic_num=2, length=12000, rir_length=2580):
    '''
    get random batch data with same length
    '''
    audio_datas = []
    rir_datas = []
    for _ in range(bsz):
        audio_data, rir_data = gen_random_data(mic_num, length, rir_length)
        audio_datas.append(audio_data)
        rir_datas.append(rir_data)
    audio_tensor = torch.from_numpy(np.stack(audio_datas, axis=0))
    rir_tensor = torch.from_numpy(np.stack(rir_datas, axis=0))
    return audio_datas, rir_datas, audio_tensor, rir_tensor


def get_collate_batch(bsz=3, mic_num=2, min_length=11000, max_length=12000, rir_length=2580):
    '''get collate batch with padding 0'''
    audio_datas = []
    rir_datas = []
    lengths = np.random.randint(min_length, max_length, bsz)

    max_length = max(lengths)
    audio_tensor = torch.zeros(bsz, mic_num, max_length)
    for bid in range(bsz):
        audio_data, rir_data = gen_random_data(mic_num, lengths[bid], rir_length)
        audio_datas.append(audio_data)
        rir_datas.append(rir_data)
        audio_tensor[bid][..., : audio_data.shape[-1]] = torch.from_numpy(audio_data)
    rir_tensor = torch.from_numpy(np.stack(rir_datas, axis=0))
    return audio_datas, rir_datas, audio_tensor, rir_tensor


def test_fftconvolve_no_padding(bsz=3, mic_num=2, length=12000, rir_length=2580):
    '''
    test fftconvolve without padding 0
    '''
    audio_datas, rir_datas, audio_tensor, rir_tensor = get_random_batch(
        bsz, mic_num, length, rir_length
    )
    base_out = []
    for bid in range(bsz):
        temp_list = []
        for mic in range(mic_num):
            temp = fftconv(audio_datas[bid][mic], rir_datas[bid][mic])[:length]
            temp_list.append(temp)
        base_out.append(np.stack(temp_list, axis=0))
    base_out = np.stack(base_out, axis=0)
    new_out = fftconvolve(
        audio_tensor.cuda(get_local_rank()), rir_tensor.cuda(get_local_rank()), axes=-1
    )[..., :length]
    base_out = torch.from_numpy(base_out)
    new_out = new_out.cpu()
    assert torch.allclose(base_out, new_out)


def test_fftconvolve_with_padding(
    bsz=3, mic_num=2, min_length=12000, max_length=13000, rir_length=2580
):
    '''
    test fftconvolve with padding 0
    '''
    audio_datas, rir_datas, audio_tensor, rir_tensor = get_collate_batch(
        bsz, mic_num, min_length, max_length, rir_length
    )
    base_out = []
    max_length = max(item.shape[-1] for item in audio_datas)
    base_out = torch.zeros(bsz, mic_num, max_length).double()
    for bid in range(bsz):
        temp_list = []
        length = audio_datas[bid].shape[-1]
        for mic in range(mic_num):
            temp = fftconv(audio_datas[bid][mic], rir_datas[bid][mic])[:length]
            temp_list.append(temp)
        base_out[bid][..., :length] = torch.from_numpy(np.stack(temp_list, axis=0))
    new_out = fftconvolve(
        audio_tensor.cuda(get_local_rank()), rir_tensor.cuda(get_local_rank()), axes=-1
    )[..., :max_length]
    for bid in range(bsz):  # do mask
        length = audio_datas[bid].shape[-1]
        new_out[bid, :, length:] = 0
    new_out = new_out.cpu()
    assert torch.allclose(base_out, new_out, atol=5e-5)
