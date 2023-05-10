''' test for fbank. '''
import random
import pickle
import io
from scipy.io import wavfile
import numpy as np
import torch
from packaging import version
from dataloader import FalconReader
from torchaudio.compliance.kaldi import fbank as org_fbank
from core.dataset.preprocess.kaldi_fbank import SpeedFbank
from core.dataset.preprocess.fbank import SpeedKaldiFbank


def _get_waveform(num=1):
    '''get wavforms.'''
    path = (
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
        'datasets/dolphin/librispeech_wav/train_sub2'
    )
    reader = FalconReader(path, 10)
    _keys = reader.list_keys()
    chunk_num = (num + 10 - 1) // 10
    vals = reader.read_many(list(range(chunk_num)), True)
    vals = sum(vals, [])[:num]
    waveforms = []
    for v in vals:
        wav = pickle.loads(v)['wav']
        sample_rate, waveform = wavfile.read(io.BytesIO(wav))
        waveform = waveform.reshape(1, -1).astype(np.float32)
        waveforms.append([waveform, sample_rate])
    return waveforms


def _batch_waveform(waveforms):
    '''batch waveform.'''
    bsz = len(waveforms)
    sample_rate = waveforms[0][1]
    window_shift = sample_rate // 100
    # wavform shape (1, -1)
    max_samples = max(waveform.shape[1] for waveform, _ in waveforms)

    wav_batch = torch.zeros([bsz, max_samples], dtype=torch.float32)
    for i, (wav, _) in enumerate(waveforms):
        wav = wav.reshape(-1)
        samples = wav.shape[0]
        pad_frames = 0
        pad_samples = pad_frames * window_shift
        wav_batch[i, pad_samples : pad_samples + samples] = torch.from_numpy(wav)
    return wav_batch


def test_speed_torchaudio_fbank():
    '''test'''
    seed = 150
    for waveform, sample_rate in _get_waveform():
        torch.manual_seed(seed)
        fbank1 = org_fbank(
            torch.Tensor(waveform),
            sample_frequency=sample_rate,
            num_mel_bins=80,
            use_energy=False,
            dither=1.0,
        )
        torch.manual_seed(seed)
        fn = SpeedFbank('cpu', num_mel_bins=80, sample_frequency=sample_rate)
        fbank2 = fn(torch.Tensor(waveform), dither=1.0)
        if version.parse(torch.__version__) >= version.parse('1.7.0'):
            assert torch.allclose(fbank1, fbank2, atol=1e-10)
        else:
            # before torch 1.7.1
            # sum with mel fiterbanks over the power spectrum is implement as mul and sum,
            # instead of torch.mm
            assert torch.allclose(fbank1, fbank2, atol=1e-5)


def test_batch_speed_fbank():
    '''test'''
    waveforms = _get_waveform(2)
    sample_rate = waveforms[0][1]
    waveform_batch = _batch_waveform(waveforms)
    fn = SpeedFbank('cpu', num_mel_bins=80, sample_frequency=sample_rate)
    fbank2 = fn(torch.Tensor(waveform_batch))

    for i, (waveform, sample_rate) in enumerate(waveforms):
        fbank1 = org_fbank(
            torch.Tensor(waveform),
            sample_frequency=sample_rate,
            num_mel_bins=80,
            use_energy=False,
            dither=0.0,
        )
        frame_len = fbank1.shape[0]
        if version.parse(torch.__version__) >= version.parse('1.7.0'):
            assert torch.allclose(fbank1, fbank2[i, :frame_len, :], atol=1e-10)
        else:
            # before torch 1.7.1
            # sum with mel fiterbanks over the power spectrum is implement as mul and sum,
            # instead of torch.mm
            assert torch.allclose(fbank1, fbank2[i, :frame_len, :], atol=1e-5)


def test_speed_fbank_all_shapes():
    """
    test diffrent inputs
    """
    num_channels = random.randint(2, 10)
    num_samples = random.randint(10000, 50000)
    batch_size = random.randint(1, 20)
    item = dict()
    wav_fn = SpeedKaldiFbank(
        in_key='waveform',
        out_key='fbank',
        mask_key='src_mask',
        use_energy=False,
        dither=1.0,
        dynamic_dither=False,
        device='cpu',
        out_numpy=True,
        fbank_dim=None,
    )
    batch_fn = SpeedKaldiFbank(
        in_key='waveform',
        out_key='fbank',
        mask_key='src_mask',
        use_energy=False,
        dither=1.0,
        dynamic_dither=False,
        device='cpu',
        out_numpy=False,
        fbank_dim=None,
    )

    # non batch test 1: 1 * num_samples
    waveform = np.random.rand(1, num_samples)
    item['waveform'] = waveform
    out = wav_fn(item)
    assert out['fbank'].ndim == 2

    # non batch test 2: channels * num_samples
    waveform = np.random.rand(num_channels, num_samples)
    item['waveform'] = waveform
    out = wav_fn(item)
    assert out['fbank'].ndim == 3 and out['fbank'].shape[0] == num_channels

    # batch test 1: bsz * num_samples
    waveform = np.random.rand(batch_size, num_samples)
    item['waveform'] = waveform
    out = batch_fn(item)
    assert out['fbank'].ndim == 3 and out['fbank'].shape[0] == batch_size

    # batch test 2: bsz * num_channels * num_samples
    waveform = np.random.rand(batch_size, num_channels, num_samples)
    item['waveform'] = waveform
    out = batch_fn(item)
    assert (
        out['fbank'].ndim == 4
        and out['fbank'].shape[0] == batch_size
        and out['fbank'].shape[1] == num_channels
    )

    # batch test 3: bsz * 1 * num_samples
    waveform = np.random.rand(batch_size, 1, num_samples)
    item['waveform'] = waveform
    out = batch_fn(item)
    assert out['fbank'].ndim == 3 and out['fbank'].shape[0] == batch_size
