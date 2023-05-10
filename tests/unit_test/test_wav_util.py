"""
Verify the correctness of get_frames_length in wav_utils
"""
import pickle
from dataloader import FalconReader
from core.dataset.preprocess.wav import WavParser
from core.dataset.preprocess.fbank import KaldiFbank
from core.utils import get_frames_len


def test_wav_util(
    path='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
    'datasets/dolphin/librispeech_wav/train_sub0',
):
    """
    Verify the correctness of get_frames_length in wav_utils
    """
    reader = FalconReader(path, 10)
    _keys = reader.list_keys()
    wav_parser = WavParser()
    kaldifbank = KaldiFbank()
    datas = reader.read_many([0], True)[0]
    for data in datas:
        item = pickle.loads(data)
        item = wav_parser(item)
        waveform = item['waveform']
        item = kaldifbank(item)
        assert get_frames_len(waveform.shape[1]) == item['fbank'].shape[0]
