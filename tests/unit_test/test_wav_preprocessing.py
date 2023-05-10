''' test wav preprocessing. '''

import io
import os
import pickle
import time
from scipy.io import wavfile
from subword_nmt.apply_bpe import BPE as ApplyBPE
from dataloader import FalconReader
from core.dataset import get_meta
from core.dataset.preprocess import (
    WavParser,
    AddNoise,
    ModifyProsody,
    VolumePerturbation,
    SpeedPerturbation,
    KaldiFbank,
    AppendFrames,
    PunctuationFilter,
    BPE,
)


def _create_temp_output_dir():
    '''create a temp dir for wavs saving.'''
    unit_tests_basedir = './tests'
    if not os.path.exists(unit_tests_basedir):
        os.mkdir(unit_tests_basedir)
    temp_wavs_dir = os.path.join(unit_tests_basedir, 'temp_wavs')
    if not os.path.exists(temp_wavs_dir):
        os.mkdir(temp_wavs_dir)
    return temp_wavs_dir


def test_wav_preprocessing():
    '''test function for wav preprocessing'''
    meta_file = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/resources/zh_meta/meta'
    meta_data = get_meta(meta_file)
    reorder_tgt_dict = meta_data['reorder_tgt_dict']
    total_code = meta_data['total.code']
    bpe_fn = ApplyBPE(io.StringIO(total_code))

    # pylint: disable=line-too-long
    reader = FalconReader(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liangzhenlin/dolphin/data_wav/voice_input_third_2_27000h/sub_999',
        10,
    )
    _keys = reader.list_keys()
    wav_parser = WavParser()
    prosody_transform = ModifyProsody(speed_factor=1.5, tempo_factor=0.65, adjust_prob=1.0)
    noise_transform1 = AddNoise(p=0.3, min_snr=0, max_snr=5)
    noise_transform2 = SpeedPerturbation(p=0.3)
    noise_transform3 = VolumePerturbation(p=0.3, scale_low=0.01, scale_high=0.1)
    kaldi_fbank = KaldiFbank(in_key='waveform', out_key='fbank')
    feat_padding = AppendFrames(frame=12, key='fbank', value=-15.0, seg_frames=20)
    punct_filter = PunctuationFilter(key='label')
    bpe = BPE(bpe_fn=bpe_fn, reorder_tgt_dict=reorder_tgt_dict, use_eos=False)
    temp_wavs_dir = _create_temp_output_dir()
    vals = reader.read_many([0], True)[0]
    for val in vals:
        item_data = pickle.loads(val)
        ori_wav_path = os.path.join(temp_wavs_dir, 'ori_' + item_data['uttid'] + '.wav')
        with open(ori_wav_path, 'wb') as fout:
            fout.write(item_data['wav'])
        item_data = wav_parser(item_data)
        print('raw label:\n', item_data['label'])
        item_data = punct_filter(item_data)
        print('label without punctuation:\n', item_data['label'])
        item_data = bpe(item_data)
        print('bpe char indices:\n', item_data['char'])
        print(item_data['waveform'][0, 0:10])
        st = time.time()
        item_data = prosody_transform(item_data)
        print(item_data['waveform'][0, 0:10])
        print('time elapsed for prosody modification: {}'.format(time.time() - st))
        print('--------')
        st = time.time()
        item_data = noise_transform1(item_data)
        item_data = noise_transform2(item_data)
        item_data = noise_transform3(item_data)
        print(item_data['waveform'][0, 0:10])
        print('time elapsed for adding noise: {}'.format(time.time() - st))
        item_data = kaldi_fbank(item_data)
        print('feat size:', item_data['fbank'].shape)
        item_data = feat_padding(item_data)
        print('feat size after padding:', item_data['fbank'].shape)
        print('last 40 frames of feat:', item_data['fbank'][-40:])
        noise_wav_path = os.path.join(temp_wavs_dir, 'noise_' + item_data['uttid'] + '.wav')
        wavfile.write(
            noise_wav_path, item_data['sample_rate'], item_data['waveform'].astype('int16').T
        )
    print("all wavefiles are saved to " + temp_wavs_dir)
