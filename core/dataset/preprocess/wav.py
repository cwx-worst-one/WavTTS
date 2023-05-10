'''
waveform processing
'''
# pylint:disable=too-many-lines

import warnings
import os
import io
import random
import pickle
from collections import deque, defaultdict
import math
import csv
from packaging.version import Version
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import librosa
import soxbindings as sox
import torchaudio
import torch
from dataloader import FalconReader
from core.extensions import add_noise as add_noise_c
from core.utils import logging, get_frames_len, get_wav_len, wav_start_pos
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import dist_file_get
from .preprocess import PREPROCESS

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


MILLISECONDS_TO_SECONDS = 0.001
FRAMES_PER_SECOND = 100


@PREPROCESS.register_module()
class WavConvert:
    '''Convert waveform from other binary data'''

    def __init__(
        self, in_key='frames', out_key='waveform', bits=16, sample_rate=16000, skip_prob=0.0
    ):
        '''init.
        Args:
            in_key: which key contain convert binary data
            out_key: out key after wavconvert
        '''
        self.in_key = in_key
        self.out_key = out_key
        self.bits = bits
        self.sample_rate = sample_rate
        self.skip_prob = skip_prob

    def __call__(self, item, **_kwargs):
        '''extract waveform and sample_rate'''
        if item is None or self.in_key not in item:
            return item
        if self.skip_prob > 0 and random.uniform(0, 1) < self.skip_prob:
            # skip parser waveform for TOG
            return item
        raw_data = item.pop(self.in_key)
        waveform = raw_data.reshape(1, -1).astype(np.float32)
        item[self.out_key] = waveform
        item['bits'] = item.get('bits', self.bits)
        item['sample_rate'] = item.get('sample_rate', self.sample_rate)
        if 'augmentation' not in item:
            item['augmentation'] = []
        elif not isinstance(item['augmentation'], list):
            item['augmentation'] = []
        return item


@PREPROCESS.register_module()
class WavParser:
    '''extract waveform from wav binary data'''

    def __init__(self, in_key='wav', min_len=0.05, max_len=100, skip_prob=0.0):
        '''init
        Args:
            in_key: which key contain wave binary data
            min_len: the minimum value of audio length
            max_len: the maximum value of audio length
        '''
        self.in_key = in_key
        self.out_key = in_key.replace('wav', 'waveform')
        self.min_len = min_len
        self.max_len = max_len
        self.skip_prob = skip_prob

    def __call__(self, item, **_kwargs):
        '''extract waveform and sample_rate'''
        if item is None or self.in_key not in item:
            return item
        if self.skip_prob > 0 and random.uniform(0, 1) < self.skip_prob:
            # skip parser waveform for TOG
            return item
        waveform, sample_rate = sf.read(io.BytesIO(item[self.in_key]), dtype='int16')
        channel_num = 1 if waveform.ndim == 1 else waveform.shape[1]
        if self.min_len < waveform.shape[0] / float(sample_rate) < self.max_len:
            item['bits'] = np.iinfo(waveform.dtype).bits
            item['sample_rate'] = sample_rate
            item['channel_num'] = channel_num
            if 'augmentation' not in item:
                item['augmentation'] = []
            elif not isinstance(item['augmentation'], list):
                item['augmentation'] = []

            item[self.out_key] = waveform.reshape(channel_num, -1).astype(np.float32)
            return item
        return None


@PREPROCESS.register_module()
class SimpleWavParser:
    '''extract waveform from wav binary data'''

    def __init__(self, in_key='wav', out_key=None, out_dtype='np.float32'):
        '''init
        Args:
            in_key: which key contain wave binary data
            min_len: the minimum value of audio length
            max_len: the maximum value of audio length
        '''
        self.in_key = in_key
        if out_key is None:
            self.out_key = self.in_key
        else:
            self.out_key = out_key
        self.out_dtype = eval(out_dtype)

    def __call__(self, item, **_kwargs):
        '''extract waveform and sample_rate'''
        if item is None or self.in_key not in item:
            return item
        item[self.out_key] = item[self.in_key].astype(self.out_dtype)
        return item


@PREPROCESS.register_module()
class WavNorm:
    '''norm wav to (-1, 1) and Z-score standardization'''

    def __init__(self, bits_shift=15, key="waveform", zscore=False):
        '''init'''
        self.bits_shift = bits_shift
        self.key = key
        self.zscore_norm = zscore

    def __call__(self, item, **_kwargs):
        '''call'''
        if item is None or self.key not in item:
            return None
        bits_shift = self.bits_shift
        if "bits" in item:
            bits_shift = item["bits"] - 1
        item[self.key] /= 1 << bits_shift
        if self.zscore_norm:
            item[self.key] = (item[self.key] - np.mean(item[self.key])) / np.sqrt(
                np.var(item[self.key]) + 1e-5
            )
        return item


@PREPROCESS.register_module()
class AppendSilence:
    '''append silence to audio tail'''

    def __init__(
        self,
        key='waveform',
        dur=0.12,
        sil_key='wav',
        sil_shards=256,
        sil_chunk=200,
        sil_dir=None,
        sil_prefix=None,
        resample_backend='sox',
        skip_enough=False,
        sil_snr_max=-1,
        sil_snr_min=-1,
        is_eos_sil_frame=False,
        chunk_size=20,
        prefetch_chunk_num=20,
    ):
        '''init
        Args:
            key: (string) waveform key in item(dict)
            dur: (float) append duration in seconds
            sil_key: (string) wave binary key in item(dict)
            sil_shards: (int) number of shards of silence dataset
            sil_chunk: unused args, it will be abandoned later
            sil_dir: (string) hdfs root path of silence dataset
            sil_prefix: (string) prefix of file list
            resample_backend: (string) the method for silence resample, sox or librosa
            chunk_size: (int) chunk size when reading silence dataset
            prefetch_chunk_num: (int) chunk read num when reading silence dataset
            skip_enough: (bool) skip appending if there is enough silence
            sil_snr_max: (float) max SNR between appended silence and original audio
                if set to -1, we will use the original silence without scaling
            sil_snr_min: (float) min SNR between appended silence and original audio
            is_eos_sil_frame: (bool) whether if item["eos"] is the number of frames in audio tail
                default is False, which means item["eos"] is the end frame of last non-silence token
        '''
        self.key = key
        self.dur = dur
        self.sil_key = sil_key
        self.sil_out_key = sil_key.replace('wav', 'waveform')
        self.sil_shards = sil_shards
        self.sil_chunk = sil_chunk
        self.sil_dir = sil_dir
        self.sil_prefix = sil_prefix
        self.resample_backend = resample_backend
        self.sil_reader = None
        self.sil_array = None
        self.sil_rate = None
        self.sil_parser = WavParser(in_key=self.sil_key)
        self.chunk_size = chunk_size
        self.skip_enough = skip_enough
        self.sil_snr_max = sil_snr_max
        self.sil_snr_min = sil_snr_min
        self.is_eos_sil_frame = is_eos_sil_frame
        self.chunks_buf = deque()
        self.data_buf = deque()
        self.prefetch_chunk_num = prefetch_chunk_num

    def get_reader(self):
        '''
        get sil reader
        '''
        sil_list = [
            '{}/{}{}'.format(self.sil_dir, self.sil_prefix, shard)
            for shard in range(self.sil_shards)
        ]
        self.sil_reader = FalconReader(
            sil_list,
            256,  # fd_cache_size
            8,  # io_thread_num
            5,  # io_retry
            "AppendSilence",  # unused
            get_local_size(),  # GPU num in one worker
            get_local_rank(),  # GPU idx in one worker
            self.chunk_size,  # chunk_size
        )

    def update_buf(self):
        '''update data buf'''
        if not self.chunks_buf:
            if self.sil_reader is None:
                self.get_reader()
            shard = random.randint(0, self.sil_shards - 1)
            entry_nums = self.sil_reader.get_entry_num([shard], False)
            chunks_idxs = [i * self.chunk_size for i in range(entry_nums // self.chunk_size)]
            self.chunks_buf.extend(
                [
                    chunks_idxs[i : i + self.prefetch_chunk_num]
                    for i in range(0, len(chunks_idxs), self.prefetch_chunk_num)
                ]
            )
        chunks = self.chunks_buf.popleft()
        datas = sum(self.sil_reader.read_many(chunks), [])
        self.data_buf.extend(datas)

    def get_data(self):
        '''get one sil data'''
        if not self.data_buf:
            self.update_buf()
        item = self.data_buf.popleft()
        item = pickle.loads(item)
        item = self.sil_parser(item)
        return item

    def resample(self, sil_array, num_samples, data_rate, bits):
        '''resample silence data using different backend'''
        sample_factor = self.sil_rate / data_rate
        if self.resample_backend == "sox":
            tfm = sox.Transformer()
            tfm.rate(data_rate)
            data = sil_array[0][: 1 + int(num_samples * sample_factor)].reshape(-1, 1)
            # scale to [-1, 1]
            data /= 1 << bits
            tmp_sil = tfm.build_array(input_array=data, sample_rate_in=self.sil_rate)
            # scale back
            tmp_sil *= 1 << bits
            cur_sil = tmp_sil[:num_samples].reshape(1, num_samples)
        else:
            tmp_sil = librosa.core.resample(
                sil_array[0, : 1 + int(num_samples * sample_factor)],
                orig_sr=self.sil_rate,
                target_sr=self.sil_rate / sample_factor,
                res_type='kaiser_fast',
            )
            cur_sil = tmp_sil[:num_samples].reshape(1, num_samples)
        return cur_sil

    def update_sil(self, num_samples, data_rate, bits=15):
        '''make sure the silence is long enough'''
        sample_factor = self.sil_rate / data_rate
        while self.sil_array is None or self.sil_array.shape[1] <= num_samples * sample_factor:
            sil = self.get_data()
            if sil is None or self.sil_out_key not in sil:
                continue
            self.sil_rate = sil['sample_rate']
            if self.sil_array is None:
                self.sil_array = sil[self.sil_out_key]
            else:
                self.sil_array = np.concatenate((self.sil_array, sil[self.sil_out_key]), axis=1)

        if sample_factor != 1:
            cur_sil = self.resample(
                self.sil_array, bits=bits, num_samples=num_samples, data_rate=data_rate
            )
        else:
            cur_sil = self.sil_array[:, :num_samples]
        self.sil_array = self.sil_array[:, int(num_samples * sample_factor) :]
        return cur_sil

    def refine_eos(self, item):
        '''unify the item['eos'] to the end frame of last non-silence token'''
        sil_frame = 0
        audio_frame = int(item[self.key].shape[1] / item['sample_rate'] * FRAMES_PER_SECOND)
        if 'eos' in item and item['eos'] <= 0:
            # the audio has alignment but no silence in the tail,
            # set eos to the last frame of the audio
            item['eos'] = audio_frame
        elif 'eos' in item:
            # item['eos'] > 0, meaning there is silence in audio tail
            if self.is_eos_sil_frame:
                # item['eos'] is the number of silence frames in the tail
                sil_frame = item['eos']
                item['eos'] = audio_frame - sil_frame
            else:
                # item['eos'] is the end frame of last non-silence token
                sil_frame = audio_frame - item['eos']
        else:
            # the audio does not have alignment, set eos as 0 to avoid penalty
            item['eos'] = 0
        return sil_frame

    def apply_snr_to_silence(self, silence, audio_energy):
        '''scale silence according to a random snr'''
        ori_sil_energy = np.mean(silence**2) + 1e-8
        # snr = 10 * log10(audio_energy / sil_energy)
        # -> sil_energy = audio_energy / (10 ^ (snr / 10))
        snr = random.uniform(self.sil_snr_min, self.sil_snr_max)
        dst_sil_energy = audio_energy / np.power(10.0, snr / 10.0)
        silence *= np.sqrt(dst_sil_energy / ori_sil_energy)
        return silence

    def __call__(self, item, **_kwargs):
        '''call'''
        if item is None or self.key not in item:
            return item
        sil_frame = self.refine_eos(item)
        # skip appending for the audios with enough silence
        if self.skip_enough and sil_frame >= self.dur * FRAMES_PER_SECOND:
            return item
        num_samples = int(item['sample_rate'] * self.dur)
        if self.sil_dir is None:
            # append gaussian noise as silence
            silence = np.random.randn(1, num_samples)
        else:
            # append real silence with length of [sil_dur, dur]
            if self.sil_rate is None:
                self.sil_rate = item['sample_rate']
            bits = item.get("bits", 16) - 1
            silence = self.update_sil(num_samples, item['sample_rate'], bits=bits)
        if self.sil_snr_max > 0:
            audio_energy = np.mean(item[self.key] ** 2) + 1e-8
            silence = self.apply_snr_to_silence(silence, audio_energy)
        # convert float to int16 before concatenate
        silence = silence.astype(np.int16).astype(np.float32)
        item[self.key] = np.concatenate((item[self.key], silence), axis=1)
        return item


@PREPROCESS.register_module()
class AppendAudio:
    '''append audio to wav head or tail'''

    def __init__(
        self,
        key='waveform',
        audio_key='wav',
        audio_chunk=200,
        audio_dir=None,
        audio_prefix=None,
        resample_backend='sox',
        chunk_size=20,
        append_pos_list=('head', 'tail'),
        p=0.05,
        enable_limit=False,
        add_min_len=0,
        add_max_len=1e8,
    ):
        '''init
        Args:
            key: (string) waveform key in item(dict)
            audio_key: (string) wave binary key in item(dict)
            audio_shards: (int) number of shards of audio dataset
            audio_chunk: (int) how many audio item to fetch in once reading
            audio_dir: (string) hdfs root path of audio dataset
            audio_prefix: (string) prefix of file list
            resample_backend: (string) the method for audio resample, sox or librosa
            chunk_size: (int) chunk size when reading audio dataset
            append_pos_list: (list) append pos list
            p: prob of append audio
        '''
        self.key = key
        self.audio_key = audio_key
        self.audio_out_key = audio_key.replace('wav', 'waveform')
        self.audio_chunk = audio_chunk
        self.audio_dir = audio_dir
        self.audio_prefix = audio_prefix
        self.resample_backend = resample_backend
        self.audio_chunks = []
        self.audio_reader = None
        self.audio_array = []
        self.audio_rate = None
        self.values = []
        self.audio_parser = WavParser(in_key=self.audio_key)
        self.chunk_size = chunk_size
        self.append_pos_list = append_pos_list
        self.prob = p
        self.enable_limit = enable_limit
        self.add_min_len = add_min_len
        self.add_max_len = add_max_len

    def initialize(self):
        '''
        get audio reader
        '''
        audio_list = ['{}/{}'.format(self.audio_dir, self.audio_prefix)]
        self.audio_reader = FalconReader(
            audio_list,
            256,  # fd_cache_size
            8,  # io_thread_num
            5,  # io_retry
            "AppendAudio",  # unused
            get_local_size(),  # GPU num in one worker
            get_local_rank(),  # GPU idx in one worker
            self.chunk_size,  # chunk_size
        )

    def read_random_keys(self):
        '''read from another audio shard'''
        if self.audio_reader is None:
            self.initialize()
        shard = 0
        self.entry_nums = self.audio_reader.get_entry_num([shard], False)
        self.audio_chunks = [i * self.chunk_size for i in range(self.entry_nums // self.chunk_size)]

    def update_audio(self):
        '''make sure the audio is long enough'''
        while len(self.audio_array) < 1:
            if not self.audio_chunks:
                self.read_random_keys()
            num = min(self.audio_chunk // self.chunk_size, len(self.audio_chunks))
            if not self.values:
                self.values = sum(self.audio_reader.read_many(self.audio_chunks[0:num], True), [])
                del self.audio_chunks[0:num]
            for v in self.values:
                audio = self.audio_parser(pickle.loads(v))
                if audio is None:
                    continue
                self.audio_rate = audio['sample_rate']
                label = audio['label']
                if self.enable_limit:
                    if (
                        audio[self.audio_out_key].shape[1] < self.audio_rate * self.add_min_len
                        or audio[self.audio_out_key].shape[1] > self.audio_rate * self.add_max_len
                    ):
                        continue
                self.audio_array.append((audio[self.audio_out_key], label))
        self.values = []
        cur_audio = self.audio_array.pop()
        return cur_audio

    def __call__(self, item, **_kwargs):
        '''call'''
        if item is None or self.key not in item or random.uniform(0, 1) > self.prob:
            return item
        audio, label = self.update_audio()
        # convert float to int16 before concatenate
        audio = audio.astype(np.int16).astype(np.float32)
        append_pos = random.choice(self.append_pos_list)
        if append_pos == 'head':
            item[self.key] = np.concatenate((audio, item[self.key]), axis=1)
            item['label'] = label + item['label']
        elif append_pos == 'tail':
            item[self.key] = np.concatenate((item[self.key], audio), axis=1)
            item['label'] = item['label'] + label
        return item


@PREPROCESS.register_module()
class CalculateFrameLength:
    '''get length of frame num.'''

    def __init__(
        self,
        in_key='waveform',
        sr_key='sample_rate',
        out_key='length',
        channel_key='channel_num',
        frame_length=25,
        frame_shift=10,
    ):
        '''init.'''
        self.in_key = in_key
        self.sr_key = sr_key
        self.out_key = out_key
        self.channel_key = channel_key
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, item, **_kwargs):
        '''get frame length before get fbank'''
        if item is None or self.in_key not in item:
            return None

        waveform = item[self.in_key]
        sample_rate = item[self.sr_key]
        channel_num = 1
        if item.__contains__(self.channel_key):
            channel_num = item[self.channel_key]
        window_size = int(sample_rate * self.frame_length * MILLISECONDS_TO_SECONDS)
        de_shift = int(1.0 / (self.frame_shift * MILLISECONDS_TO_SECONDS))
        wav_len = waveform.shape[1] / channel_num
        item[self.out_key] = int((wav_len - window_size) * de_shift // float(sample_rate) + 1)
        return item


@PREPROCESS.register_module()
class VolumePerturbation:
    '''perturbs the volume of audios'''

    def __init__(self, key='waveform', p=0.5, scale_low=0.125, scale_high=2.0, skip_num=0):
        '''init'''
        self.key = key
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.prob = p
        self.skip_num = skip_num

    def __call__(self, item, **_kwargs):
        '''call'''
        if self.skip_num > 0:
            self.skip_num -= 1
            return item
        if item is None or self.key not in item or random.uniform(0, 1) > self.prob:
            return item
        scale = random.uniform(self.scale_low, self.scale_high)
        item[self.key] *= scale
        max_v = np.abs(item[self.key]).max()
        item['augmentation'].append(self.__class__.__name__)
        if max_v > 2**15 - 1:
            item[self.key] /= max_v
            item[self.key] *= 2**15 - 1
        return item


@PREPROCESS.register_module()
class SpeedPerturbation:
    '''http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
    sox API: https://pysox.readthedocs.io/en/latest/api.html
    '''

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        key='waveform',
        p=0.0,
        speed_rate_list=[0.95, 1.05],
        skip_num=0,
        use_both=False,
        cls_relabel=False,
        relabel_key='relabel',
        vad_key='vad',
    ):
        '''init
        p: the prob for SpeedPerturbation
        speed_rate_list: the speeds for perturbation
        '''
        assert isinstance(speed_rate_list, list), 'speed_rate_list must be a list'
        self.key = key
        self.prob = p
        self.transform_list = []
        for speed in speed_rate_list:
            try:
                speed = float(speed)
            except Exception:
                continue
            self.transform_list.append(self.get_transform(speed))
        self.skip_num = skip_num
        self.use_both = use_both
        self.cls_relabel = cls_relabel
        self.relabel_key = relabel_key
        self.speed_list = list(range(len(speed_rate_list)))
        self.vad_key = vad_key

    @staticmethod
    def get_transform(speed):
        '''get sox transform'''
        tfm = sox.Transformer()
        tfm.speed(speed)
        return tfm

    @staticmethod
    def scale_vad(vad, target_len):
        '''
        Args:
            vad: a 1d-array numpy
        '''
        new_vad = np.zeros(target_len)
        src_len = vad.shape[0]
        rat = src_len / float(target_len)
        for i, _ in enumerate(new_vad):
            new_vad[i] = vad[int(i * rat)]
        return new_vad

    def __call__(self, item, **_kwargs):
        '''call'''
        if self.skip_num > 0:
            self.skip_num -= 1
            return item
        if not self.transform_list:
            return item
        if item is None or self.key not in item or random.uniform(0, 1) > self.prob:
            return item
        if not self.use_both:
            if (
                'SpeedPerturbation' in item['augmentation']
                or 'TempoPerturbation' in item['augmentation']
            ):
                return item
        bits_shift = 15  # default we use 16 bits
        if 'bits' in item:
            bits_shift = item['bits'] - 1
        speed_id = random.choice(self.speed_list)
        tfm = self.transform_list[speed_id]
        data = item[self.key].reshape(-1, 1)
        data_type = data.dtype
        # scale to [-1, 1]
        data /= 1 << bits_shift
        p_data = tfm.build_array(input_array=data, sample_rate_in=item['sample_rate'])
        # scale back
        p_data *= 1 << bits_shift
        item['augmentation'].append(self.__class__.__name__)
        item[self.key] = p_data.astype(np.int16).astype(data_type).reshape(1, -1)
        if self.cls_relabel:
            # start from 1
            item[self.relabel_key] = speed_id + 1
            if self.vad_key in item:
                target_len = get_frames_len(item[self.key].shape[1])
                item[self.vad_key] = self.scale_vad(item[self.vad_key], target_len)
        return item


@PREPROCESS.register_module()
class TempoPerturbation(SpeedPerturbation):
    '''sox API: https://pysox.readthedocs.io/en/latest/api.html'''

    @staticmethod
    def get_transform(speed):
        '''get sox transform'''
        tfm = sox.Transformer()
        tfm.tempo(speed)
        return tfm


@PREPROCESS.register_module()
class WavResample:
    '''resample the waveform'''

    def __init__(self, sample_rate=8000, key='waveform'):
        '''init
        Args:
            sample_rate: target sample rate
            key: the key to resample
        '''
        self.sample_rate = sample_rate
        self.key = key
        self.tfm = sox.Transformer()
        self.tfm.rate(self.sample_rate)

    def __call__(self, item, **_kwargs):
        '''call to resample'''
        if item is None or 'sample_rate' not in item or item['sample_rate'] == self.sample_rate:
            return item
        if self.key not in item:
            return item
        bits_shift = 15  # default we use 16 bits
        data = item[self.key][0, :]
        if 'bits' in item:
            bits_shift = item['bits'] - 1
        data = item[self.key].reshape(-1, 1)
        data_type = data.dtype
        # scale to [-1, 1]
        data /= 1 << bits_shift
        p_data = self.tfm.build_array(input_array=data, sample_rate_in=item['sample_rate'])
        # scale back
        p_data *= 1 << bits_shift
        item['sample_rate'] = self.sample_rate
        item[self.key] = p_data.astype(np.int16).astype(data_type).reshape(1, -1)
        return item


@PREPROCESS.register_module()
class RandomWavResample:
    '''resample the waveform'''

    # pylint: disable=dangerous-default-value
    def __init__(self, sample_rate_list=[8000, 16000], key='waveform'):
        '''init
        Args:
            sample_rate: target sample rate
            key: the key to resample
        '''
        self.sample_rate_list = sample_rate_list
        self.key = key
        self.sample_rate_tfm = {}
        for sample_rate in self.sample_rate_list:
            tfm = sox.Transformer()
            tfm.rate(sample_rate)
            self.sample_rate_tfm[sample_rate] = tfm

    def __call__(self, item, **_kwargs):
        '''call to resample'''
        sample_rate = random.choice(self.sample_rate_list)
        if item is None or 'sample_rate' not in item or item['sample_rate'] == sample_rate:
            return item
        bits_shift = 15  # default we use 16 bits
        data = item[self.key][0, :]
        if 'bits' in item:
            bits_shift = item['bits'] - 1
        data = item[self.key].reshape(-1, 1)
        data_type = data.dtype
        # scale to [-1, 1]
        data /= 1 << bits_shift
        p_data = self.sample_rate_tfm[sample_rate].build_array(
            input_array=data, sample_rate_in=item['sample_rate']
        )
        # scale back
        p_data *= 1 << bits_shift
        item['sample_rate'] = sample_rate
        item[self.key] = p_data.astype(np.int16).astype(data_type).reshape(1, -1)
        return item


@PREPROCESS.register_module()
class AddNoise:
    '''add noise to the waveform'''

    # pylint: disable=line-too-long
    def __init__(
        self,
        key='waveform',
        noise_key='wav',
        noise_prefix='noise_sub',
        noise_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/dataset/xiansuotong_sil',
        noise_shards=256,
        min_snr=5.0,
        max_snr=20.0,
        p=0.5,
        snr_type='uniform',
        noise_chunk=20,
        resample_backend="sox",
        skip_num=0,
        prefetch_chunk_num=20,
    ):
        '''init
        Args:
            key: add noise to which key
            noise_key: noise binary data's key
            noise_prefix: prefix of noise file lists
            noise_dir: hdfs dir of noise dataset
            noise_shards: number of shards of noise dataset
            min_snr: min value of signal-to-noise ratio
            max_snr: max value of signal-to-noise ratio
            p: the prob of add noise
            snr_type: the type of random snr chosen from min_snr, max_snr
        '''
        self.key = key
        self.noise_key = noise_key
        self.noise_out_key = noise_key.replace('wav', 'waveform')
        self.noise_prefix = noise_prefix
        self.noise_dir = noise_dir
        self.noise_shards = noise_shards
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.prob = p
        self.type = snr_type
        self.chunks_buf = deque()
        self.data_buf = deque()
        self.noise_reader = None
        self.noise_array = None
        self.noise_rate = None
        self.noise_parser = WavParser(in_key=self.noise_key, min_len=0, max_len=float('inf'))
        self.chunk_size = min(noise_chunk, 50)
        self.resample_backend = resample_backend
        self.skip_num = skip_num
        self.prefetch_chunk_num = prefetch_chunk_num
        self.total_noise_shards = 0

    def get_reader(self):
        '''
        get noise reader
        '''
        noise_list = []
        # for compatibility of the old configs
        if isinstance(self.noise_dir, str):
            noise_list = [
                '{}/{}{}'.format(self.noise_dir, self.noise_prefix, shard)
                for shard in range(self.noise_shards)
            ]
        elif isinstance(self.noise_dir, (list, tuple)):
            assert len(self.noise_dir) == len(self.noise_prefix) == len(self.noise_shards)
            for data_dir, shard_prefix, shard_num in zip(
                self.noise_dir, self.noise_prefix, self.noise_shards
            ):
                noise_list.extend(
                    ['{}/{}{}'.format(data_dir, shard_prefix, idx) for idx in range(shard_num)]
                )
        self.total_noise_shards = len(noise_list)
        assert self.total_noise_shards > 0

        self.noise_reader = FalconReader(
            noise_list,
            256,  # fd_cache_size
            8,  # io_thread_num
            5,  # io_retry
            "AddNoise",  # unused
            get_local_size(),  # GPU num in one worker
            get_local_rank(),  # GPU idx in one worker
            self.chunk_size,  # chunk_size
        )

    def update_buf(self):
        '''update data buf'''
        if not self.chunks_buf:
            if self.noise_reader is None:
                self.get_reader()
            shard = random.randint(0, self.total_noise_shards - 1)
            entry_nums = self.noise_reader.get_entry_num([shard], False)
            chunks_idxs = [i * self.chunk_size for i in range(entry_nums // self.chunk_size)]
            self.chunks_buf.extend(
                [
                    chunks_idxs[i : i + self.prefetch_chunk_num]
                    for i in range(0, len(chunks_idxs), self.prefetch_chunk_num)
                ]
            )
        chunks = self.chunks_buf.popleft()
        datas = sum(self.noise_reader.read_many(chunks), [])
        self.data_buf.extend(datas)

    def get_data(self):
        '''get one noise data'''
        if not self.data_buf:
            self.update_buf()
        item = self.data_buf.popleft()
        item = pickle.loads(item)
        item = self.noise_parser(item)
        return item

    def resample(self, noise_array, ori_len, data_rate, bits):
        '''resample noise data using different backend'''
        sample_factor = self.noise_rate / data_rate
        if self.resample_backend == "sox":
            tfm = sox.Transformer()
            tfm.rate(data_rate)
            data = noise_array[0][: 1 + int(ori_len * sample_factor)].reshape(-1, 1)
            # scale to [-1, 1]
            data /= 1 << bits
            tmp_noise = tfm.build_array(input_array=data, sample_rate_in=self.noise_rate)
            # scale back
            tmp_noise *= 1 << bits
            cur_noise = tmp_noise[:ori_len].reshape(1, ori_len)
        else:
            tmp_noise = librosa.core.resample(
                noise_array[0, : 1 + int(ori_len * sample_factor)],
                orig_sr=self.noise_rate,
                target_sr=self.noise_rate / sample_factor,
                res_type='kaiser_fast',
            )
            cur_noise = tmp_noise[:ori_len].reshape(1, ori_len)
        return cur_noise

    def update_noise(self, ori_len, data_rate, bits=15):
        '''make sure the noise is long enough'''
        sample_factor = self.noise_rate / data_rate
        while self.noise_array is None or self.noise_array.shape[1] <= ori_len * sample_factor:
            noise = self.get_data()
            if noise is None or self.noise_out_key not in noise:
                continue
            self.noise_rate = noise['sample_rate']
            if self.noise_array is None:
                self.noise_array = noise[self.noise_out_key]
            else:
                self.noise_array = np.concatenate(
                    (self.noise_array, noise[self.noise_out_key]), axis=1
                )
        if sample_factor != 1:
            cur_noise = self.resample(
                self.noise_array, bits=bits, ori_len=ori_len, data_rate=data_rate
            )
        else:
            cur_noise = self.noise_array[:, :ori_len]
        self.noise_array = self.noise_array[:, int(ori_len * sample_factor) :]
        return cur_noise

    def __call__(self, item, **_kwargs):
        '''add noise to each item'''
        if self.skip_num > 0:
            self.skip_num -= 1
            return item
        if random.random() > self.prob or item is None or self.key not in item:
            return item
        ori_waveform = item[self.key]
        ori_len = ori_waveform.shape[1]
        if self.noise_rate is None:
            self.noise_rate = item['sample_rate']
        bits = item.get("bits", 16) - 1
        noise_waveform = self.update_noise(ori_len, item['sample_rate'], bits=bits)
        if self.type == 'uniform':
            snr = random.uniform(self.min_snr, self.max_snr)

        # c++ version add_noise, faster than python version
        # but must satisfy：
        #  1. ori_waveform.shape[1] == noise_waveform.shape[1]
        #  2. dtype == np.float32
        #  3. have same sample_rate
        new_waveform = add_noise_c(ori_waveform, noise_waveform, snr)

        new_max = np.abs(new_waveform).max()
        if new_max > 2**bits - 1:
            new_waveform /= new_max
            new_waveform *= 2**bits - 1
        item['augmentation'].append(self.__class__.__name__)
        item[self.key] = new_waveform
        return item


@PREPROCESS.register_module()
class AddSpeechNoise(AddNoise):
    '''add speech to the waveform with random offset.'''

    def __init__(
        self,
        key='waveform',
        noise_key='wav',
        noise_prefix='noise_sub',
        noise_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/dolphin/'
                  'dataset/xiansuotong_sil',
        noise_shards=256,
        min_snr=5.0,
        max_snr=20.0,
        p=0.5,
        snr_type='uniform',
        noise_chunk=20,
        resample_backend="sox",
        skip_num=0,
        prefetch_chunk_num=20,
        sc_token='$',
        avg_overlap_dur=3.0,
        extra_start=0.5,
        extra_end=0.0,
        label_key='label',
        spkid_key='spkid',
    ):
        super().__init__(
            key=key,
            noise_key=noise_key,
            noise_prefix=noise_prefix,
            noise_dir=noise_dir,
            noise_shards=noise_shards,
            min_snr=min_snr,
            max_snr=max_snr,
            p=p,
            snr_type=snr_type,
            noise_chunk=noise_chunk,
            resample_backend=resample_backend,
            skip_num=skip_num,
            prefetch_chunk_num=prefetch_chunk_num,
        )
        self.sc_token = sc_token
        self.avg_overlap_dur = avg_overlap_dur
        self.extra_start = extra_start
        self.extra_end = extra_end
        self.label_key = label_key
        self.spkid_key = spkid_key

    def attach_waveform_noise(
        self, waveform, noise_waveform, sample_rate, offset=0, is_same_spkid=False
    ):
        '''
        attach noise after waveform (overlap allowed)
        '''
        assert waveform.ndim == 2, waveform.shape
        if is_same_spkid:
            overlap = 0
        else:
            es = round(sample_rate * self.extra_start)
            ed = round(sample_rate * self.extra_end)
            max_overlap = max(0, min(waveform.shape[1] - offset - es, noise_waveform.shape[1] - ed))
            overlap = round(sample_rate * np.random.exponential(self.avg_overlap_dur))
            overlap = min(overlap, max_overlap)

        if overlap < noise_waveform.shape[1]:
            offset = noise_waveform.shape[1] - overlap
            waveform, noise_waveform = np.pad(
                waveform, ((0, 0), (0, noise_waveform.shape[1] - overlap))
            ), np.pad(noise_waveform, ((0, 0), (waveform.shape[1] - overlap, 0)))
        else:
            shift = random.randint(0, waveform.shape[1] - noise_waveform.shape[1])
            offset = shift + noise_waveform.shape[1]
            noise_waveform = np.pad(
                noise_waveform,
                ((0, 0), (shift, waveform.shape[1] - shift - noise_waveform.shape[1])),
            )

        waveform = waveform.astype(np.float32)
        noise_waveform = noise_waveform.astype(np.float32)
        snr = 0
        # c++ version add_noise, faster than python version
        # but must satisfy：
        #  1. ori_waveform.shape[1] == noise_waveform.shape[1]
        #  2. dtype == np.float32
        #  3. have same sample_rate
        waveform = add_noise_c(waveform, noise_waveform, snr)
        return waveform, offset

    def attach_label_noise(self, label, noise_label):
        '''
        attach noise_label after label (with sc token in between)
        '''
        label.append(self.sc_token)
        label += noise_label
        return label

    def __call__(self, item, **_kwargs):
        '''add noise to each item'''
        if self.skip_num > 0:
            self.skip_num -= 1
            return item
        if random.random() > self.prob or item is None or self.key not in item:
            return item

        if self.noise_rate is None:
            self.noise_rate = item['sample_rate']

        noise = self.get_data()
        while noise is None or self.noise_out_key not in noise:
            continue
        noise_waveform = noise[self.noise_out_key]
        noise_label = noise[self.label_key]
        noise_spkid = noise.get(self.spkid_key, None)

        waveform = item[self.key]
        label = item[self.label_key]
        spkid = item.get(self.spkid_key, None)
        sample_rate = item['sample_rate']

        is_same_spk = spkid and spkid == noise_spkid
        waveform, _ = self.attach_waveform_noise(
            waveform,
            noise_waveform,
            sample_rate,
            offset=0,
            is_same_spkid=is_same_spk,
        )
        bits = item.get("bits", 16) - 1
        max_mag = np.abs(waveform).max()
        if max_mag > 2**bits - 1:
            waveform /= max_mag
            waveform *= 2**bits - 1
        item[self.key] = waveform

        label = self.attach_label_noise(label, noise_label)
        item[self.label_key] = label
        item['augmentation'].append(self.__class__.__name__)
        return item


@PREPROCESS.register_module()
class AddNoiseFalcon(AddNoise):
    '''add noise to the waveform, saved for compatible'''

    def __init__(self, *args, **kwargs):
        warnings.warn("AddNoiseFalcon may be deprecated, it is recommended to use AddNoise first")
        super().__init__(*args, **kwargs)


@PREPROCESS.register_module()
class ModifyProsody:
    '''Modify prosody on the waveform.
    This module is literally for prosody modification, not for speed perturbation on audio.
    The changing of speed is to adjust the pitch feature, and then modify the tempo to
    restore the original pacing of an utterance. Note that the speed_factor and tempo_factor
    are coupled, and one is roughly a reciprocal of the other.
    If you want to do speed perturbation only, please refer to the module of SpeedPerturbation.
    '''

    def __init__(
        self, key='waveform', sample_rate=16000, speed_factor=1.0, tempo_factor=1.0, adjust_prob=0.0
    ):
        '''init
        Args:
            key: the key of waveform to adjust tempo
            sample_rate: the sample_rate of wave
            speed_factor: the ratio of the new speed to the old speed
            tempo_factor: the ratio of new tempo to the old tempo
            adjust_prob: the probability of an item's prosody to be adjusted
        '''
        self.key = key
        self.sample_rate = sample_rate
        self.speed_factor = speed_factor
        self.tempo_factor = tempo_factor
        self.adjust_prob = adjust_prob
        self.modify_prosody = (speed_factor != 1.0 or tempo_factor != 1.0) and adjust_prob > 0.0
        self.sox_trans = None
        self.version = Version(torchaudio.__version__) < Version('0.7.0')

    def __call__(self, item, **_kwargs):
        '''adjust the prosody of an adult speech waveform'''
        if not self.modify_prosody or item is None:
            return item
        if 'child' in item.get('tag', []) or self.key not in item or 'sample_rate' not in item:
            return item
        if np.random.rand() > self.adjust_prob:
            return item
        # To be compatible with torchaudio v0.5.0 and multiprocessing,
        # the sox.effects' initialization should be placed inside
        # of __call__() body, and only initialized once.

        if self.version:
            if self.sox_trans is None:
                torchaudio.initialize_sox()
                self.sox_trans = torchaudio.sox_effects.SoxEffectsChain()
                # the order of sox effects for prosody modification is:
                # speed, tempo and resampling.
                if self.speed_factor != 1.0:
                    self.sox_trans.append_effect_to_chain('speed', [self.speed_factor])
                if self.tempo_factor != 1.0:
                    self.sox_trans.append_effect_to_chain('tempo', [self.tempo_factor])
            self.sox_trans.append_effect_to_chain('rate', [self.sample_rate])
            # NOTE: The following codes save the waveform to a local file and feed it to the
            # sox backend for aduio processing. This is an interim solution.
            # TODO: @Jinkun, Make the torchaudio.sox_effects module into an extention, or
            # upgrade torchaudio to version of 0.7.0 if possible.
            tmp_wav_path = '/tmp/tmp_{}.wav'.format(item['uttid'])
            wavfile.write(tmp_wav_path, item['sample_rate'], item[self.key].T.astype(np.int16))
            self.sox_trans.set_input_file(tmp_wav_path)
            waveform, _ = self.sox_trans.sox_build_flow_effects()
            item['augmentation'].append(self.__class__.__name__)
            item[self.key] = waveform.numpy()
        else:
            # Defines the effects to apply
            effects = []
            if self.speed_factor != 1.0:
                effects.append(['speed', str(self.speed_factor)])
            if self.tempo_factor != 1.0:
                effects.append(['tempo', str(self.tempo_factor)])
            effects.append(['rate', str(self.sample_rate)])
            tmp_wav_path = '/tmp/tmp_{}.wav'.format(item['uttid'])
            wavfile.write(tmp_wav_path, item['sample_rate'], item[self.key].T.astype(np.int16))

            # Apply effects and load data with channels_first=True
            waveform, _ = torchaudio.sox_effects.apply_effects_file(tmp_wav_path, effects)
            item['augmentation'].append(self.__class__.__name__)
            item[self.key] = waveform.numpy()

        try:
            os.remove(tmp_wav_path)
        except Exception:
            pass
        return item

    def __del__(self):
        '''shutdown the sox backend if necessary'''
        if self.version:
            if self.sox_trans is not None:
                torchaudio.shutdown_sox()


@PREPROCESS.register_module()
class WavCrop:
    '''crop waveform'''

    def __init__(self, key='waveform', crop_sample_length=480000, two_dim_waveform=False):
        '''init
        Args:
            key: the key to crop
            crop_sample_length: crop sample length
            two_dim_waveform: fix the invalid usage when the previous item transform
                generates 2-dim waveform (e.g. WavResample)
        '''
        self.key = key
        self.crop_sample_length = crop_sample_length
        self.two_dim_waveform = two_dim_waveform

    def __call__(self, item, **_kwargs):
        '''crop data item to (0, crop_sample_length]'''
        if self.two_dim_waveform:
            left_size = np.size(item[self.key], 1) - self.crop_sample_length
        else:
            left_size = len(item[self.key]) - self.crop_sample_length

        if left_size > 0:
            start = random.randint(0, left_size)
            if self.two_dim_waveform:
                item[self.key] = item[self.key][:, start : self.crop_sample_length + start]
            else:
                item[self.key] = item[self.key][start : self.crop_sample_length + start]

        return item


@PREPROCESS.register_module()
class ChangeAmplitude:
    '''change amplitude'''

    def __init__(self, prop=0.5, amplitude_range=(0.7, 1.1)):
        '''init.'''
        self.amplitude_range = amplitude_range
        self.prop = prop

    def __call__(self, data: torch.Tensor):
        '''Changes amplitude of an audio randomly.'''
        if random.uniform(0, 1) <= self.prop:
            data = data * random.uniform(*self.amplitude_range)
        return data


@PREPROCESS.register_module()
class FixAudioLength:
    '''fix audio length'''

    def __init__(self, time=1, sample_rate=16000):
        '''init.'''
        self.target_len = time * sample_rate

    def __call__(self, data: torch.Tensor):
        '''Either pads or truncates an audio into a fixed length.'''
        cur_len = data.shape[1]
        if self.target_len <= cur_len:
            data = data[:, : self.target_len]
        else:
            data = torch.nn.functional.pad(data, (0, self.target_len - cur_len))
        return data


@PREPROCESS.register_module()
class ChangeSpeedAndPitchAudio:
    '''change speed and pitch audio'''

    def __init__(self, prop=0.5, max_scale=0.2, sample_rate=16000):
        '''init.'''
        self.max_scale = max_scale
        self.sample_rate = sample_rate
        self.prop = prop

    def __call__(self, data):
        '''Change the speed of an audio. This transform also changes the pitch of the audio.'''
        if random.uniform(0, 1) <= self.prop:
            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            data = torch.nn.functional.interpolate(
                data.unsqueeze(1), scale_factor=speed_fac, mode="nearest"
            ).squeeze(1)
        return data


@PREPROCESS.register_module()
class TimeshiftAudio:
    '''time shift audio'''

    def __init__(self, prop=0.5, max_shift_seconds=0.2, sample_rate=16000):
        '''init.'''
        self.shift_len = max_shift_seconds * sample_rate
        self.prop = prop

    def __call__(self, data):
        '''Shifts an audio randomly.Shifts an audio randomly.'''
        if random.uniform(0, 1) <= self.prop:
            shift = random.randint(-self.shift_len, self.shift_len)
            a = -min(0, shift)
            b = max(0, shift)
            data = torch.nn.functional.pad(data, (a, b), "constant")
            data = data[:, : data.shape[1] - a] if a else data[:, b:]
        return data


@PREPROCESS.register_module()
class PreSelAug:
    """
    randomly select a type of noise or rir
    """

    def __init__(
        self,
        aug_probs=None,
    ):
        '''init.'''
        self.aug_types = []
        self.aug_probs = []
        if aug_probs is None:
            aug_probs = {'no_aug': 1, 'music': 1, 'speech': 1, 'noise': 1, 'reverb': 1}
        self.norm_probs(aug_probs)

    def norm_probs(self, aug_probs):
        '''
        normalize the weights of aug_types
        in: dict
        after:
        types : []
        probs : []
        '''
        tot = sum(aug_probs.values())
        for aug_type, aug_prob in aug_probs.items():
            self.aug_types.append(aug_type)
            self.aug_probs.append(1.0 * aug_prob / tot)
            logging.info("aug_type : %s, aug_prob: %.3f ", aug_type, 1.0 * aug_prob / tot)

    def rand_type(self):
        """get a random type with each prob"""
        return np.random.choice(self.aug_types, p=self.aug_probs)

    def __call__(self, item, **_kwargs):
        """call func"""
        aug_type = self.rand_type()
        item['augmentation_type'] = aug_type
        return item


@PREPROCESS.register_module()
class KaldiAddNoise(AddNoise):
    """Add Special type noise as kaldi"""

    def __init__(
        self,
        key='waveform',
        noise_key='wav',
        noise_prefix='noise',
        noise_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/wuyangcheng/data/new/wav_pkg/musan',
        noise_shards=16,
        noise_snr=None,
        noise_times=None,
        noise_chunk=20,
        prefetch_chunk_num=10,
        aug_key='augmentation_type',
        noise_type='noise',
        ground_mode='foreground',
        noise_interval=0,
        sr_key='sample_rate',
        min_snr=5.0,
        max_snr=20.0,
        p=0.5,
        snr_type='uniform',
        resample_backend="sox",
        skip_num=0,
    ):
        """
        noise_ground is noise groundtype
        if background, noise add method is:
            if noise < waveform:
                noise += noise
            noise = noise[:wave_len]
        if foreground:
            noise0 - interval - noise1 - interval....
            snr0                snr1 ...
        """
        super().__init__(
            key=key,
            noise_key=noise_key,
            noise_prefix=noise_prefix,
            noise_dir=noise_dir,
            noise_shards=noise_shards,
            noise_chunk=noise_chunk,
            prefetch_chunk_num=prefetch_chunk_num,
            min_snr=min_snr,
            max_snr=max_snr,
            p=p,
            snr_type=snr_type,
            resample_backend=resample_backend,
            skip_num=skip_num,
        )
        self.noise_snr = [5, 10] if noise_snr is None else noise_snr
        self.noise_times = [1] if noise_times is None else noise_times
        self.aug_key = aug_key
        self.noise_type = noise_type
        self.ground_mode = ground_mode
        self.noise_interval = noise_interval
        self.sr_key = sr_key

    def get_noise(self):
        '''get one piece of noise data'''
        noise_data = None
        while True:
            noise_data = self.get_data()
            if noise_data is None or self.noise_out_key not in noise_data:
                continue
            break
        return noise_data

    # pylint: disable=too-many-function-args
    def add_noise(self, ori_waveform, sample_rate, early_energy):
        """
        backgroung mode:
            noise repeat until noise longer than train waveform
        foreground mode:
            random choice some pieces noises, and spilt by interval,
            each has own snr
        """
        ori_len = ori_waveform.shape[1]
        snr = random.choice(self.noise_snr)
        start = 0
        if self.ground_mode == 'background':
            noise_waveform = self.get_noise()[self.noise_out_key]
            noise = noise_waveform
            while noise_waveform.shape[1] < ori_len:  # noise repeat
                noise_waveform = np.concatenate((noise_waveform, noise), axis=1)
            noise_waveform = noise_waveform[:, :ori_len]
            ori_waveform = add_noise_c(
                ori_waveform,
                noise_waveform,
                snr,
                early_energy,
                start,
                ori_len,
            )
        elif self.ground_mode == 'foreground':
            while start < ori_waveform.shape[1]:
                noise_waveform = self.get_noise()[self.noise_out_key]
                snr = random.choice(self.noise_snr)
                ori_waveform = add_noise_c(
                    ori_waveform, noise_waveform, snr, early_energy, start, ori_len
                )
                start += noise_waveform.shape[1]
                start += self.noise_interval * sample_rate  # noise interval

        return ori_waveform

    def __call__(self, item, **_kwargs):
        if self.aug_key not in item:
            return item
        if not item[self.aug_key] == self.noise_type:
            return item
        noise_times = random.choice(self.noise_times)
        ori_waveform = item[self.key]
        early_energy = ori_waveform[0] @ ori_waveform[0] / ori_waveform.shape[1]
        for _ in range(noise_times):
            ori_waveform = self.add_noise(ori_waveform, item[self.sr_key], early_energy)
        power_after_reverb = ori_waveform[0] @ ori_waveform[0] / ori_waveform.shape[1]
        ori_waveform = ori_waveform * (math.sqrt(early_energy / power_after_reverb))
        item[self.key] = ori_waveform
        return item


@PREPROCESS.register_module()
class SelectWavChannel:
    """
    Audio channel selection
    """

    def __init__(self, wav_key='src', target_channel=0):
        """
        init.
        Args:
            wav_key: waveform key
            target_channel: target channel. If target_channel==-1,
                            all channels are randomly selected.
                            If target_channel is None, the multi-channel
                            signal is output. This is only used when the input is multi-channel.
        """
        self.wav_key = wav_key
        self.target_channel = target_channel

    def __call__(self, item, **_kwargs):
        """
        call function
        """
        if item is None or self.wav_key not in item:
            return None

        waveform = item[self.wav_key]
        if self.target_channel == -1:
            # randomly select a channel
            item['target_channel'] = random.randint(0, waveform.shape[0] - 1)
        elif self.target_channel is None:
            # use multi-channel data
            item['target_channel'] = -1
        else:
            # use the selected channel
            item['target_channel'] = self.target_channel

        # Note that we use -1 to denote the multi-channel data!
        if item['target_channel'] != -1:
            # select the data from the channel
            waveform = waveform[[item['target_channel']], :]
        item[self.wav_key] = waveform
        return item


@PREPROCESS.register_module()
class WavSplicing:
    """
    Wav splicing to target length in the desired mode
    Labels can be intercepted at the same time
    """

    def __init__(
        self,
        wav_key='src',
        target_len=16,
        mode='time',
        sample_rate=16000,
        label_key='label',
        random_clip=True,
    ):
        """
        Args:
            wav_key: waveform key
            target_len: want get length
            mode: target length mode, can be 'time', 'waveform', 'frames'
                  if mode=='time':
                      target_len means the duration of the target audio
                  if mode=='waveform':
                      target_len means the num_samples of the target audio
                  if mode=='frames':
                      target_len means the frame length of the target audio

        """
        self.wav_key = wav_key
        self.sample_rate = sample_rate
        self.label_key = label_key
        self.get_target_lens(mode, target_len)
        self.random_clip = random_clip

    def get_target_lens(self, mode, target_len):
        """
        get target num_samples and frames num
        """
        if mode == 'time':
            self.target_samples = target_len * self.sample_rate
            self.target_frames = get_frames_len(self.target_samples, self.sample_rate)
        elif mode == 'waveform':
            self.target_samples = target_len
            self.target_frames = get_frames_len(target_len, self.sample_rate)
        elif mode == 'frames':
            self.target_frames = target_len
            self.target_samples = get_wav_len(target_len)

    def __call__(self, item, **_kwargs):
        """
        call
        """
        if item is None or self.wav_key not in item:
            return None

        # split waveform
        waveform = item[self.wav_key].astype(np.float32)
        if waveform.shape[1] < self.target_samples:
            return item
        if self.random_clip:
            frames_len = get_frames_len(waveform.shape[1])
            start_frame_id = random.randint(0, frames_len - self.target_frames)
            start = wav_start_pos(start_frame_id, self.sample_rate)
            waveform = waveform[:, start : start + self.target_samples]
        else:
            start_frame_id = 0
            waveform = waveform[:, : self.target_samples]
        item[self.wav_key] = waveform
        # spilt label
        if self.label_key in item:
            label = item[self.label_key]
            label = label[..., start_frame_id : start_frame_id + self.target_frames]
            item[self.label_key] = label
        item['sample_rate'] = self.sample_rate
        return item


@PREPROCESS.register_module()
class AudioReplace:
    """
    speaker_audio replace
    """

    def __init__(
        self,
        data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/ts_vad/'
        'pkg/Train_Ali_far_reorder',
        file_list=None,
        file_prefix='spk_train',
        shards_num=8,
        sample_rate=16000,
        key='src',
        label_key='label',
        speakers_key='speakers',
        p=0.9,
        embedding_keys=None,
        use_same_utt_embedding=True,
        use_same_utt=False,
        swap_dim=True,
    ):
        """
        init
        """
        self.data_root = data_root
        self.file_list = file_list
        self.file_prefix = file_prefix
        self.shards_num = shards_num
        self.sample_rate = sample_rate
        self.key = key
        self.label_key = label_key
        self.speakers_key = speakers_key
        self.prob = p
        self.speakers = set()  # spk list
        self.speakers_datas = None
        self.speakers_durs = None
        self.speakers_max_dur = None
        self.reader = None
        self.frame_shift = 10.0
        self.init_data_paths()
        self.get_embedding_keys(embedding_keys, use_same_utt_embedding)
        self.use_same_utt_embedding = use_same_utt_embedding
        self.use_same_utt = use_same_utt
        self.swap_dim = swap_dim

    def get_embedding_keys(self, embedding_keys, use_same_utt_embedding):
        """
        get embedding keys list
        """
        self.emb_keys = set()
        if embedding_keys is None:
            return
        for key in embedding_keys:
            if use_same_utt_embedding:
                self.emb_keys.add(key.rsplit('-', 1)[0])  # utt-spk
            else:
                self.emb_keys.add(key.rsplit("-", 2)[1])  # spk

    def init_data_paths(self):
        """
        get all data_paths,
        and download all csv paths in gpu-0
        """
        # 1. {data_root}/{file_prefix}{shards_num}
        # 2. {data_root}/{file_list}
        if self.file_list is None:
            self.file_list = (
                f'[\"{self.file_prefix}' + "{}\".format(i)" + f' for i in range({self.shards_num})]'
            )
        if isinstance(self.data_root, str):
            self.data_root = [self.data_root]
            self.file_list = [self.file_list]
        self.data_list = []
        assert len(self.data_root) == len(self.file_list)
        for data_root, data_files in zip(self.data_root, self.file_list):
            dataset_now = [os.path.join(data_root, p) for p in eval(data_files)]
            self.data_list += dataset_now
        self.path_list = ["".join(path.split()) for path in self.data_list]
        for idx, path in enumerate(self.path_list):
            remote_csv_path = path + '.csv'
            local_csv_path = f"path_{idx}_" + os.path.basename(remote_csv_path)
            dist_file_get(remote_file=remote_csv_path, local_file=local_csv_path)

    def get_reader(self):
        """
        get FalconReader
        """
        self.reader = FalconReader(
            self.path_list,
            256,
            8,
            5,
            "AudioReplace",
            get_local_size(),
            get_local_rank(),
            1,
        )

    def update_speakers(self):
        """
        get speakers and its value
        """
        if self.reader is None:
            self.get_reader()
        # [spk][utt-spk] -> key-idx
        self.speakers_datas = defaultdict(lambda: defaultdict(lambda: []))
        # [spk][utt-spk] -> dur
        self.speakers_durs = defaultdict(lambda: defaultdict(lambda: []))
        # [spk] -> max_dur
        self.speakers_max_dur = defaultdict(lambda: 0)

        keys = self.reader.list_keys()
        key2idx = dict()
        for key_id, key in enumerate(keys):
            if key == 'meta':
                continue
            key2idx[key] = key_id
        for idx, path in enumerate(self.path_list):
            remote_csv_path = path + '.csv'
            local_csv_path = f"path_{idx}_" + os.path.basename(remote_csv_path)
            with open(local_csv_path, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    spk = row['spk']
                    wav_len = int(row['src_len'])
                    utt_spk = row['key'].rstrip('-' + row['key'].split('-')[-1])
                    if self.emb_keys:
                        tmp_key = utt_spk if self.use_same_utt_embedding else spk
                        if tmp_key not in self.emb_keys:
                            continue
                    self.speakers.add(spk)
                    self.speakers_datas[spk][utt_spk].append(key2idx[row['key']])
                    self.speakers_durs[spk][utt_spk].append(wav_len)
                    self.speakers_max_dur[spk] = max(self.speakers_max_dur[spk], wav_len)
        del keys
        del key2idx

    def get_speakers(self, raw_speakers, raw_speaker_dur):
        """
        get speaker_num pieces speakers waveform and its clss
        it requires the audio of the sampled speaker is longer than the original duration

        the raw_speakers follow the format "utt-spk", so the output should be in the same format.
        """
        if len(self.speakers) <= 0:
            self.update_speakers()

        new_speakers = []  # utt-spk
        data_idxs = []
        speakers = set()
        for spk, dur in zip(raw_speakers, raw_speaker_dur):
            if spk == '':
                continue

            # Step 1: choose a speaker
            valid_speakers = [s for s in self.speakers if self.speakers_max_dur[s] > dur]
            if self.use_same_utt:
                raw_utt = spk.split('-')[0]
                valid_speakers = [
                    s
                    for s in valid_speakers
                    if (raw_utt + '-' + s)
                    in self.speakers_datas[s].keys()  # raw_utt-spk in speaker datas
                    and max(self.speakers_durs[s][(raw_utt + '-' + s)]) > dur  # max_dur > dur
                ]

            if not valid_speakers:
                return None, None

            while True:
                new_spk = random.choice(valid_speakers)
                if new_spk not in speakers:
                    break
            speakers.add(new_spk)

            # Step 2: choose a utterance
            if self.use_same_utt:
                new_utt_spk = raw_utt + '-' + new_spk
            else:
                valid_utt_spks = [
                    utt_spk
                    for utt_spk in self.speakers_datas[new_spk].keys()
                    if max(self.speakers_durs[new_spk][utt_spk]) > dur
                ]
                new_utt_spk = random.choice(valid_utt_spks)

            # Step 3: choose a segment
            new_speakers.append(new_utt_spk)
            data_idxs.append(
                random.choice(
                    [
                        this_data
                        for this_dur, this_data in zip(
                            self.speakers_durs[new_spk][new_utt_spk],
                            self.speakers_datas[new_spk][new_utt_spk],
                        )
                        if this_dur > dur
                    ]
                )
            )

        values = self.reader.read_many(data_idxs, True)
        spk_datas = []
        for val in values:
            waveform = pickle.loads(val[0])[self.key]
            if self.swap_dim:
                waveform = waveform.transpose()
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            spk_datas.append(waveform)
        return deque(new_speakers), deque(spk_datas)

    def audio_replace(self, raw_waveform, spk_waveform, label):
        """
        use add to replace spk
        """
        valid_segments = []
        start = 0
        valid_frames = 0
        if np.sum(label) == 0:
            return raw_waveform
        for frame_id, frame_label in enumerate(label):
            if frame_label == 1:
                if valid_frames == 0:
                    start = frame_id
                valid_frames += 1
            elif frame_label == 0:
                if valid_frames > 0:
                    valid_segments.append([start, frame_id])
                    valid_frames = 0
        # at the end of the label
        if valid_frames > 0:
            valid_segments.append([start, len(label)])
        assert np.sum(label) == sum(seg[1] - seg[0] for seg in valid_segments)
        valid_frames = sum(seg[1] - seg[0] for seg in valid_segments)
        spk_wavlen = get_wav_len(valid_frames, self.sample_rate)
        if spk_wavlen >= spk_waveform.shape[1]:
            logging.warning("Not enough waveform found in the candidate.")
            spk_wavlen = spk_waveform.shape[1]

        spk_start = random.randint(0, spk_waveform.shape[1] - spk_wavlen)
        for seg in valid_segments:
            wav_start = wav_start_pos(seg[0], self.sample_rate)
            # For the last frame, the wavlen is longer
            wav_end = (
                raw_waveform.shape[1]
                if seg[1] == len(label)
                else wav_start_pos(seg[1], self.sample_rate)
            )
            this_wavlen = min(wav_end - wav_start, spk_waveform.shape[1] - spk_start)
            raw_waveform[:, wav_start : wav_start + this_wavlen] += spk_waveform[
                :, spk_start : spk_start + this_wavlen
            ]
            spk_start += this_wavlen
        return raw_waveform

    def get_effective_len(self, spk_list, label):
        '''
        get the effective duration of each speaker in the utterance
        '''
        dur = []
        for i, spk in enumerate(spk_list):
            if spk == '':
                dur.append(0)
            else:
                valid_frames = np.sum(label[i])
                dur.append(
                    get_wav_len(valid_frames, sample_rate=self.sample_rate)
                    if valid_frames > 0
                    else 0
                )
        return dur

    def __call__(self, item, **_kwargs):
        """
        call
        """
        if item is None or self.key not in item:
            return item

        if random.uniform(0, 1) > self.prob:
            raw_speakers = item[self.speakers_key]
            for spk in raw_speakers:
                tmp_key = spk if self.use_same_utt_embedding else spk.rsplit("-", 1)[1]
                if self.emb_keys and tmp_key not in self.emb_keys:
                    return None
            return item

        raw_waveform = item[self.key]
        new_waveform = np.zeros(raw_waveform.shape, dtype=np.float32)
        label = item[self.label_key]
        raw_speakers = item[self.speakers_key]

        # effective speakers num and duration
        raw_speakers_durs = self.get_effective_len(raw_speakers, label)
        speakers, waveforms = self.get_speakers(raw_speakers, raw_speakers_durs)
        if speakers is None and waveforms is None:
            return item
        new_speakers = []
        for spk_id, spk in enumerate(raw_speakers):
            if spk == '':
                new_speakers.append(spk)
            else:
                speaker = speakers.popleft()
                waveform = waveforms.popleft()
                if item['target_channel'] != -1:
                    waveform = waveform[[item['target_channel']], :]
                new_speakers.append(speaker)
                new_waveform = self.audio_replace(new_waveform, waveform, label[spk_id])
        # TODO(liuyi): Do we need to normalize the waveform?
        if np.max(np.abs(new_waveform)) > 32767.0:
            scale = np.max(np.abs(new_waveform)) / 32767.0
            new_waveform /= scale
        item[self.key] = new_waveform
        item[self.speakers_key] = new_speakers
        return item


@PREPROCESS.register_module()
class WaveFormat:
    """
    Standardize the format of waveform to [channels, num_samples]
    """

    def __init__(self, key='waveform', swap_dim=False):
        """
        Args:
            key(str): waveform key
            swap_dim(bool): whether swap channel dim and samples dim
        """
        self.key = key
        self.swap_dim = swap_dim

    def __call__(self, item, **_kwargs):
        """call."""
        if item is None or self.key not in item:
            return item
        waveform = item[self.key]

        if waveform.ndim == 1:
            item[self.key] = waveform[np.newaxis, :]
        if self.swap_dim:
            item[self.key] = waveform.transpose()
        return item
