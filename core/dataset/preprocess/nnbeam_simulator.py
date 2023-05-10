'''
nnbeam data simulation
'''
# pylint: disable=simplifiable-if-statement
# pylint: disable=no-value-for-parameter
# pylint: disable=missing-function-docstring
import random
import numpy as np
import torch
import librosa
from falconclaw.dataloader import KVReader  # pylint:disable=import-error
from scipy.signal import fftconvolve
from core.utils import gen_diffuse
from core.utils.se.simulator_module import ModuleSimulator
from core.utils import Config
from .preprocess import PREPROCESS


MILLISECONDS_TO_SECONDS = 0.001


@PREPROCESS.register_module()
class MultiChannelModuleSimu:
    '''item transform for multi-channel training.'''

    def __init__(
        self,
        wav_key="waveform",
        simulator_config_path='./configs/se/simulator_config_jointtrain.py',
        simulate_type='simulate',
    ):
        """init."""
        self.wav_key = wav_key
        self.simulator = ModuleSimulator(Config.fromfile(simulator_config_path).cfg_dict)
        self.simulate_type = simulate_type

    def __call__(self, item, **_kwargs):
        """do simulate as config modified type"""
        if self.wav_key not in item:
            return item
        simulate_func = getattr(self.simulator, self.simulate_type)
        return simulate_func(
            item,
            wav_key=self.wav_key,
        )


@PREPROCESS.register_module()
class DataCollateNNbeam:
    '''collate data for nnbeam training.'''

    def __init__(
        self,
        wav_key="waveform",
        target_wav_key='target_waveform',
        target_2ch_wav_key='target_2ch_waveform',
        rir_dir_key='direction',
    ):
        '''init.'''
        self.wav_key = wav_key
        self.rir_dir_key = rir_dir_key
        self.target_wav_key = target_wav_key
        self.target_2ch_wav_key = target_2ch_wav_key

    def __call__(self, bucket_list, batch_out):
        '''do collate for rir and data.'''

        batch_waveform = batch_out[self.wav_key]
        bsz, max_samples, _ = batch_waveform.shape
        pad_samples = batch_out.get('pad_samples', [0] * bsz)

        batch_2ch_target_wav = torch.zeros([bsz, max_samples, 2]).float()
        batch_target_wav = torch.zeros([bsz, max_samples]).float()
        batch_rir_dir = torch.zeros([bsz, 1]).float()

        for i, item in enumerate(bucket_list):
            target_waveform = item[self.target_wav_key]
            target_2ch_waveform = item[self.target_2ch_wav_key]
            rir_dir = item[self.rir_dir_key]

            pad_sample = pad_samples[i]
            target_waveform_len = target_waveform.shape[0]
            batch_target_wav[i, pad_sample : pad_sample + target_waveform_len] = torch.from_numpy(
                target_waveform
            ).float()
            target_2ch_waveform_len = target_2ch_waveform.shape[0]
            batch_2ch_target_wav[
                i, pad_sample : pad_sample + target_2ch_waveform_len
            ] = torch.from_numpy(target_2ch_waveform).float()
            batch_rir_dir[i] = float(rir_dir)

        batch_out[self.target_wav_key] = batch_target_wav
        batch_out[self.target_2ch_wav_key] = batch_2ch_target_wav
        batch_out[self.rir_dir_key] = batch_rir_dir


@PREPROCESS.register_module()
class DataCollateNNbeamInfer:
    '''collate data for nnbeam inference.'''

    def __init__(
        self,
        wav_key="waveform",
        doa_key='wkp_angle',
        wkp_start_key='wkp_start',
        wkp_end_key='wkp_end',
    ):
        '''init.'''
        self.wav_key = wav_key
        self.doa_key = doa_key
        self.wkp_start_key = wkp_start_key
        self.wkp_end_key = wkp_end_key

    def __call__(self, bucket_list, batch_out):
        '''do collate for data.'''

        batch_waveform = batch_out[self.wav_key]
        bsz, _, _ = batch_waveform.shape

        batch_doa = torch.zeros([bsz, 1]).float()
        batch_wkp_start = torch.zeros([bsz, 1]).float()
        batch_wkp_end = torch.zeros([bsz, 1]).float()

        for i, item in enumerate(bucket_list):
            doa = item[self.doa_key]
            wkp_start = item[self.wkp_start_key]
            wkp_end = item[self.wkp_end_key]

            batch_doa[i] = float(doa)
            batch_wkp_start[i] = float(wkp_start)
            batch_wkp_end[i] = float(wkp_end)

        batch_out[self.doa_key] = batch_doa
        batch_out[self.wkp_start_key] = batch_wkp_start
        batch_out[self.wkp_end_key] = batch_wkp_end


@PREPROCESS.register_module()
class CalculateFrameLengthFake:
    '''fake frame length number for every uttrance'''

    def __init__(
        self,
        in_key='waveform',
        sr_key='sample_rate',
        out_key='length',
        frame_length=25,
        frame_shift=10,
        speech_time=8000,
    ):
        '''init.'''
        self.in_key = in_key
        self.sr_key = sr_key
        self.out_key = out_key
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.speech_time = speech_time

    def __call__(self, item, **_kwargs):
        '''get frame length before get fbank'''
        if item is None or self.in_key not in item:
            return None

        sample_rate = item[self.sr_key]
        window_size = int(sample_rate * self.frame_length * MILLISECONDS_TO_SECONDS)
        de_shift = int(1.0 / (self.frame_shift * MILLISECONDS_TO_SECONDS))
        wav_len = self.speech_time * MILLISECONDS_TO_SECONDS * sample_rate
        item[self.out_key] = int((wav_len - window_size) * de_shift // float(sample_rate) + 1)
        return item


@PREPROCESS.register_module()
class DataGeneratorNNbeam:
    """
    nnbeam data generator, aligned with local simulation process
    waveform_label_decouple is True only to totally align with local nnbeam simulation process
    """

    def __init__(
        self,
        wav_key="waveform",
        rir_dir=None,
        diffuse_rir_dir=None,
        disturb_path=None,
        noise_path=None,
        clean_path=None,
        sampling_rate=16000,
        frame_size=160,
        sir_min=-5,
        sir_max=15,
        snr_min=0,
        snr_max=25,
        agc_min=300,
        agc_max=5000,
        disturb_ratio=0.7,
        noise_ratio=0.6,
        rir_mic_num=2,
        rir_length=2048,
        rir_diffuse_length=2048,
        mic_space=0.027,
        use_vad_merge=False,
        waveform_label_decouple=False,
        max_speech_time=8,
        use_orig2ch=True,
        use_orig2ch_ratio=0.10,
        orig2ch_hdfs_path=None,
        fake_long_utterance=False,
        merge_utterance=False,
    ):
        '''init.'''
        # pylint:disable=too-many-locals
        # pylint:disable=too-many-locals
        self.wav_key = wav_key

        self.rir_reader = []
        self.rir_diffuse_reader = []
        self.rir_keys = []
        self.rir_diffuse_keys = []
        for angle in range(5, 185, 10):
            rir_reader_tmp = KVReader(f"{rir_dir}/rir_data{angle}", 1)
            self.rir_reader.append(rir_reader_tmp)
            self.rir_keys.append(rir_reader_tmp.list_keys())
            rir_diffuse_reader_tmp = KVReader(f"{diffuse_rir_dir}/rir_diffuse_data{angle}", 1)
            self.rir_diffuse_reader.append(rir_diffuse_reader_tmp)
            self.rir_diffuse_keys.append(rir_diffuse_reader_tmp.list_keys())

        self.noise_reader = KVReader(noise_path, 1)
        self.noise_keys = self.noise_reader.list_keys()
        self.disturb_reader = KVReader(disturb_path, 1)
        self.disturb_keys = self.disturb_reader.list_keys()
        self.clean_reader = KVReader(clean_path, 1)
        self.clean_keys = self.clean_reader.list_keys()

        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.disturb_ratio = disturb_ratio
        self.noise_ratio = noise_ratio
        self.mic_num = rir_mic_num
        self.rir_length = rir_length
        self.rir_diffuse_length = rir_diffuse_length
        self.sir_max = sir_max
        self.sir_min = sir_min
        self.snr_max = snr_max
        self.snr_min = snr_min
        self.agc_max = agc_max
        self.agc_min = agc_min
        self.mic_space = mic_space
        self.use_vad_merge = use_vad_merge

        self.waveform_label_decouple = waveform_label_decouple
        if self.waveform_label_decouple:
            self.max_speech_time = max_speech_time
            self.max_speech_length = int(max_speech_time * self.sampling_rate)
            self.use_orig2ch = use_orig2ch
            self.use_orig2ch_ratio = use_orig2ch_ratio
            self.orig2ch_hdfs_path = orig2ch_hdfs_path
            self.orig2ch_reader = KVReader(orig2ch_hdfs_path, 1)
            self.orig2ch_keys = self.orig2ch_reader.list_keys()

        self.fake_long_utterance = fake_long_utterance
        self.merge_utterance = merge_utterance
        if self.fake_long_utterance:
            self.max_speech_time = max_speech_time
            self.max_speech_length = int(max_speech_time * self.sampling_rate)

    @staticmethod
    def vad_merge(w):
        '''energy based vad split and merge'''
        intervals = librosa.effects.split(w, top_db=40)
        temp = []
        for s, e in intervals:
            temp.append(w[s:e])
        return np.concatenate(temp, axis=None)

    def get_clean(self):
        '''get clean data'''
        val_len = self.max_speech_length
        values = self.clean_reader.read_many([random.choice(self.clean_keys)])
        temp = np.frombuffer(values[0], dtype=np.int16)
        clean_data = temp.copy()
        if self.use_vad_merge:
            clean_data = self.vad_merge(clean_data.astype(np.float32)).astype(np.int16)
        while len(clean_data) < val_len:
            values = self.clean_reader.read_many([random.choice(self.clean_keys)])
            temp = np.frombuffer(values[0], dtype=np.int16)
            clean_data1 = temp.copy()
            if self.use_vad_merge:
                clean_data1 = self.vad_merge(clean_data1.astype(np.float32)).astype(np.int16)
            clean_data = np.concatenate((clean_data, clean_data1))
        if len(clean_data) >= val_len:
            clean_start = random.randint(0, len(clean_data) - val_len)
            clean_data = clean_data[clean_start : clean_start + val_len]
        clean_data = clean_data.astype(np.float32)
        return clean_data

    def get_orig_2ch(self):
        '''get online original 2-channel data'''
        val_len = self.max_speech_length
        values = self.orig2ch_reader.read_many([random.choice(self.orig2ch_keys)])
        temp = np.frombuffer(values[0], dtype=np.int16).reshape(-1, 2)

        clean_orig_data = temp[1:, :].copy()
        wkp_angle1 = temp[0, 1]
        wkp_angle = wkp_angle1
        while clean_orig_data.shape[0] < val_len:
            values = self.orig2ch_reader.read_many([random.choice(self.orig2ch_keys)])
            temp = np.frombuffer(values[0], dtype=np.int16).reshape(-1, 2)
            wkp_angle2 = temp[0, 1]
            if abs(wkp_angle1 - wkp_angle2) < 5:
                clean_data1 = temp[1:, :].copy()
                clean_orig_data = np.concatenate((clean_orig_data, clean_data1), axis=0)
                wkp_angle = 0.8 * wkp_angle + 0.2 * wkp_angle2

        if clean_orig_data.shape[0] >= val_len:
            clean_start = random.randint(0, clean_orig_data.shape[0] - val_len)
            clean_orig_data = clean_orig_data[clean_start : clean_start + val_len, :]
        clean_orig_data = clean_orig_data.astype(np.float32)
        return clean_orig_data, wkp_angle

    def get_rir(self, wkp_angle=None):
        '''create rir angle and data for clean and disturb'''
        # pylint:disable=too-many-branches
        random_num = random.random()
        if random_num < 0.45:  # 40%
            target_zone_idx = random.choice([8, 9])
        elif random_num < 0.65:
            target_zone_idx = random.choice([7, 10])
        elif random_num < 0.8:  # 30%
            target_zone_idx = random.choice([5, 6, 11, 12])
        elif random_num < 0.9:  # 20%
            target_zone_idx = random.choice([3, 4, 13, 14])
        else:  # 10%
            target_zone_idx = random.choice([0, 1, 2, 15, 16, 17])

        key_tmp = random.choice(self.rir_keys[target_zone_idx])
        target_direction = float(key_tmp)
        value_tmp = self.rir_reader[target_zone_idx].read_many([key_tmp])[0]
        rir_target_data = np.frombuffer(value_tmp, dtype=np.float32).reshape(
            self.mic_num, self.rir_length
        )

        if wkp_angle is None:
            rand_rto = np.clip(np.random.randn() * 0.4, -1, 1)  # -1~1
            target_direction_use = target_direction + rand_rto * 15.0
            target_direction_use = max(min(target_direction_use, 180), 0)
        else:
            target_direction = wkp_angle
            target_direction_use = wkp_angle
        target_zone_use_idx = target_direction_use // 10.0
        while True:
            target_zone_idx2 = random.choice(
                [tmp for tmp in range(18) if abs(tmp - target_zone_use_idx) <= 1]
            )
            key_tmp = random.choice(self.rir_keys[target_zone_idx2])
            target_direction2 = float(key_tmp)
            value_tmp = self.rir_reader[target_zone_idx2].read_many([key_tmp])[0]
            rir_target_data2 = np.frombuffer(value_tmp, dtype=np.float32).reshape(
                self.mic_num, self.rir_length
            )
            angle_diff = abs(target_direction2 - target_direction_use)
            if angle_diff < 20.0:
                break

        while True:
            random_num = random.random()
            if random_num < 0.5:
                angle_zone_idx = random.choice(
                    [
                        tmp
                        for tmp in range(18)
                        if abs(tmp - target_zone_use_idx) >= 2
                        and abs(tmp - target_zone_use_idx) <= 4
                    ]
                )
            elif random_num < 0.85:
                angle_zone_idx = random.choice(
                    [
                        tmp
                        for tmp in range(18)
                        if abs(tmp - target_zone_use_idx) >= 4
                        and abs(tmp - target_zone_use_idx) <= 5
                    ]
                )
            else:
                angle_zone_idx = random.choice(
                    [tmp for tmp in range(18) if abs(tmp - target_zone_use_idx) >= 5]
                )

            key_tmp = random.choice(self.rir_keys[angle_zone_idx])
            disturb_direction = float(key_tmp)
            if abs(disturb_direction - target_direction_use) > 25.0:
                value_tmp = self.rir_reader[angle_zone_idx].read_many([key_tmp])[0]
                rir_disturb_data = np.frombuffer(value_tmp, dtype=np.float32).reshape(
                    self.mic_num, self.rir_length
                )
                break

        return target_direction_use, rir_target_data, rir_target_data2, rir_disturb_data

    @staticmethod
    def get_given_source(source_reader, source_keys, given_length):
        '''get source item according to given source reader and length'''
        values = source_reader.read_many([random.choice(source_keys)])
        value = np.frombuffer(values[0], dtype=np.int16)
        value = value.copy()
        while len(value) < given_length:
            values = source_reader.read_many([random.choice(source_keys)])
            value1 = np.frombuffer(values[0], dtype=np.int16)
            value1 = value1.copy()
            value = np.concatenate((value, value1))
        if len(value) > given_length:
            start = random.randint(0, len(value) - given_length)
            value = value[start : start + given_length]
        return value

    def gen_trunc_diffuse(self, val_len):
        '''get diffuse noise by convolving truncated room inpulse response'''
        noise_data = self.get_given_source(self.disturb_reader, self.disturb_keys, val_len)
        noise_data = noise_data.astype(np.float32)

        noise_zone_idx = random.randint(0, 17)
        key_tmp = random.choice(self.rir_diffuse_keys[noise_zone_idx])
        value_tmp = self.rir_diffuse_reader[noise_zone_idx].read_many([key_tmp])[0]
        rir_noise_data = np.frombuffer(value_tmp, dtype=np.float32).reshape(
            self.mic_num, self.rir_diffuse_length
        )
        noise_data_left = fftconvolve(noise_data, rir_noise_data[0], mode='full')[:val_len]

        noise_zone_idx = random.randint(0, 17)
        key_tmp = random.choice(self.rir_diffuse_keys[noise_zone_idx])
        value_tmp = self.rir_diffuse_reader[noise_zone_idx].read_many([key_tmp])[0]
        rir_noise_data = np.frombuffer(value_tmp, dtype=np.float32).reshape(
            self.mic_num, self.rir_diffuse_length
        )
        noise_data_right = fftconvolve(noise_data, rir_noise_data[1], mode='full')[:val_len]
        noise_data_diffuse = np.stack((noise_data_left, noise_data_right), axis=0)
        return noise_data_diffuse

    def get_diffuse_noise(self, val_len):
        '''get diffuse noise'''
        if random.random() < 0.6:
            # generate diffuse noise according to diffuse matrix
            noise_data_left = self.get_given_source(self.noise_reader, self.noise_keys, val_len)
            noise_data_right = self.get_given_source(self.noise_reader, self.noise_keys, val_len)
            noise_data_orig = np.stack(
                (noise_data_left.astype(np.float32), noise_data_right.astype(np.float32)), axis=-1
            )
            noise_data_diffuse = gen_diffuse(
                noise_data_orig, 16000.0, 2, "linear", self.mic_space, None
            )
            noise_data_diffuse = noise_data_diffuse.transpose()
        else:
            # generate diffuse noise according to truncated room inpulse response
            noise_data_diffuse = self.gen_trunc_diffuse(val_len)
            if random.random() < 0.5:
                noise_data_diffuse2 = self.gen_trunc_diffuse(val_len)
                noise_data_diffuse = noise_data_diffuse + noise_data_diffuse2
            energy_tmp = np.mean(noise_data_diffuse**2, -1)
            noise_data_diffuse[1] *= np.sqrt(energy_tmp[0] / energy_tmp[1])
        return noise_data_diffuse

    def get_disturb(self, val_len):
        '''get disturb data'''
        if random.random() < 0.1:
            cur_reader = self.disturb_reader
            cur_keys = self.disturb_keys
        else:
            cur_reader = self.clean_reader
            cur_keys = self.clean_keys

        values = cur_reader.read_many([random.choice(cur_keys)])
        temp = np.frombuffer(values[0], dtype=np.int16)
        disturb_data = temp.copy()
        if self.use_vad_merge:
            disturb_data = self.vad_merge(disturb_data.astype(np.float32)).astype(np.int16)
        while len(disturb_data) < val_len:
            values = cur_reader.read_many([random.choice(cur_keys)])
            temp = np.frombuffer(values[0], dtype=np.int16)
            disturb_data1 = temp.copy()
            if self.use_vad_merge:
                disturb_data1 = self.vad_merge(disturb_data1.astype(np.float32)).astype(np.int16)
            disturb_data = np.concatenate((disturb_data, disturb_data1))
        if len(disturb_data) >= val_len:
            disturb_start = random.randint(0, len(disturb_data) - val_len)
            disturb_data = disturb_data[disturb_start : disturb_start + val_len]
        disturb_data = disturb_data.astype(np.float32)
        return disturb_data

    @staticmethod
    def addnoise_array_cpu(speech, noise, frame_size, snr_min, snr_max):
        '''add noise with given parameters'''
        speech = speech.astype(np.float32)
        noise = noise.astype(np.float32)
        max_length = speech.shape[1]
        noisy = speech.copy()
        snr = np.random.uniform(snr_min, snr_max)
        n_frames = int(max_length / frame_size)
        speech_t = speech[0, :]
        noise_t = noise[0, :]
        speech_framed = speech_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        noise_framed = noise_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        speech_power = np.max(np.mean((speech_framed**2), axis=1), axis=0)
        noise_power = np.max(np.mean((noise_framed**2), axis=1), axis=0)
        noise_scale = np.sqrt(speech_power / (noise_power + 1e-5) / np.power(10.0, snr / 10.0))
        noise = noise * noise_scale
        noisy = noisy + noise
        return speech, noise, noisy

    @staticmethod
    def add_diffuse(speech, noise, frame_size, snr_min, snr_max):
        '''add diffuse with given parameters'''
        speech_t = speech[0]
        noise_t = noise[0]
        max_length = speech.shape[1]
        n_frames = int(max_length / frame_size)
        snr = np.random.uniform(snr_min, snr_max)
        speech_framed = speech_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        noise_framed = noise_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        speech_power = np.max(np.mean((speech_framed**2), axis=1), axis=0)
        noise_power = np.max(np.mean((noise_framed**2), axis=1), axis=0)
        noise_scale = np.sqrt(speech_power / (noise_power + 1e-5) / np.power(10.0, snr / 10.0))
        noisy = speech + noise * noise_scale
        return noisy

    @staticmethod
    def rand_sync_agc_array_cpu(agc_min, agc_max, data):
        '''get volume scale value and max absolute value'''
        agc_value = np.random.uniform(agc_min, agc_max)
        data_t = data[0, :]
        cur_max = np.abs(data_t.astype(np.float32)).max()
        agc_ratio = agc_value / (cur_max + 1e-5)
        return agc_ratio, cur_max

    def __call__(self, item, **_kwargs):
        '''do data simulation process like local nnbeam.'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements
        if item is None or (self.wav_key not in item):
            return item
        if 'channel_num' not in item or item['channel_num'] != 1:
            return item
        add_disturb_flag = random.random() < self.disturb_ratio
        add_diffuse_flag = random.random() < self.noise_ratio
        if not self.waveform_label_decouple:
            waveform = item[self.wav_key]
            _, org_len = waveform.shape
            clean = waveform.squeeze(0).astype(np.float32)

            target_direction, target_rir, target_rir2, disturb_rir = self.get_rir()

            clean_data1 = fftconvolve(clean, target_rir[0], mode='full')[:org_len]
            clean_data2 = fftconvolve(clean, target_rir[1], mode='full')[:org_len]
            clean_data_add1 = fftconvolve(clean, target_rir2[0], mode='full')[:org_len]
            clean_data_add2 = fftconvolve(clean, target_rir2[1], mode='full')[:org_len]
            clean_data = np.stack((clean_data1, clean_data2), axis=0).astype(np.float32)
            clean_data_add = np.stack((clean_data_add1, clean_data_add2), axis=0).astype(np.float32)
            if random.random() < 0.3:
                clean_data = clean_data_add

            if not self.fake_long_utterance:
                disturb = self.get_disturb(org_len)
                noise_data = self.get_diffuse_noise(org_len)
                if add_disturb_flag:
                    disturb_data1 = fftconvolve(disturb, disturb_rir[0], mode='full')[:org_len]
                    disturb_data2 = fftconvolve(disturb, disturb_rir[1], mode='full')[:org_len]
                    disturb_data = np.stack((disturb_data1, disturb_data2), axis=0).astype(
                        np.float32
                    )
                    clean_data, disturb_data, noisy_data = self.addnoise_array_cpu(
                        clean_data, disturb_data, self.frame_size, self.snr_min, self.snr_max
                    )
                else:
                    noisy_data = clean_data.copy()
                    disturb_data = np.zeros_like(clean_data)

                if add_diffuse_flag and not add_disturb_flag:
                    noisy_data = self.add_diffuse(
                        noisy_data, noise_data, self.frame_size, self.snr_min, self.snr_max
                    )
            elif not self.merge_utterance:
                disturb = self.get_disturb(self.max_speech_length)
                noise_data = self.get_diffuse_noise(self.max_speech_length)
                if org_len < self.max_speech_length:
                    head_padding_size = random.randint(0, self.max_speech_length - org_len)
                    tail_padding_size = self.max_speech_length - org_len - head_padding_size
                    clean_data = np.pad(
                        clean_data, ((0, 0), (head_padding_size, tail_padding_size))
                    )
                elif self.max_speech_length < org_len:
                    head_padding_size = random.randint(0, org_len - self.max_speech_length)
                    tail_padding_size = org_len - self.max_speech_length - head_padding_size

                if add_disturb_flag:
                    disturb_data1 = fftconvolve(disturb, disturb_rir[0], mode='full')[
                        : self.max_speech_length
                    ]
                    disturb_data2 = fftconvolve(disturb, disturb_rir[1], mode='full')[
                        : self.max_speech_length
                    ]
                    disturb_data = np.stack((disturb_data1, disturb_data2), axis=0).astype(
                        np.float32
                    )
                    if self.max_speech_length < org_len:
                        disturb_data = np.pad(
                            disturb_data, ((0, 0), (head_padding_size, tail_padding_size))
                        )
                    clean_data, disturb_data, noisy_data = self.addnoise_array_cpu(
                        clean_data, disturb_data, self.frame_size, self.snr_min, self.snr_max
                    )
                else:
                    noisy_data = clean_data.copy()
                    disturb_data = np.zeros_like(clean_data)

                if add_diffuse_flag and not add_disturb_flag:
                    if self.max_speech_length < org_len:
                        noise_data = np.pad(
                            noise_data, ((0, 0), (head_padding_size, tail_padding_size))
                        )
                    noisy_data = self.add_diffuse(
                        noisy_data, noise_data, self.frame_size, self.snr_min, self.snr_max
                    )
        else:
            if self.use_orig2ch and random.random() < self.use_orig2ch_ratio:
                use_orig2ch_flag = True
            else:
                use_orig2ch_flag = False
            if use_orig2ch_flag:
                clean_orig_data, wkp_angle = self.get_orig_2ch()
                clean_data = clean_orig_data.copy().transpose()
                target_direction, target_rir, target_rir2, disturb_rir = self.get_rir(
                    wkp_angle=wkp_angle
                )
            else:
                clean = self.get_clean()
                target_direction, target_rir, target_rir2, disturb_rir = self.get_rir()
                clean_data1 = fftconvolve(clean, target_rir[0], mode='full')[
                    : self.max_speech_length
                ]
                clean_data2 = fftconvolve(clean, target_rir[1], mode='full')[
                    : self.max_speech_length
                ]
                clean_data_add1 = fftconvolve(clean, target_rir2[0], mode='full')[
                    : self.max_speech_length
                ]
                clean_data_add2 = fftconvolve(clean, target_rir2[1], mode='full')[
                    : self.max_speech_length
                ]
                clean_data = np.stack((clean_data1, clean_data2), axis=0).astype(np.float32)
                clean_data_add = np.stack((clean_data_add1, clean_data_add2), axis=0).astype(
                    np.float32
                )
                if random.random() < 0.4:
                    val_len = self.max_speech_length
                    add_postion = random.randint(val_len // 4, val_len // 4 * 3)
                    clean_data[:, add_postion:] = clean_data_add[:, add_postion:]

            disturb = self.get_disturb(self.max_speech_length)
            noise_data = self.get_diffuse_noise(self.max_speech_length)
            if add_disturb_flag:
                disturb_data1 = fftconvolve(disturb, disturb_rir[0], mode='full')[
                    : self.max_speech_length
                ]
                disturb_data2 = fftconvolve(disturb, disturb_rir[1], mode='full')[
                    : self.max_speech_length
                ]
                disturb_data = np.stack((disturb_data1, disturb_data2), axis=0).astype(np.float32)
                clean_data, disturb_data, noisy_data = self.addnoise_array_cpu(
                    clean_data, disturb_data, self.frame_size, self.snr_min, self.snr_max
                )
            else:
                noisy_data = clean_data.copy()
                disturb_data = np.zeros_like(clean_data)

            if add_diffuse_flag and not add_disturb_flag:
                noisy_data = self.add_diffuse(
                    noisy_data, noise_data, self.frame_size, self.snr_min, self.snr_max
                )

        agc_ratio, _ = self.rand_sync_agc_array_cpu(self.agc_min, self.agc_max, noisy_data)
        clean_data = clean_data * agc_ratio
        disturb_data = disturb_data * agc_ratio
        noisy_data = noisy_data * agc_ratio

        # [0.72~1.40]
        amp_low = 0.707
        amp_high = 1.414
        rand_rto = np.clip(np.random.randn() * 0.2, -1, 1) * 0.5 + 0.5  # 0~1
        amp_use = rand_rto * (amp_high - amp_low) + amp_low
        noisy_data[1] *= amp_use
        noisy_data = np.stack((noisy_data[0], noisy_data[1]), axis=0)

        item["target_waveform"] = clean_data[0]
        item["target_2ch_waveform"] = clean_data.transpose()
        item[self.wav_key] = noisy_data.transpose().reshape(-1)
        item["direction"] = target_direction
        item['channel_num'] = 2

        item[self.wav_key] = np.expand_dims(item[self.wav_key], 0)
        return item
