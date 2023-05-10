"""simulator"""

import random
import pickle
import numpy as np
import librosa
from scipy.signal import fftconvolve
from core.utils import gen_diffuse
from core.utils import DataIter, MultiDatasIter


class Data:
    """Base data"""

    def __init__(self, data_dir, cfg, max_speech_length=None):
        '''init.'''
        self._cfg = cfg
        self._top_db = cfg.general.top_db
        self._use_vad_merge = cfg.general.use_vad_merge
        self._max_speech_length = max_speech_length
        if self._max_speech_length is None:
            self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)
        self.data_dir = data_dir
        self.init_reader()

    def init_reader(self):
        """Init reader

        Normal data only has one reader.
        """
        self.data_iter = DataIter(data_paths=self.data_dir)

    def vad_merge(self, data, top_db=None, out_dtype=None):
        """Vad merge.

        Args:
            data: data to be merged.
            top_db: if None, use config top db.
            out_dtype: dtype of result, if None, is the same as input data dtype.
        """
        if top_db is None:
            top_db = self._top_db
        if out_dtype is None:
            out_dtype = data.dtype
        data = data.astype(np.float32)
        intervals = librosa.effects.split(data, top_db=top_db)
        temp = []
        for s, e in intervals:
            temp.append(data[s:e])
        res = np.concatenate(temp, axis=None)
        return res.astype(out_dtype)

    def read_from_reader(self, out_dtype=np.int16):
        """Read from reader.

        Args:
            reader_idx: index of reader, if None, randomly choose a index.
            key_idx: index of keys, if None, randomly choose a index according to reader_idx.
            out_dtype: dtype of data readed, default np.int16

        Return:
            data readed of dtype out_dtype.
        """
        value_tmp = self.data_iter.get_data()
        value_tmp = bytes(value_tmp)
        return np.frombuffer(value_tmp, dtype=out_dtype).copy()

    def get_data(self, length=None, use_vad_merge=None, out_dtype=np.float32):
        """Get data.

        Args:
            length: length of the data to be got, if None, use max_speech_length.
            use_vad_merge: whther to use vad merge, if None, according to default config.
            out_dtype: dtype of result data, default np.float32.

        Return:
            result data.
        """
        if length is None:
            length = self._max_speech_length
        length = int(length)
        data = self.read_from_reader()
        if use_vad_merge is None:
            use_vad_merge = self._use_vad_merge
        if use_vad_merge:
            data = self.vad_merge(data)
        while len(data) < length:
            nex_data = self.read_from_reader()
            if use_vad_merge:
                nex_data = self.vad_merge(nex_data)
            data = np.concatenate((data, nex_data))
        if len(data) >= length:
            rand_start = random.randint(0, len(data) - length)
            data = data[rand_start : rand_start + length]
        data = data.astype(out_dtype)
        return data


class RirData(Data):
    """Rir data"""

    def read_from_reader(self, target_zone_idx=None):
        """Read from reader.

        Args:
            target_angle: target angle

        Return:
            data readed of dtype out_dtype.
        """
        if target_zone_idx is None:
            target_angle = None
        else:
            target_angle = self.angle_list[target_zone_idx]
        value_tmp = self.data_iter.get_data(target_angle)
        value = pickle.loads(value_tmp)
        directions = value['direction']
        rir = value['rir']
        return rir, directions

    def init_reader(self):
        """Init Reader.

        Rir data has more than one reader, load rir readers according to config.
        """
        rir_cfg = self._cfg.rir
        self.angle_list = []
        for angle in range(rir_cfg.angle_start, rir_cfg.angle_end, rir_cfg.angle_step):
            self.angle_list.append(angle)
        self.data_iter = MultiDatasIter(
            data_root=self.data_dir, file_prefix='cicular_6mic_h_rir_', shard_list=self.angle_list
        )

    def get_target_data(self, wkp_angle=None):
        """Get target rir data.

        Args:
            wkp_angle: wake up angle.
            out_dtype: dtype of target rir data, default np.float32.

        Return:
            target_direction_use: direction of target rir data.
            target_data: target rir data, with shape (mic_num, rir_length).
        """
        rir_cfg = self._cfg.rir
        random_num = random.random()
        for cond in rir_cfg.target_conditions:
            if random_num < cond.prob:
                target_zone_idx = random.choice(cond.choices)
                break
        # target_direction = float(key_idx)
        # target_data = self.read_from_reader(target_zone_idx, key_idx, out_dtype=out_dtype)
        # target_data.reshape(gen_cfg.mic_num, rir_cfg.rir_length)
        rir, directions = self.read_from_reader(target_zone_idx)
        target_direction = directions[0]
        target_data = rir[0]

        if wkp_angle is None:
            rand_rto = np.clip(np.random.randn() * rir_cfg.wkp_angle_variance, -1, 1)
            target_direction_use = target_direction + rand_rto * rir_cfg.wkp_angle_diffuse_degree
            target_direction_use = max(min(target_direction_use, 180), 0)
        else:
            target_direction_use = wkp_angle
        return target_direction_use, target_data

    def get_disturb_data(self, direction):
        """Get disturb rir data according to given direction.

        Args:
            direction: direction of target rir, to generate disturb rir.
            out_dtype: dtype of target rir data, default np.float32.

        Return:
            disturb_data: disturb rir data, with shape (mic_num, rir_length)
        """
        rir_cfg = self._cfg.rir
        gen_cfg = self._cfg.general
        target_zone_use_idx = direction // 10.0
        while True:
            random_num = random.random()
            for cond in rir_cfg.disturb_conditions:
                if random_num < cond.prob:
                    angle_zone_idx = random.choice(
                        [
                            tmp
                            for tmp in range(gen_cfg.rir_num)
                            if abs(tmp - target_zone_use_idx) >= cond.low
                            and abs(tmp - target_zone_use_idx) <= cond.high
                        ]
                    )
                    break
            rir, directions = self.read_from_reader(angle_zone_idx)
            disturb_direction = directions[0]
            disturb_data = rir[0]
            # disturb_direction = float(key_idx)
            angle_diff = abs(disturb_direction - direction)
            if angle_diff > rir_cfg.disturb_angle_diff:
                # disturb_data = self.read_from_reader(angle_zone_idx, key_idx, out_dtype=out_dtype)
                # disturb_data.reshape(gen_cfg.mic_num, rir_cfg.rir_length)
                return disturb_data


class DataManager:
    """Data manager"""

    def __init__(self, cfg) -> None:
        '''init.'''
        self._cfg = cfg
        self._clean = Data(cfg.clean.data_path, cfg)
        self._noise = Data(cfg.noise.data_path, cfg)
        self._disturb = Data(cfg.disturb.data_path, cfg)
        self._directed_rir = RirData(cfg.rir.directed_data_path, cfg)
        self._diffused_rir = RirData(cfg.rir.diffused_data_path, cfg)
        self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)

    def _choose_data(self, data_type, length=None):
        """Choose data according to data type.

        Args:
            data_type: data type to be selected.
            length: length of result data,
                    if None, use max_speech_length as length.

        Return:
            data with length of data type.
        """
        if data_type == "clean":
            return self.load_clean_data(length)
        if data_type == "disturb":
            return self.load_disturb_data(length)
        if data_type == "noise":
            return self.load_noise_data(length)
        raise Exception("DataManager: data type error!")

    def _gen_trunc_diffuse(self, length=None):
        """Generate trunc diffuse.

        Args:
            length: length of trunc diffuse,
                    if None, use max_speech_length.

        Return:
            trunc diffuse after convolution.
        """
        if length is None:
            length = self._max_speech_length
        mic_num = self._cfg.general.mic_num
        noise_data = self.load_disturb_data(length)
        noise_data_diffuse = []
        for idx in range(mic_num):
            rir_noise_data = self.load_diffused_rir()
            data = fftconvolve(noise_data, rir_noise_data[idx], mode='full')[:length]
            noise_data_diffuse.append(data)
        noise_data_diffuse = np.stack(noise_data_diffuse, axis=0)
        return noise_data_diffuse

    def load_target_rir(self, wkp_angle=None):
        """Load target rir.

        Args:
            wkp_angle: wake up angle.

        Return:
            target rir data with shape (mic_num, rir_length)
        """
        return self._directed_rir.get_target_data(wkp_angle)

    def load_disturb_rir(self, direction):
        """Load disturb rir.

        Args:
            direction: direction of target rir.

        Return:
            disturb rir data with shape (mic_num, rir_length)
        """
        return self._directed_rir.get_disturb_data(direction)

    def load_diffused_rir(self):
        """Load diffused rir.

        Return:
            diffused rir data with shape (mic_num, rir_length)
        """
        rir, _ = self._diffused_rir.read_from_reader()
        diffused_rir = rir[0]
        return diffused_rir

    def load_clean_data(self, length=None):
        """Load clean data.

        Args:
            length: length of clean data,
                    if None, use max_speech_length.

        Return:
            clean data of length length_param.
        """
        return self._clean.get_data(length)

    def load_noise_data(self, length=None):
        """Load noise data.

        Args:
            length: length of noise data,
                    if None, use max_speech_length.

        Return:
            noise data of length length_param.
        """
        return self._noise.get_data(length, use_vad_merge=False)

    def load_disturb_data(self, length=None):
        """Load disturb data.

        Args:
            length: length of noise data,
                    if None, use max_speech_length.

        Return:
            disturb data of length length_param.
        """
        return self._disturb.get_data(length)

    def load_disturb(self, length=None):
        """Load disturb according to config.

        Args:
            length: length of noise data,
                    if None, use max_speech_length.

        Return:
            data of length length_param according to config.
        """
        dst_cfg = self._cfg.disturb
        random_num = random.random()
        for cond in dst_cfg.conditions:
            if random_num < cond.prob:
                return self._choose_data(cond.datatype, length)
        raise Exception("DataManager: load disturb error!")

    def load_noise(self, length=None):
        """Load noise according to config.

        Args:
            length: length of noise data,
                    if None, use max_speech_length.

        Return:
            diffused noise data.
        """
        prob = self._cfg.noise.choose_prob
        gen_cfg = self._cfg.general
        mic_num = gen_cfg.mic_num
        if random.random() < prob:
            noise_data = []
            for _ in range(mic_num):
                noise_data.append(self.load_noise_data(length))
            noise_data_orig = np.stack(noise_data, axis=-1)
            noise_data_diffuse = gen_diffuse(
                noise_data_orig,
                gen_cfg.sampling_rate,
                gen_cfg.mic_num,
                gen_cfg.array_type,
                gen_cfg.mic_distance,
                gen_cfg.radius,
            )
            noise_data_diffuse = noise_data_diffuse.transpose()
        else:
            noise_data_diffuse = self._gen_trunc_diffuse(length)
            if random.random() < 0.5:
                noise_data_diffuse2 = self._gen_trunc_diffuse(length)
                noise_data_diffuse = noise_data_diffuse + noise_data_diffuse2
            energy_tmp = np.mean(noise_data_diffuse**2, -1)
            noise_data_diffuse[1] *= np.sqrt(energy_tmp[0] / energy_tmp[1])  # ?
        return noise_data_diffuse


class Simulator:
    """Simulator"""

    def __init__(self, cfg):
        '''init.'''
        self._cfg = cfg
        self._data_manager = DataManager(cfg)
        self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)

    def set_random_seed(self, seed=None):
        """Set random seed.

        Args:
            seed: random seed, if None, use config seed.
        """
        if seed is None:
            seed = self._cfg.general.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def mix(self, speech, noise, snr_min, snr_max):
        """Mix noise into speech.

        Args:
            sppech: speech to be mixed.
            noise: noise to be mixed.
            snr_min: minimum value of snr/sir.
            snr_max: maximum value of snr/sir.

        Return:
            noise: scaled noise.
            noisy: product after mixing.
        """
        gen_cfg = self._cfg.general
        frame_size = gen_cfg.frame_size
        speech = speech.astype(np.float32)
        noise = noise.astype(np.float32)
        speech_t = speech[0]
        noise_t = noise[0]
        max_length = speech.shape[1]
        n_frames = int(max_length / frame_size)
        snr = np.random.uniform(snr_min, snr_max)
        speech_framed = speech_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        noise_framed = noise_t[: n_frames * frame_size].reshape(n_frames, frame_size)
        speech_power = np.max(np.mean((speech_framed**2), axis=1), axis=0)
        noise_power = np.max(np.mean((noise_framed**2), axis=1), axis=0)
        noise_scale = np.sqrt(
            speech_power / (noise_power + gen_cfg.min_limit) / np.power(10.0, snr / 10.0)
        )
        noise = noise * noise_scale
        noisy = speech + noise
        return noise, noisy

    def rand_sync_agc_array_cpu(self, data):
        """Calculate agc ratio"""
        gen_cfg = self._cfg.general
        agc_value = np.random.uniform(gen_cfg.agc_min, gen_cfg.agc_max)
        data_t = data[0, :]
        cur_max = np.abs(data_t.astype(np.float32)).max()
        agc_ratio = agc_value / (cur_max + gen_cfg.min_limit)
        return agc_ratio

    def conv_target_audio(self, org_data, rir_data, length=None):
        """Convolve to generate target audio.

        Convolve org_data and rir_data to generate target audio.

        Args:
            org_data: original data to be convolved.
            rir_data: rir data to be convolved.
            length: length of the result after convolution,
                    if length is None, use max_speech_length as length.

        Return:
            target_data: result of convolution.
        """
        if length is None:
            length = self._max_speech_length
        mic_num = self._cfg.general.mic_num
        target_data = []
        for idx in range(mic_num):
            data = fftconvolve(org_data, rir_data[idx], mode='full')[:length]
            target_data.append(data)
        target_data = np.stack(target_data, axis=0).astype(np.float32)
        return target_data

    def generate_target_audio(self, item, wav_key=None):
        """Generate target audio.

        Args:
            item: input dict.

        Return:
            clean_data: target audio generated.
            data_len: length of targe audio generated.
            direction: direction of rir data used to generate target audio.
        """
        gen_cfg = self._cfg.general
        if wav_key is None:
            wav_key = gen_cfg.wave_key
        if gen_cfg.waveform_label_decouple:
            data_len = self._max_speech_length
            clean = self._data_manager.load_clean_data()
            rir_direction, rir_data = self._data_manager.load_target_rir()
            clean_data = self.conv_target_audio(clean, rir_data, length=data_len)
        else:
            waveform = item[wav_key]
            _, data_len = waveform.shape
            clean = waveform.squeeze(0).astype(np.float32)
            rir_direction, rir_data = self._data_manager.load_target_rir()
            clean_data = self.conv_target_audio(clean, rir_data, length=data_len)
        return clean_data, data_len, rir_direction

    def add_noise(self, target_data, length, rir_direction):
        """Add noise to target audio.

        Args:
            target_data: target audio, data to be noised.
            length: length of clean data.
            rir_direction: direction of rir data used to generate clean

        Return:
            target_data: original target data.
            disturb_data: disturb data for noise.
            noisy_data: data after adding noise.
        """
        gen_cfg = self._cfg.general
        and_flag = gen_cfg.noise_disturb_and_flag
        add_disturb_flag = random.random() < gen_cfg.disturb_ratio
        add_noise_flag = random.random() < gen_cfg.noise_ratio

        disturb_data = np.zeros_like(target_data)
        noisy_data = target_data.copy()

        if add_disturb_flag:
            disturb = self._data_manager.load_disturb(length=length)
            disturb_rir = self._data_manager.load_disturb_rir(rir_direction)
            disturb_data = self.conv_target_audio(disturb, disturb_rir, length=length)
            disturb_data, noisy_data = self.mix(
                target_data, disturb_data, gen_cfg.sir_min, gen_cfg.sir_max
            )

        if add_noise_flag and add_disturb_flag and and_flag:
            noise_data = self._data_manager.load_noise(length=length)
            _, noisy_data = self.mix(noisy_data, noise_data, gen_cfg.snr_min, gen_cfg.snr_max)

        if add_noise_flag and not add_disturb_flag:
            noise_data = self._data_manager.load_noise(length=length)
            _, noisy_data = self.mix(target_data, noise_data, gen_cfg.snr_min, gen_cfg.snr_max)

        return target_data, disturb_data, noisy_data

    def randomize_volume(self, clean_data, disturb_data, noisy_data):
        """Randomize volume.

        Args:
            clean_data: target audio data.
            disturb_data: distubr noise data.
            noisy_data: clean data after adding disturb data.

        Return:
            clean_data: clean data after volume randomization.
            disturb_data: disturb data after volume randomization.
            noisy_data: noisy data after volume randomization.
        """
        agc_ratio = self.rand_sync_agc_array_cpu(noisy_data)
        clean_data = clean_data * agc_ratio
        disturb_data = disturb_data * agc_ratio
        noisy_data = noisy_data * agc_ratio

        vol_cfg = self._cfg.volume_rand
        mic_num = self._cfg.general.mic_num
        amp_low = vol_cfg.amp_low
        amp_high = vol_cfg.amp_high

        for idx in range(1, mic_num):
            rand_rto = np.clip(np.random.randn() * vol_cfg.variance, -1, 1) * 0.5 + 0.5
            amp_use = rand_rto * (amp_high - amp_low) + amp_low
            noisy_data[idx] *= amp_use
        noisy_data = np.stack(noisy_data, axis=0)
        return clean_data, disturb_data, noisy_data

    def simulate(
        self,
        item,
        wav_key='waveform',
        out_wav_key='mc_waveform',
        target_waveform_key='target_waveform',
        dir_key='direction',
        disturb_key='disturb_waveform',
    ):
        """Simulator entry.

        Args:
            item: input dict.

        Return:
            item: include data simulated.
        """
        self.set_random_seed()
        clean_data, length, direction = self.generate_target_audio(item, wav_key)
        clean_data, disturb_data, noisy_data = self.add_noise(clean_data, length, direction)
        clean_data, disturb_data, noisy_data = self.randomize_volume(
            clean_data, disturb_data, noisy_data
        )

        gen_cfg = self._cfg.general
        item[target_waveform_key] = clean_data
        item[disturb_key] = disturb_data
        item[out_wav_key] = noisy_data.reshape(-1)
        item[dir_key] = direction
        item["simu_channel_num"] = gen_cfg.mic_num
        item[out_wav_key] = np.expand_dims(item[out_wav_key], 0)
        return item
