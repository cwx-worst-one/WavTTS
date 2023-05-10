"""simulator"""

import random
import pickle
import librosa
import numpy as np
import colorednoise as cn
from scipy.signal import fftconvolve
from core.utils import gen_diffuse, DataIter, MultiDatasIter

# pylint: disable=consider-using-enumerate
# pylint:disable=too-many-lines


class Data:
    """Base data"""

    def __init__(self, data_paths=None, cfg=None, max_speech_length=None):
        '''init.'''
        self._cfg = cfg
        self._top_db = cfg.general.top_db
        self._max_speech_length = max_speech_length
        if self._max_speech_length is None:
            self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)
        if data_paths is None:
            self.data_reader = None
        else:
            self.data_iter = DataIter(data_paths=data_paths, shuffle=True)

    def read_from_reader(self, out_dtype=np.int16):
        """Read from reader.
        Args:
            out_dtype: dtype of data readed, default np.int16

        Return:
            data readed of dtype out_dtype.
        """
        data = self.data_iter.get_data()
        return np.frombuffer(bytes(data), dtype=out_dtype).copy()

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

    def get_data(
        self,
        outside_data=None,
        length=None,
        use_vad_merge=None,
        in_dtype=np.int16,
        out_dtype=np.float32,
    ):
        """Get data.

        Args:
            outside_data: additional data to concate
            length: length of the data to be got, if None, use max_speech_length.
            use_vad_merge: whther to use vad merge, if None, according to default config.
            in_dtype: dtype of outside_data, default np.int16
            out_dtype: dtype of result data, default np.float32.

        Return:
            result data.
        """
        if length is None:
            length = self._max_speech_length
        length = int(length)
        if outside_data is not None:
            assert (
                outside_data.dtype == in_dtype
            ), 'datatype of outside_data and data does not match!'
            data = outside_data
        else:
            data = self.read_from_reader()
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
        return data, length


class RirData(Data):
    """Rir data"""

    def __init__(
        self,
        data_dir=None,
        prefix=None,
        cfg=None,
        max_speech_length=None,
        length=None,
        unserialize_type='pickle',
    ):
        super().__init__(cfg=cfg, max_speech_length=max_speech_length)
        self.prefix = prefix
        self.length = length
        self.unserialize_type = unserialize_type
        rir_cfg = self._cfg.rir
        self.angle_list = list(range(rir_cfg.angle_start, rir_cfg.angle_end, rir_cfg.angle_step))
        self.zone_list = [int(angle // rir_cfg.angle_step) for angle in self.angle_list]
        self.rir_num = len(self.angle_list)
        self.data_iter = MultiDatasIter(
            data_root=data_dir,
            file_prefix=prefix,
            shard_list=self.angle_list,
            mem_shared=False,
            shuffle=True,
        )

    def read_from_reader(self, target_angle=None):
        """Read from reader.

        Args:
            target_angle: target angle want get

        Return:
            data readed of dtype out_dtype.
        """
        value_tmp = self.data_iter.get_data(target_angle)
        if self.unserialize_type == 'pickle':
            value = pickle.loads(value_tmp)
            directions = value['direction']
            rir = value['rir']
        elif self.unserialize_type == 'numpy':
            value_tmp = pickle.loads(value_tmp)
            value = np.frombuffer(bytes(value_tmp['rir']), dtype=np.float32).reshape(
                -1, self.length
            )
            direction = float(value_tmp['direction'])
            directions = [direction]
            rir = [value]
        else:
            raise ValueError("Unknown unserialize type!")
        return rir, directions

    def zone_dis(self, a, b):
        """calculate zone distance between two vals"""

        if self._cfg.general.array_type == "circular":
            dis = min((a - b) % self.rir_num, (b - a) % self.rir_num)
        elif self._cfg.general.array_type == 'linear':
            dis = abs(a - b)
        return dis

    def angle_dis(self, a, b):
        """calculate angle distance between two vals"""

        rir_cfg = self._cfg.rir
        if self._cfg.general.array_type == "circular":
            dis = min((a - b) % rir_cfg.angle_max, (b - a) % rir_cfg.angle_max)
        elif self._cfg.general.array_type == 'linear':
            dis = abs(a - b)
        return dis

    def get_rir_withoutref(self, wkp_angle=None, cfg_key=None, src_idx=0):
        """Get rir data without angle condition.

        Args:
            wkp_angle: wake up angle.
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module

        Return:
            target_direction: real direction which matches target_data
            target_direction_use: direction of target rir data with a perturbation.
            target_data: target rir data, with shape (mic_num, rir_length).
        """
        rir_cfg = self._cfg.rir
        if wkp_angle is None:
            rir_condition = self._cfg[cfg_key]['rir_condition']
            if isinstance(rir_condition, list):
                rir_condition = rir_condition[src_idx]
            else:
                assert isinstance(rir_condition, dict)
            if len(rir_condition['direction_cfg']) == 0:
                target_shard = random.choice(self.angle_list)
            else:
                random_num = random.random()
                for cond in rir_condition['direction_cfg']:
                    if random_num < cond['prob']:
                        target_zone_idx = random.choice(cond.choices)
                        target_shard = self.angle_list[target_zone_idx]
                        break
            rir, directions = self.read_from_reader(target_shard)
            target_direction = directions[0]
            target_data = rir[0]
            wkp_angle_variance = rir_condition.get('angle_variance', 0)
            wkp_angle_diffuse_degree = rir_condition.get('angle_diffuse_degree', 0)
            rand_rto = np.clip(np.random.randn() * wkp_angle_variance, -1, 1)
            target_direction_use = target_direction + rand_rto * wkp_angle_diffuse_degree
            target_direction_use = max(
                min(target_direction_use, rir_cfg.angle_max), rir_cfg.angle_min
            )
        else:
            target_direction_use = wkp_angle
            target_data = None
        return target_direction, target_direction_use, target_data

    def get_rir_withref(self, ref_direction, cfg_key=None, src_idx=0):
        """Get rir data with angle constraint.

        Args:
            ref_direction: reference angle
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module

        Return:
            disturb direction: direction of rir data.
            disturb_data: rir data, with shape (mic_num, rir_length).
        """
        rir_cfg = self._cfg.rir
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        target_zone_use_idx = (
            ref_direction[0] // rir_cfg.angle_step
        )  # TODO: ref_direction has more than one item
        target_direction = ref_direction[0]
        min_angle2ref = rir_condition.get('min_angle2ref', -0.1)
        max_angle2ref = rir_condition.get('max_angle2ref', 360.1)
        while True:
            random_num = random.random()
            for cond in rir_condition['direction_cfg']:
                if random_num < cond['prob']:
                    candidates = [
                        tmp
                        for tmp in range(self.rir_num)
                        if self.zone_dis(tmp, target_zone_use_idx) >= cond.low
                        and self.zone_dis(tmp, target_zone_use_idx) <= cond.high
                    ]
                    angle_zone_idx = random.choice(candidates)
                    break

            target_shard = self.angle_list[angle_zone_idx]
            rir, directions = self.read_from_reader(target_shard)
            disturb_direction = directions[0]
            disturb_data = rir[0]
            angle_diff = self.angle_dis(disturb_direction, target_direction)
            if min_angle2ref < angle_diff < max_angle2ref:
                break
        return disturb_direction, disturb_data


class DataManager:
    """Data manager"""

    def __init__(self, cfg) -> None:
        '''init.'''
        self._cfg = cfg
        self._audio_data = dict()  #
        self._rir_data = dict()
        for data in cfg.audio_data:
            self._audio_data[data['key']] = Data(data['path'], cfg)
        for data in cfg.rir_data:
            unserialize_type = data.get('unserialize_type', 'pickle')
            length = data.get('length', None)
            self._rir_data[data['key']] = RirData(
                data_dir=data['path'],
                prefix=data['prefix'],
                cfg=cfg,
                length=length,
                unserialize_type=unserialize_type,
            )
        self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)

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

    @staticmethod
    def gen_colornoise(beta_min=1.4, beta_max=2.0, shape=None, out_dtype=np.float32):
        """generate colored noise

        github: https://github.com/felixpatzelt/colorednoise

        Args:
            beta_min: minimum exponents
            beta_max: maximum exponents
            shape: Tuple of colornoise shape
            out_dtype: output datatype

        Return:
            wnoise: colored noise with shape

        """
        beta = random.uniform(beta_min, beta_max)
        wnoise = cn.powerlaw_psd_gaussian(beta, shape)
        wnoise = wnoise.astype(out_dtype)
        return wnoise

    def gen_trunc_diffuse(self, rir_key=None, cfg_key=None, src_idx=0, length=None):
        """Generate trunc diffuse.

        Args:
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            trunc diffuse after convolution.
        """
        mic_num = self._cfg.general.mic_num
        noise_data, length = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
        noise_data_diffuse = []
        for idx in range(mic_num):
            _, _, rir_noise_data = self.load_rir_withoutref(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=0
            )
            data = fftconvolve(noise_data, rir_noise_data[idx], mode='full')[:length]
            noise_data_diffuse.append(data)
        noise_data_diffuse = np.stack(noise_data_diffuse, axis=0)
        return noise_data_diffuse

    def load_rir_withoutref(self, wkp_angle=None, rir_key=None, cfg_key=None, src_idx=0):
        """Load rir without angle constraint.

        Args:
            wkp_angle: wake up angle.
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module

        """
        if rir_key is not None:
            return self._rir_data[rir_key].get_rir_withoutref(
                wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
            )
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                return self._rir_data[cond['key']].get_rir_withoutref(
                    wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
                )
        cond = rir_condition['path_condition'][-1]
        return self._rir_data[cond['key']].get_rir_withoutref(
            wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
        )

    def load_rir_withref(self, ref_direction, rir_key=None, cfg_key=None, src_idx=0):
        """Load rir with angle constraint.

        Args:
            ref_direction: reference angle
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module

        """
        assert ref_direction is not None
        if rir_key is not None:
            return self._rir_data[rir_key].get_rir_withref(
                ref_direction, cfg_key=cfg_key, src_idx=src_idx
            )
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                return self._rir_data[cond['key']].get_rir_withref(
                    ref_direction, cfg_key=cfg_key, src_idx=src_idx
                )
        cond = rir_condition['path_condition'][-1]
        return self._rir_data[cond['key']].get_rir_withref(
            ref_direction, cfg_key=cfg_key, src_idx=src_idx
        )

    def load_audio_data(self, outside_data=None, cfg_key=None, src_idx=0, length=None):
        """Load audio data.

        Args:
            outside_data: additional data to concate
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        """
        src_condition = self._cfg[cfg_key]['src_condition']
        if isinstance(src_condition, list):
            src_condition = src_condition[src_idx]
        else:
            assert isinstance(src_condition, dict)
        use_vad_merge = src_condition.get('use_vad_merge', False)
        path_condition = src_condition.get('path_condition', None)
        in_dtype = eval(src_condition.get('in_dtype', 'np.int16'))
        out_dtype = eval(src_condition.get('out_dtype', 'np.float32'))
        if length is None:
            length = src_condition.get('length', self._max_speech_length)
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)
        if path_condition is None:
            raise ValueError("path condition must be assigned in config!")
        random_num = random.random()
        for cond in path_condition:
            if random_num < cond['prob']:
                return self._audio_data[cond['key']].get_data(
                    outside_data=outside_data,
                    length=length,
                    use_vad_merge=use_vad_merge,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                )
        cond = path_condition[-1]
        return self._audio_data[cond['key']].get_data(
            outside_data=outside_data,
            length=length,
            use_vad_merge=use_vad_merge,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )

    def generate_audio_use_direct_rir(
        self,
        item=None,
        wav_key=None,
        ref_direction=None,
        waveform_decouple_type='total',
        rir_type=None,
        rir_key=None,
        cfg_key=None,
        src_idx=0,
        length=None,
    ):

        """generate audio data with directional rir

        Args:
            item: dict contains outside data
            wav_key: item key of outside data
            ref_direction: [optional] direction of reference angle when generating rir
            waveform_decouple_type: whether to use outside data or not or partially
            rir_type: whether to generate rir with constraint(withref) or not(withoutref)
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                mc_data: generated multichannel audio (mch, length)
                mono_data: audio before convolution
                rir_data: rir data
                rir_direction: rir direction
                rir_direction_perturb: rir direction with perturbation
                length: generated audio length
        """
        # pylint:disable=too-many-branches
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)
        # get audio data
        if waveform_decouple_type == "partial":
            waveform = item[wav_key]
            waveform = waveform.squeeze()
            if not waveform.ndim == 1:
                waveform_axis = self._cfg.general.get('waveform_decouple_partial_axis', 0)
                waveform = waveform[waveform_axis]
            clean, length = self.load_audio_data(
                outside_data=waveform, cfg_key=cfg_key, src_idx=src_idx, length=length
            )
        elif waveform_decouple_type == 'pass':
            waveform = item[wav_key]
            if waveform.ndim == 1:
                data_len = waveform.shape[0]
            elif waveform.ndim == 2:
                data_len = max(waveform.shape)
            clean = waveform.squeeze().astype(np.float32)
            length = data_len
        elif waveform_decouple_type == 'total':
            clean, length = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
        else:
            raise ValueError('Unrecognized waveform_decouple_type!')
        # get rir data
        if rir_type == 'withoutref':
            rir_direction, rir_direction_use, rir_data = self.load_rir_withoutref(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx
            )
        elif rir_type == 'withref':
            rir_direction, rir_data = self.load_rir_withref(
                ref_direction,
                rir_key=rir_key,
                cfg_key=cfg_key,
                src_idx=src_idx,
            )
            rir_direction_use = rir_direction
        # convolve
        if rir_data is not None:
            clean_data = self.conv_target_audio(clean, rir_data, length=length)
        else:
            clean_data = clean
        outdict = dict(
            mc_data=clean_data,
            mono_data=clean,
            rir_data=rir_data,
            rir_direction=rir_direction,
            rir_direction_perturb=rir_direction_use,
            length=length,
        )
        return outdict

    def generate_audio_use_diffuse_rir(self, rir_key=None, cfg_key=None, src_idx=0, length=None):
        """generate convolved audio data with diffuse rir

        Args:
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                mc_data: generated multichannel audio (mch, length)
                length: generated audio length
        """
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)
        noise_data_diffuse = self.gen_trunc_diffuse(
            rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx, length=length
        )
        if random.random() < 0.5:
            noise_data_diffuse2 = self.gen_trunc_diffuse(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx, length=length
            )
            noise_data_diffuse = noise_data_diffuse + noise_data_diffuse2
        energy_tmp = np.mean(noise_data_diffuse**2, -1) + 1e-8
        noise_data_diffuse[1] *= np.sqrt(energy_tmp[0] / energy_tmp[1])  # ?
        outdict = dict(mc_data=noise_data_diffuse, length=noise_data_diffuse.shape[-1])
        return outdict

    def generate_audio_use_diffuse_equ(self, cfg_key=None, src_idx=0, length=None):
        """generate audio data use diffuse equation

        Args:
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                mc_data: generated multichannel audio (mch, length)
                length: generated audio length
        """
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)
        gen_cfg = self._cfg.general
        noise_data = []
        for _ in range(gen_cfg.mic_num):
            tmp, length = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
            noise_data.append(tmp)
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
        outdict = dict(mc_data=noise_data_diffuse, length=noise_data_diffuse.shape[-1])
        return outdict

    def get_one_audio(
        self,
        item=None,
        wav_key=None,
        ref_direction=None,
        waveform_decouple_type='total',
        length=None,
        cfg_key=None,
        src_idx=0,
    ):
        """Get multichannel audio.

        Args:
            item: [optional] dict contains outside data
            wav_key: [optional] item key of outside data
            ref_direction: [optional] direction of reference angle when generating rir
            waveform_decouple_type: whether to use outside data or not or partially
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        """
        gen_cfg = self._cfg.general
        if wav_key is None:
            wav_key = gen_cfg.wave_key
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                if cond['key'] == "diffuse_equation":
                    return self.generate_audio_use_diffuse_equ(
                        cfg_key=cfg_key, src_idx=src_idx, length=length
                    )
                if cond['key'] == 'diffuse':
                    rir_key = cond['key']
                    return self.generate_audio_use_diffuse_rir(
                        rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx, length=length
                    )
                if cond['key'] == 'direct':
                    rir_key = cond['key']
                    rir_type = rir_condition['type']
                    return self.generate_audio_use_direct_rir(
                        item=item,
                        wav_key=wav_key,
                        ref_direction=ref_direction,
                        waveform_decouple_type=waveform_decouple_type,
                        rir_key=rir_key,
                        rir_type=rir_type,
                        cfg_key=cfg_key,
                        src_idx=src_idx,
                        length=length,
                    )
        return 0

    def get_colornoise(self, cfg_key='colornoise', src_idx=0, length=None):
        """Get colored noise.

        Args:
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                mc_data: generated multichannel audio (mch, length)
                rir_data: rir data
                rir_direction: rir direction
                length: generated audio length
        """
        cfg = self._cfg[cfg_key]
        beta_min = cfg.get('beta_min', 1.4)
        beta_max = cfg.get('beta_max', 2.0)
        if length is None:
            length = cfg.get('length', self._max_speech_length)
        if length is None:
            length = self._max_speech_length
        elif isinstance(length, str):
            length = eval(length)
        else:
            length = int(length)
        if cfg['type'] == "uncorrelated":
            mic_num = self._cfg.general.mic_num
            clean_data = self.gen_colornoise(beta_min, beta_max, shape=(mic_num, length))
            rir_direction = None
            rir_data = None
        elif cfg['type'] == "correlated":
            wnoise = self.gen_colornoise(beta_min, beta_max, shape=(length))
            rir_direction, _, rir_data = self.load_rir_withoutref(
                rir_key='direct', cfg_key=cfg_key, src_idx=src_idx
            )
            clean_data = self.conv_target_audio(wnoise, rir_data, length=length)
        outdict = dict(
            mc_data=clean_data,
            rir_data=rir_data,
            rir_direction=rir_direction,
            length=clean_data.shape[-1],
        )
        return outdict


class ModuleSimulator:
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
            snr: snr
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
        return noise, noisy, snr

    def rand_sync_agc_array_cpu(self, data):
        """Calculate agc ratio"""
        gen_cfg = self._cfg.general
        agc_value = np.random.uniform(gen_cfg.agc_min, gen_cfg.agc_max)
        data_t = data[0, :]
        cur_max = np.abs(data_t.astype(np.float32)).max()
        agc_ratio = agc_value / (cur_max + gen_cfg.min_limit)
        return agc_ratio, agc_value

    def randomize_volume(self, noisy_data, *args):
        """Randomize volume.

        Args:
            noisy_data: noisy data.
            args: tuple of data to adjust volume

        Return:
            noisy_data: noisy data after volume randomization.
            args: tuple of data after adjusting volume in the same order
            agc_val: max sample leval
        """
        agc_ratio, agc_val = self.rand_sync_agc_array_cpu(noisy_data)
        noisy_data = noisy_data * agc_ratio
        args = list(args)
        for i in range(len(args)):
            args[i] *= agc_ratio

        vol_cfg = self._cfg.volume_rand
        mic_num = self._cfg.general.mic_num
        amp_low = vol_cfg.amp_low
        amp_high = vol_cfg.amp_high

        rand_rto = (
            np.clip(np.random.randn(mic_num).reshape(-1, 1) * vol_cfg.variance, -1, 1) * 0.5 + 0.5
        )
        amp_use = rand_rto * (amp_high - amp_low) + amp_low
        noisy_data *= amp_use
        return noisy_data, args, agc_val

    def parse_block(self, cfg_key):
        """parse signal module"""
        cfg = self._cfg[cfg_key]
        if isinstance(cfg, dict):
            return cfg_key
        while isinstance(cfg, list):
            random_num = random.random()
            for cond in cfg:
                if random_num < cond['prob']:
                    cfg = cond['key']
                    cfg_key = cfg
                    break
        if isinstance(cfg, str):
            return cfg_key
        raise ValueError("block type must be string!")

    def parse_src_num(self, cfg_key):
        """parse source number inside a signal module"""
        src_num = self._cfg[cfg_key].get('src_num', 1)
        if isinstance(src_num, list):
            random_num = random.random()
            for cond in src_num:
                if random_num < cond['prob']:
                    src_num = cond['val']
                    break
        src_num = int(src_num)
        assert isinstance(src_num, int)
        return src_num

    def simulate(self, item, wav_key='waveform'):
        """Simulator entry for joint training.

        Args:
            item: [optional] dict contains outside data
            wav_key: [optional] item key of outside data

        Return:
            item: include data simulated.
        """
        # self.set_random_seed()
        gen_cfg = self._cfg.general
        noise_disturb_and_flag = gen_cfg.noise_disturb_and_flag
        disturb_flag = random.random() < gen_cfg.disturb_ratio
        if noise_disturb_and_flag:
            noise_flag = random.random() < gen_cfg.noise_ratio
        else:
            noise_flag = (random.random() < gen_cfg.noise_ratio) and (not disturb_flag)

        cfg_key = self.parse_block('clean')
        waveform_decouple_type = gen_cfg.get('waveform_decouple_type', 'total')
        out_dict = self._data_manager.get_one_audio(
            item, wav_key, waveform_decouple_type=waveform_decouple_type, cfg_key=cfg_key
        )
        clean_data = out_dict['mc_data']
        length = out_dict['length']
        clean_direction = out_dict['rir_direction']

        noisy_data = clean_data.copy()
        disturb_data = np.zeros_like(clean_data)
        if disturb_flag:
            cfg_key = self.parse_block('disturb')
            out_dict = self._data_manager.get_one_audio(
                ref_direction=[clean_direction], cfg_key=cfg_key, length=length
            )
            disturb_data = out_dict['mc_data']
            length = out_dict['length']
            disturb_data, noisy_data, _ = self.mix(
                noisy_data,
                disturb_data,
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
        if noise_flag:
            cfg_key = self.parse_block('noise')
            out_dict = self._data_manager.get_one_audio(cfg_key=cfg_key, length=length)
            noise_data = out_dict['mc_data']
            length = out_dict['length']
            noise_data, noisy_data, _ = self.mix(
                noisy_data, noise_data, self._cfg[cfg_key]['snr_min'], self._cfg[cfg_key]['snr_max']
            )
        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data,
            clean_data,
            disturb_data,
        )
        item['target_waveform'] = clean_data
        item['disturb_waveform'] = disturb_data
        item['mc_waveform'] = noisy_data.reshape(-1)
        item['direction'] = clean_direction
        item["simu_channel_num"] = gen_cfg.mic_num
        item['mc_waveform'] = np.expand_dims(item['mc_waveform'], 0)
        return item

    def simulate_ssl(self, item, wav_key='waveform'):
        """Simulator entry for ssl training.

        Args:
            item: [optional] dict contains outside data
            wav_key: [optional] item key of outside data

        Return:
            item: include data simulated.
        """
        gen_cfg = self._cfg.general
        noise_disturb_and_flag = gen_cfg.noise_disturb_and_flag
        disturb_flag = random.random() < gen_cfg.disturb_ratio
        if noise_disturb_and_flag:
            noise_flag = random.random() < gen_cfg.noise_ratio
        else:
            noise_flag = (not disturb_flag) and (random.random() < gen_cfg.noise_ratio)
        wnoise_flag = random.random() < gen_cfg.colornoise_ratio

        src_num_total = 0
        # generate src1
        cfg_key = self.parse_block('clean')
        src_num_total += self.parse_src_num(cfg_key)
        waveform_decouple_type = gen_cfg.get('waveform_decouple_type', 'total')
        out_dict = self._data_manager.get_one_audio(
            item,
            wav_key,
            waveform_decouple_type=waveform_decouple_type,
            cfg_key=cfg_key,
        )
        clean_data = out_dict['mc_data']
        length = out_dict['length']
        clean_direction = out_dict['rir_direction']

        noisy_data = clean_data.copy()
        disturb_data = np.zeros_like(clean_data)
        noise_data = np.zeros_like(clean_data)
        wnoise_data = np.zeros_like(clean_data)
        disturb_direction = -1
        snr_list = [-1, -1, -1]
        # generate src2
        if disturb_flag:
            cfg_key = self.parse_block('disturb')
            src_num_total += self.parse_src_num(cfg_key)
            out_dict = self._data_manager.get_one_audio(
                ref_direction=[clean_direction], cfg_key=cfg_key, length=length
            )
            disturb_data = out_dict['mc_data']
            length = out_dict['length']
            disturb_direction = out_dict['rir_direction']
            disturb_data, noisy_data, disturb_snr = self.mix(
                noisy_data,
                disturb_data,
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            snr_list[0] = disturb_snr
        # generate noise
        if noise_flag:
            cfg_key = self.parse_block('noise')
            out_dict = self._data_manager.get_one_audio(
                ref_direction=[clean_direction], cfg_key=cfg_key, length=length
            )
            noise_data = out_dict['mc_data']
            length = out_dict['length']
            noise_data, noisy_data, noise_snr = self.mix(
                noisy_data, noise_data, self._cfg[cfg_key]['snr_min'], self._cfg[cfg_key]['snr_max']
            )
            snr_list[1] = noise_snr
        # generate wnoise
        if wnoise_flag:
            cfg_key = self.parse_block('colornoise')
            out_dict = self._data_manager.get_colornoise(cfg_key=cfg_key, length=length)
            wnoise_data = out_dict['mc_data']
            length = out_dict['length']
            wnoise_data, noisy_data, wnoise_snr = self.mix(
                noisy_data,
                wnoise_data,
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            snr_list[2] = wnoise_snr
        # randomize_volume
        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data,
            clean_data,
            disturb_data,
        )
        # add key to item
        directional_data = np.stack((clean_data[0], disturb_data[0]), axis=0)
        # if item is None:
        #     item = dict()
        item['mc_waveform'] = noisy_data  # nch, length
        item['directional_waveform'] = directional_data  # num_src, length
        item['direction'] = np.array([clean_direction, disturb_direction])[None, :]  # 1, length
        item['source_num'] = float(src_num_total)
        item['noise_flag'] = float(noise_flag)
        item['length'] = int(length)
        # item['clean_data'] = clean_data
        # item['disturb_data'] = disturb_data
        # item['noise_data'] = noise_data
        # item['wnoise_data'] = wnoise_data
        # item['snr_list'] = snr_list
        # item['agc'] = agc_val
        return item

    def simulate_bf(self, item, wav_key='waveform'):
        """Simulator entry for bf training.

        Args:
            item: input dict.

        Return:
            item: include data simulated.
        """
        # pylint:disable=too-many-locals
        gen_cfg = self._cfg.general
        # self.set_random_seed()
        noise_disturb_and_flag = gen_cfg.noise_disturb_and_flag
        disturb_flag = random.random() < gen_cfg.disturb_ratio
        if noise_disturb_and_flag:
            noise_flag = random.random() < gen_cfg.noise_ratio
        else:
            noise_flag = (random.random() < gen_cfg.noise_ratio) and (not disturb_flag)
        wnoise_flag = random.random() < gen_cfg.colornoise_ratio

        direction_all = [-1 for _ in range(gen_cfg.max_src_num + 2)]
        snr_list = [-1, -1, -1]
        src_num_total = 0
        # generate src1
        cfg_key = self.parse_block('clean')
        src_num = self.parse_src_num(cfg_key)
        src_num_total += int((src_num + 1) // 2)
        waveform_decouple_type = gen_cfg.get('waveform_decouple_type', 'total')
        out_dict = self._data_manager.get_one_audio(
            item,
            wav_key,
            waveform_decouple_type=waveform_decouple_type,
            cfg_key=cfg_key,
        )
        clean_data = out_dict['mc_data']
        length = out_dict['length']
        clean_direction = out_dict['rir_direction']
        clean_direction_use = out_dict['rir_direction_perturb']
        clean_mono = out_dict['mono_data']

        direction_all[0] = clean_direction_use
        direction_all[1] = clean_direction
        if src_num > 1:
            clean_direction2, rir = self._data_manager.load_rir_withref(
                [clean_direction_use], cfg_key=cfg_key, src_idx=1
            )
            clean_data2 = self._data_manager.conv_target_audio(clean_mono, rir, length=length)
            if isinstance(self._cfg[cfg_key]['src_condition'][1]['length'], str):
                length2 = eval(self._cfg[cfg_key]['src_condition'][1]['length'])
            length2 = int(length2)
            # clean_data2, length2, clean_direction2, _ = self._data_manager.get_one_audio(
            #     ref_direction=[clean_direction_use],
            #     cfg_key=cfg_key,
            #     src_idx=1
            # )
            clean_data[:, length - length2 :] = clean_data2[:, length - length2 :]
            direction_all[2] = clean_direction2
            assert abs(clean_direction2 - clean_direction_use) <= 25.0
        noisy_data = clean_data.copy()
        disturb_data = np.zeros_like(clean_data)
        noise_data = np.zeros_like(clean_data)
        wnoise_data = np.zeros_like(clean_data)
        # generate src2
        if disturb_flag:
            cfg_key = self.parse_block('disturb')
            src_num_total += self.parse_src_num(cfg_key)
            out_dict = self._data_manager.get_one_audio(
                ref_direction=[clean_direction_use], cfg_key=cfg_key, length=length
            )
            disturb_data = out_dict['mc_data']
            length = out_dict['length']
            disturb_direction = out_dict['rir_direction']
            disturb_data, noisy_data, disturb_snr = self.mix(
                noisy_data,
                disturb_data,
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            direction_all[3] = disturb_direction
            snr_list[0] = disturb_snr
        # generate noise
        if noise_flag:
            cfg_key = self.parse_block('noise')
            out_dict = self._data_manager.get_one_audio(cfg_key=cfg_key, length=length)
            noise_data = out_dict['mc_data']
            length = out_dict['length']
            noise_data, noisy_data, noise_snr = self.mix(
                noisy_data, noise_data, self._cfg[cfg_key]['snr_min'], self._cfg[cfg_key]['snr_max']
            )
            snr_list[1] = noise_snr
        # generate wnoise
        if wnoise_flag:
            cfg_key = self.parse_block('colornoise')
            out_dict = self._data_manager.get_colornoise(cfg_key=cfg_key, length=length)
            wnoise_data = out_dict['mc_data']
            length = out_dict['length']
            wnoise_data, noisy_data, _ = self.mix(
                noisy_data,
                wnoise_data,
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            snr_list[2] = disturb_snr
        # randomize_volume
        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data, clean_data, disturb_data
        )
        # add key to item
        item['mc_waveform'] = noisy_data  # nch, length
        item['target_sou_data'] = clean_data[0][None, :]
        item['direction_all'] = np.array(direction_all)[None, :]
        item['clean_direction_use'] = float(clean_direction_use)
        item['source_num'] = float(src_num_total)
        item['noise_flag'] = float(noise_flag)
        item['length'] = int(length)
        # item['clean_data'] = clean_data
        # item['disturb_data'] = disturb_data
        # item['noise_data'] = noise_data
        # item['wnoise_data'] = wnoise_data
        # item['snr_list'] = snr_list
        # item['agc'] = agc_val
        return item
