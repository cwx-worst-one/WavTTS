'''
Data load module in SE Simulator
Used to load audio data and rir datas
'''

import random
import pickle
import librosa
import numpy as np
import colorednoise as cn
from .diffuse_noise import gen_diffuse, gen_mix_matrix
from .colornoise import powerlaw_psd_gaussian
from core.dataset.data_fetcher import get_paths, TargetDataFetcher


class Data:
    """Load Audio Data
    Used to read audio datas with diffrent type
    such as clean, disturb, noise, colornoise and etc.
    """

    def __init__(
        self,
        data_configs=None,
        cfg=None,
        max_speech_length=None,
        cached_name='train_simulator_data_fetcher',
    ):
        '''init.
        Args:
            data_configs(list): data_type and data_path list, called `audio_data` in simulator config
            cfg(Config): full simulator config
            max_speech_length(int): max speech length
            cached_name(str): cached name used by falcon_reader
        '''
        # general config
        self._cfg = cfg
        self._top_db = cfg.general.top_db
        self._max_speech_length = max_speech_length
        self.mic_num = cfg.general.mic_num
        if self._max_speech_length is None:
            self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)

        # colornoise cfg
        self.beta_min = cfg['colornoise'].get('beta_min', 1.4)
        self.beta_max = cfg['colornoise'].get('beta_max', 2.0)
        self.color_noise_type = cfg['colornoise'].get('type', '')
        self.color_noise_fn = cfg['colornoise'].get('colornoise_fn', 'cn.powerlaw_psd_gaussian')
        self.fmin_min = cfg['colornoise'].get('fmin_min', 100)
        self.fmin_max = cfg['colornoise'].get('fmin_max', 1000)
        self.fs = cfg['colornoise'].get('fs', 16000)

        # init falcon_reader
        self.data_fetcher = None
        self.initialize_data_fetcher(data_configs, cached_name)
        self.fetcher_reader_init_flag = False  # reader must init in sub process

    def initialize_data_fetcher(self, data_configs, cached_name):
        '''initialize data fetcher'''
        # split rir data and audio data
        if data_configs is None:
            return
        data_paths = []
        # data-type to file-idxs
        # eg:
        #   {'noise':0, 'clean':1, 'disturb':2}
        # now one file type only support one file
        # TODO(litianyu.y): support data path list
        self.type2files = dict()
        idx = 0
        for data in data_configs:
            data_paths.append(data['path'])
            self.type2files[data['key']] = idx
            idx += 1
        self.data_fetcher = None
        cached_name += 'audio_data_fetcher'
        if data_paths:
            # falcon reader config
            falcon_reader_cfg = self._cfg.get('falcon_reader', dict())
            # whether use shared memory
            mem_shared = falcon_reader_cfg.get('mem_shared', True)
            chunk_size = falcon_reader_cfg.get('chunk_size', 10)
            fd_cache_size = falcon_reader_cfg.get('fd_cache_size', 256)
            parrallel_chunk_num = falcon_reader_cfg.get('parrallel_chunk_num', 16)
            io_thread_num = falcon_reader_cfg.get('io_thread_num', 8)
            self.data_fetcher = TargetDataFetcher(
                data_paths=data_paths,
                chunk_size=chunk_size,
                parrallel_chunk_num=parrallel_chunk_num,
                io_thread_num=io_thread_num,
                fd_cache_size=fd_cache_size,
                cached_name=cached_name,
                shuffle=True,
                mem_shared=mem_shared,
            )

    def fetcher_reader_init(self, local_process_num, local_pid):
        '''init data fetcher reader in sub process'''
        if self.fetcher_reader_init_flag:
            return
        self.data_fetcher.get_reader(local_process_num, local_pid)
        self.fetcher_reader_init_flag = True

    def read_from_reader(self, data_type, out_dtype=np.int16, unserialize_type='numpy'):
        """Read from reader.
        Args:
            data_type: audio data type key
            out_dtype: dtype of data readed, default np.int16

        Return:
            data readed of dtype out_dtype.
        """
        target_shard = self.type2files[data_type]
        data = self.data_fetcher.get_data(target_shard=target_shard)
        if unserialize_type == 'pickle':
            return pickle.loads(data)
        elif unserialize_type == 'numpy':
            return np.frombuffer(bytes(data), dtype=out_dtype).copy()

    def gen_colornoise(self, length=None, out_dtype=np.float32, iter_num=1):
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
        if length is None:
            length = self._max_speech_length
        if self.color_noise_type == 'uncorrelated':
            shape = (self.mic_num, length)
        else:
            shape = length
        if self.color_noise_fn == 'cn.powerlaw_psd_gaussian':
            beta = random.uniform(self.beta_min, self.beta_max)
            wnoise = cn.powerlaw_psd_gaussian(beta, shape)
        else:
            for i in range(iter_num):
                beta = self.beta_min + random.random() * (self.beta_max - self.beta_min)
                fmin = self.fmin_min + random.random() * (self.fmin_max - self.fmin_min)
                fmin = fmin / self.fs
                if i == 0:
                    wnoise = powerlaw_psd_gaussian(beta, shape, fmin)
                else:
                    wnoise += powerlaw_psd_gaussian(beta, shape, fmin)
            wnoise = wnoise / np.max(np.abs(wnoise))
        wnoise = wnoise.astype(out_dtype)
        return wnoise

    def get_one_data(self, data_type, length=None, out_dtype=np.int16, iter_num=1):
        """Read or generate one audio data.
        Args:
            data_type: audio data type key
            length: target data length used to gen colornoise

        Return:
            data readed
        """
        if data_type in self.type2files:
            return self.read_from_reader(data_type, out_dtype)  # length
        if data_type == 'colornoise':
            return self.gen_colornoise(length=length, out_dtype=out_dtype, iter_num=iter_num)

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
        data_type,
        outside_data=None,
        length=None,
        use_vad_merge=None,
        audio_repeat=False,
        random_clip=True,
        in_dtype=np.int16,
        out_dtype=np.float32,
        iter_num=1,
    ):
        """Get data.

        Args:
            outside_data: additional data to concate
            length: length of the data to be got, if None, use max_speech_length.
            use_vad_merge: whther to use vad merge, if None, according to default config.
            in_dtype: dtype of outside_data, default np.int16
            out_dtype: dtype of result data, default np.float32.
            iter_num: only used while generate colornoise data

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
            data = self.get_one_data(
                data_type=data_type, length=length, out_dtype=in_dtype, iter_num=iter_num
            )
            if use_vad_merge:
                data = self.vad_merge(data)
        # if length is -1 return outside data length
        if length == -1:
            length = data.shape[-1]
        while data.shape[-1] < length:
            if audio_repeat:
                nex_data = data
            else:
                nex_data = self.get_one_data(
                    data_type=data_type, length=length, out_dtype=in_dtype, iter_num=iter_num
                )
            if use_vad_merge:
                nex_data = self.vad_merge(nex_data)
            data = np.concatenate((data, nex_data), axis=-1)
        if data.shape[-1] >= length:
            if random_clip:
                start = random.randint(0, data.shape[-1] - length)
            else:
                start = 0
            data = data[..., start : start + length]
        data = data.astype(out_dtype)
        return data, length, data_type


class RirData(Data):
    """Rir data"""

    def __init__(
        self, data_configs=None, cfg=None, max_speech_length=None, cached_name='rir_data_fetcher'
    ):
        super().__init__(cfg=cfg, max_speech_length=max_speech_length)
        if 'rir' not in self._cfg:
            self.angle_list = ['']
        else:
            rir_cfg = self._cfg.rir
            self.angle_list = list(
                range(rir_cfg.angle_start, rir_cfg.angle_end, rir_cfg.angle_step)
            )
            self.zone_list = [int(angle // rir_cfg.angle_step) for angle in self.angle_list]
        self.rir_num = len(self.angle_list)
        self.initialize_data_fetcher(data_configs, cached_name)

    def initialize_data_fetcher(self, data_configs, cached_name):
        '''initialize data fetcher'''
        if data_configs is None:
            return
        rir_data_paths = []
        # rir_type : path_list_idxs
        self.type2files = dict()
        # rir_type: angle : path_idx
        self.type_angle2files = dict()
        # rir_type : unserialize_type
        self.type2unserialize_type = dict()
        # rir_type : length
        self.type2length = dict()
        # rir_type :
        self.type2mic_num = dict()
        file_off = 0
        for data in data_configs:
            data_type = data['key']
            data_paths = get_paths(
                data_root=data['path'], file_prefix=data['prefix'], shard_list=self.angle_list
            )
            self.type2files[data_type] = list(range(file_off, file_off + len(data_paths)))
            self.type_angle2files[data_type] = dict()
            for angle, file_idx in zip(self.angle_list, self.type2files[data_type]):
                self.type_angle2files[data_type][angle] = file_idx
            rir_data_paths += data_paths
            file_off += len(data_paths)
            self.type2unserialize_type[data_type] = data.get('unserialize_type', 'pickle')
            self.type2length[data_type] = data.get('length', None)
            self.type2mic_num[data_type] = data.get('mic_num', self.mic_num)
        cached_name += 'rir_data_fetcher'
        # falcon reader config
        falcon_reader_cfg = self._cfg.get('falcon_reader', dict())
        # whether use shared memory
        mem_shared = falcon_reader_cfg.get('mem_shared', True)
        chunk_size = falcon_reader_cfg.get('chunk_size', 10)
        fd_cache_size = falcon_reader_cfg.get('fd_cache_size', 256)
        parrallel_chunk_num = falcon_reader_cfg.get('parrallel_chunk_num', 16)
        io_thread_num = falcon_reader_cfg.get('io_thread_num', 8)
        self.data_fetcher = TargetDataFetcher(
            data_paths=rir_data_paths,
            chunk_size=chunk_size,
            parrallel_chunk_num=parrallel_chunk_num,
            io_thread_num=io_thread_num,
            fd_cache_size=fd_cache_size,
            cached_name=cached_name,
            shuffle=True,
            mem_shared=mem_shared,
        )

    def read_from_reader(self, data_type, target_angle=None, num_source=None):
        """Read from reader.

        Args:
            target_angle: target angle want get

        Return:
            data readed of dtype out_dtype.
        """
        # get target path idx
        if target_angle is None:
            target_shard = random.choice(self.type2files[data_type])
        else:
            target_shard = self.type_angle2files[data_type][target_angle]
        value_tmp = self.data_fetcher.get_data(target_shard)

        directions = None
        if self.type2unserialize_type[data_type] == 'pickle':
            value = pickle.loads(value_tmp)
            if 'direction' in value:
                directions = value['direction']
            rir = value['rir']
        elif self.type2unserialize_type[data_type] == 'numpy':
            value_tmp = pickle.loads(value_tmp)
            value = np.frombuffer(bytes(value_tmp['rir']), dtype=np.float32).reshape(
                -1, self.type2length[data_type]
            )
            if 'direction' in value_tmp:
                direction = float(value_tmp['direction'])
                directions = [direction]
            rir = [value]
        elif self.type2unserialize_type[data_type] == 'numpy_only':
            rir = np.frombuffer(value_tmp, dtype=np.float32)
        else:
            raise ValueError("Unknown unserialize type!")

        if num_source is not None:
            rir = rir.reshape(num_source, self.type2mic_num[data_type], self.type2length[data_type])

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

    def get_rir_withoutref(self, data_type, wkp_angle=None, cfg_key=None, src_idx=0):
        """Get rir data without angle condition.

        Args:
            data_type: rir data type
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
            rir, directions = self.read_from_reader(data_type, target_shard)
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

    def get_rir_withref(self, data_type, ref_direction, cfg_key=None, src_idx=0):
        """Get rir data with angle constraint.

        Args:
            data_type: rir data type
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
            rir, directions = self.read_from_reader(data_type, target_shard)
            disturb_direction = directions[0]
            disturb_data = rir[0]
            angle_diff = self.angle_dis(disturb_direction, target_direction)
            if min_angle2ref < angle_diff < max_angle2ref:
                break
        return disturb_direction, disturb_data

    def get_rir(self, data_type, cfg_key=None, src_idx=0):
        '''get single rir'''
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        num_source = rir_condition.get('num_source', 2)
        norm_rir = rir_condition.get('norm_rir', True)
        select_target_source_rir = rir_condition.get('select_target_source_rir', None)

        # ns, mic, rir_length
        rir, _ = self.read_from_reader(data_type=data_type, num_source=num_source)
        rir = rir.copy()
        if num_source is not None:
            index_rir = np.zeros(shape=[num_source], dtype=np.int16)
        else:
            index_rir = None
        # do normalization for rir
        if norm_rir:
            norm_rir = np.linalg.norm(rir, axis=-1)
            for index in range(num_source):
                index_rir[index] = np.where(rir[index][0] == np.max(rir[index][0]))[0][0]
                if np.sum(norm_rir[index] < 1e-8) > 0:
                    rir[index] = 0
                    rir[index, :, 0] = 1.0
                else:
                    rir[index] = rir[index] / norm_rir[index, :, np.newaxis]
        if select_target_source_rir is not None and isinstance(select_target_source_rir, int):
            assert select_target_source_rir < num_source
            return rir[select_target_source_rir], index_rir

        return rir, index_rir


class DataManager:
    """Data manager"""

    def __init__(self, cfg, cached_name='train') -> None:
        '''init.'''
        self._cfg = cfg
        self.mic_num = cfg.general.mic_num
        self._audio_data = Data(data_configs=cfg.audio_data, cfg=cfg, cached_name=cached_name)
        self._rir_data = RirData(data_configs=cfg.rir_data, cfg=cfg, cached_name=cached_name)
        self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)
        # for diffuse rir
        self.noise_diffuse_rir = None
        self.fetcher_reader_init_flag = False

    def fetcher_reader_init(self, local_process_num, local_pid):
        '''fetcher reader init'''
        if self.fetcher_reader_init_flag:
            return
        self._audio_data.fetcher_reader_init(local_process_num, local_pid)
        self._rir_data.fetcher_reader_init(local_process_num, local_pid)
        self.fetcher_reader_init_flag = True

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
            trunc diffuse without convolution.
                noise_data: raw audio data (mic, length)
                rir_data_diffuse: truc diffuse rir data
                length: audio data length
        """
        mic_num = self._cfg.general.mic_num
        noise_data, length, _ = self.load_audio_data(
            cfg_key=cfg_key, src_idx=src_idx, length=length
        )
        noise_data_diffuse = [noise_data] * mic_num
        rir_data_diffuse = []  # split from one rir?
        for idx in range(mic_num):
            _, _, rir_noise_data = self.load_rir_withoutref(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=0
            )
            rir_data_diffuse.append(rir_noise_data[idx])
        noise_data_diffuse = np.stack(noise_data_diffuse, axis=0)
        rir_data_diffuse = np.stack(rir_data_diffuse, axis=0)
        diffuse_data = dict(
            noise_data=noise_data_diffuse, rir_data_diffuse=rir_data_diffuse, length=length
        )
        return diffuse_data

    def load_rir_withoutref(self, wkp_angle=None, rir_key=None, cfg_key=None, src_idx=0):
        """Load rir without angle constraint.

        Args:
            wkp_angle: wake up angle.
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module

        """
        if rir_key is not None:
            return self._rir_data.get_rir_withoutref(
                data_type=rir_key, wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
            )
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                return self._rir_data.get_rir_withoutref(
                    data_type=cond['key'], wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
                )
        cond = rir_condition['path_condition'][-1]
        return self._rir_data.get_rir_withoutref(
            data_type=cond['key'], wkp_angle=wkp_angle, cfg_key=cfg_key, src_idx=src_idx
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
            return self._rir_data.get_rir_withref(
                data_type=rir_key, ref_direction=ref_direction, cfg_key=cfg_key, src_idx=src_idx
            )
        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)
        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                return self._rir_data.get_rir_withref(
                    data_type=cond['key'],
                    ref_direction=ref_direction,
                    cfg_key=cfg_key,
                    src_idx=src_idx,
                )
        cond = rir_condition['path_condition'][-1]
        return self._rir_data.get_rir_withref(
            data_type=cond['key'], ref_direction=ref_direction, cfg_key=cfg_key, src_idx=src_idx
        )

    def load_multi_source_rir(self, rir_key=None, cfg_key=None, src_idx=0):
        '''
        load multi source rir
        '''
        if rir_key is not None:
            return self._rir_data.get_rir(data_type=rir_key, cfg_key=cfg_key, src_idx=src_idx)

        rir_condition = self._cfg[cfg_key]['rir_condition']
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        else:
            assert isinstance(rir_condition, dict)

        random_num = random.random()
        for cond in rir_condition['path_condition']:
            if random_num < cond['prob']:
                return self._rir_data.get_rir(
                    data_type=cond['key'], cfg_key=cfg_key, src_idx=src_idx
                )
        cond = rir_condition['path_condition'][-1]
        return self._rir_data.get_rir(data_type=cond['key'], cfg_key=cfg_key, src_idx=src_idx)

    def load_audio_data(
        self, outside_data=None, cfg_key=None, src_idx=0, length=None, random_clip=True
    ):
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

        assert isinstance(src_condition, dict)
        use_vad_merge = src_condition.get('use_vad_merge', False)
        path_condition = src_condition.get('path_condition', None)
        in_dtype = eval(src_condition.get('in_dtype', 'np.int16'))
        out_dtype = eval(src_condition.get('out_dtype', 'np.float32'))
        audio_repeat = src_condition.get('audio_repeat', False)
        if length is None:
            length = src_condition.get('length', self._max_speech_length)
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)
        if path_condition is None:
            raise ValueError("path condition must be assigned in config!")
        random_prob = random.random()
        for cond in path_condition:
            if random_prob < cond['prob']:
                return self._audio_data.get_data(
                    data_type=cond['key'],
                    outside_data=outside_data,
                    length=length,
                    use_vad_merge=use_vad_merge,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    audio_repeat=audio_repeat,
                    random_clip=random_clip,
                )
        return None, length, None  # data, length, data_type

    def generate_audio_without_rir(
        self,
        waveform=None,
        use_outside_data='false',
        cfg_key=None,
        src_idx=0,
        length=None,
    ):
        ''' '''
        # get audio data
        if use_outside_data == "partial":
            waveform = waveform.squeeze()  # (length,)
            if not waveform.ndim == 1:
                waveform_axis = self._cfg.general.get('waveform_decouple_partial_axis', 0)
                waveform = waveform[waveform_axis]
            clean, length, _ = self.load_audio_data(
                outside_data=waveform, cfg_key=cfg_key, src_idx=src_idx, length=length
            )
        elif use_outside_data == 'true':
            # TODO(litianyu.y): use waveform.shape[-1]?
            if waveform.ndim == 1:
                data_len = waveform.shape[0]
            elif waveform.ndim == 2:
                data_len = max(waveform.shape)
            clean = waveform.squeeze().astype(np.float32)
            length = data_len
        elif use_outside_data == 'false':
            clean, length, _ = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
        else:
            print("error use_outside_data is ", use_outside_data, cfg_key)
            raise ValueError('Unrecognized use_outside_data!')

        audio_data = [clean] * self.mic_num
        audio_data = np.stack(audio_data, axis=0)  # mic_num, length

        out_dict = dict(
            audio_data=audio_data,  # (mic_num, length)
            length=length,  # int
            rir_data=None,
            rir_type='norir',  # just do fftconv
        )
        return out_dict

    def generate_audio_use_direct_rir(
        self,
        waveform=None,
        ref_direction=None,
        use_outside_data='false',
        rir_type=None,
        rir_key=None,
        cfg_key=None,
        src_idx=0,
        length=None,
    ):

        """generate audio data with directional rir

        Args:
            waveform: outside data from training data
            ref_direction: [optional] direction of reference angle when generating rir
            use_outside_data: whether to use outside data or not or partially
                (1) 'false': complete decouple (default)
                (2) 'partial': use waveform, if not enough in length, added by clean_data
                (3) 'true': use input waveform
            rir_type: whether to generate rir with constraint(withref) or not(withoutref)
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                audio_data: audio before convolution (mic, length)
                rir_data: rir data (mic, rir_length)
                rir_direction: rir direction
                rir_direction_perturb: rir direction with perturbation
                length: generated audio length
                rir_type: rir type
        """
        # get audio data
        if use_outside_data == "partial":
            waveform = waveform.squeeze()  # (length,)
            if not waveform.ndim == 1:
                waveform_axis = self._cfg.general.get('waveform_decouple_partial_axis', 0)
                waveform = waveform[waveform_axis]
            clean, length, _ = self.load_audio_data(
                outside_data=waveform, cfg_key=cfg_key, src_idx=src_idx, length=length
            )
        elif use_outside_data == 'true':
            # TODO(litianyu.y): use waveform.shape[-1]?
            if waveform.ndim == 1:
                data_len = waveform.shape[0]
            elif waveform.ndim == 2:
                data_len = max(waveform.shape)
            clean = waveform.squeeze().astype(np.float32)
            length = data_len
        elif use_outside_data == 'false':
            clean, length, _ = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
        else:
            print("error use_outside_data is ", use_outside_data, cfg_key)
            raise ValueError('Unrecognized use_outside_data!')

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
        else:
            rir_data, _ = self.load_multi_source_rir(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx
            )
            rir_direction = None
            rir_direction_perturb = None
            rir_direction_use = None
        audio_data = [clean] * self.mic_num
        audio_data = np.stack(audio_data, axis=0)  # mic_num, length
        out_dict = dict(
            audio_data=audio_data,  # (mic_num, length)
            length=length,  # int
            rir_data=rir_data,  # (mic_num, rir_length)
            rir_direction=rir_direction,  # float
            rir_direction_perturb=rir_direction_use,  # float
            rir_type='direct',  # just do fftconv
        )

        return out_dict

    def generate_audio_use_diffuse_rir(self, rir_key=None, cfg_key=None, src_idx=0, length=None):
        """generate unconvolved audio data with diffuse rir

        Args:
            rir_key: key of the rir path identifier
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                audio_data: raw audio data (mic, length)
                rir_data: truc diffuse rir data
                length: audio data length
                rir_type: rir type
                audio_data2: another raw audio data (mic, length)
                rir_data2: another truc diffuse rir data

        """
        # mic * length
        noise_data_diffuse = self.gen_trunc_diffuse(
            rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx, length=length
        )
        noise_data2 = np.zeros_like(noise_data_diffuse['noise_data'])
        rir_data_diffuse2 = np.zeros_like(noise_data_diffuse['rir_data_diffuse'])
        if random.random() < 0.5:
            noise_data_diffuse2 = self.gen_trunc_diffuse(
                rir_key=rir_key, cfg_key=cfg_key, src_idx=src_idx, length=length
            )
            noise_data2 = noise_data_diffuse2['noise_data']
            rir_data_diffuse2 = noise_data_diffuse2['rir_data_diffuse']

        out_dict = dict(
            audio_data=noise_data_diffuse['noise_data'],
            rir_data=noise_data_diffuse['rir_data_diffuse'],
            length=noise_data_diffuse['length'],
            rir_type='diffuse',
            audio_data2=noise_data2,
            rir_data2=rir_data_diffuse2,
        )
        return out_dict

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
                audio_data: raw audio data (mic, length)
                length: audio data length
                rir_data: diffuse rir data, will same
                rir_type: rir type
        """
        gen_cfg = self._cfg.general
        noise_data = []
        for _ in range(self.mic_num):
            tmp, length, _ = self.load_audio_data(cfg_key=cfg_key, src_idx=src_idx, length=length)
            noise_data.append(tmp)
        noise_data_orig = np.stack(noise_data, axis=0)  # mic_dim * sampled_len
        if self.noise_diffuse_rir is None:
            diffuse_matrix = gen_diffuse(
                gen_cfg.sampling_rate,
                gen_cfg.mic_num,
                gen_cfg.array_type,
                gen_cfg.mic_distance,
                gen_cfg.radius,
            )
            self.noise_diffuse_rir = gen_mix_matrix(diffuse_matrix, 'eig')
        # ToDo: update noise_data dim?
        out_dict = dict(
            audio_data=noise_data_orig,  # mic_dim * sampled_len
            length=noise_data_orig.shape[-1],
            rir_data=self.noise_diffuse_rir,
            rir_type='diffuse_equation',
        )
        return out_dict

    def generate_colornoise_data(
        self, cfg_key='colornoise', ref_direction=None, src_idx=0, length=None
    ):
        """Get colored noise.

        Args:
            cfg_key: identify key of a signal module
            src_idx: source idx of the signal module
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length

        Return:
            outdict: dict contains
                audio_data: generated multichannel audio (mic, length)
                rir_data: rir data (mic, rir_length)
                rir_type: colornoise type, correlated or uncorrelated
                length: generated audio length
                rir_direction: rir direction
        """
        # load colornoise data
        if length is None:
            length = self._cfg[cfg_key].get('length', self._max_speech_length)
        max_iter_num = self._cfg[cfg_key].get('max_iter_num', 1)
        iter_num = random.randint(1, max_iter_num)

        in_dtype = eval(self._cfg[cfg_key].get('in_dtype', 'np.float32'))
        out_dtype = eval(self._cfg[cfg_key].get('out_dtype', 'np.float32'))
        wnoise, length, _ = self._audio_data.get_data(
            data_type=cfg_key,
            length=length,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            iter_num=iter_num,
        )
        cfg = self._cfg[cfg_key]
        rir_data = None
        rir_direction = None
        if cfg['type'] == "correlated":
            if 'rir_condition' not in cfg:
                rir_direction, _, rir_data = self.load_rir_withoutref(
                    rir_key='direct', cfg_key=cfg_key, src_idx=src_idx
                )
            else:
                if cfg['rir_condition']['type'] == 'multi_source':
                    rir_data, _ = self.load_multi_source_rir(cfg_key=cfg_key)
                elif cfg['rir_condition']['type'] == 'withref':
                    rir_direction, _, rir_data = self.load_rir_withref(
                        rir_key='direct',
                        ref_direction=ref_direction,
                        cfg_key=cfg_key,
                        src_idx=src_idx,
                    )
                elif cfg['rir_condition']['type'] == 'withoutref':
                    rir_direction, _, rir_data = self.load_rir_withoutref(
                        rir_key='direct', cfg_key=cfg_key, src_idx=src_idx
                    )

            wnoise = np.stack([wnoise] * self.mic_num, axis=0)
        out_dict = dict(
            audio_data=wnoise,  # mic * length
            length=wnoise.shape[-1],
            rir_data=rir_data,  # mic * rir_length
            rir_type=cfg['type'],
            rir_direction=rir_direction,
        )
        return out_dict

    def get_audio_and_rir(
        self,
        # for audio data
        waveform=None,
        cfg_key=None,
        use_outside_data='false',
        length=None,
        # for rir data
        ref_direction=None,
        src_idx=0,
    ):
        """get audio and rir data for get simulator datas

        Args:
            waveform: [optional] training data waveform, item[wav_key]
            cfg_key: identify key of a signal module
            use_outside_data: whether to use outside data or not or partially
                (1) 'false': complete decouple (default)
                (2) 'partial': use waveform, if not enough in length, added by clean_data
                (3) 'true': use input waveform
            length: length of trunc diffuse, if None, use
                (1) length specified in configs
                (2) max_speech_length
            ref_direction: [optional] direction of reference angle when generating rir
            src_idx: source idx of the signal module
        """
        if length is not None:
            if isinstance(length, str):
                length = eval(length)
            else:
                length = int(length)

        if cfg_key == 'colornoise':
            return self.generate_colornoise_data(
                cfg_key=cfg_key, src_idx=src_idx, length=length, ref_direction=ref_direction
            )

        rir_condition = self._cfg[cfg_key].get('rir_condition', None)
        # TODO: whether can use random.choice
        if isinstance(rir_condition, list):
            rir_condition = rir_condition[src_idx]
        assert isinstance(rir_condition, dict)

        prob = random.random()
        for cond in rir_condition['path_condition']:
            if prob < cond['prob']:
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
                        waveform=waveform,
                        ref_direction=ref_direction,
                        use_outside_data=use_outside_data,
                        rir_key=rir_key,
                        rir_type=rir_type,
                        cfg_key=cfg_key,
                        src_idx=src_idx,
                        length=length,
                    )

        return self.generate_audio_without_rir(
            waveform=waveform,
            use_outside_data=use_outside_data,
            cfg_key=cfg_key,
            src_idx=src_idx,
            length=length,
        )

    def fix_audio_length(self, waveform, length=None, position=0, random_clip=False):
        '''
        fix audio length to match length
        '''
        if length is None:
            length = self._max_speech_length
        if waveform.shape[-1] >= length:
            if random_clip:
                position = random.randint(0, waveform.shape[-1] - length)
            waveform = waveform[position : position + length]
        else:
            num_clips = int(np.ceil(length / waveform.shape[-1]))
            waveform = np.tile(waveform, num_clips)
            waveform = waveform[:length]
        return waveform
