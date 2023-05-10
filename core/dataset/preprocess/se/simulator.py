'''
simulator module
'''

import random
import numpy as np
import torch
from .diffuse_noise import DiffuseNoise
from .fftconvolve import fftconvolve
from .data_manager import DataManager
from .aec import (
    aec_load_simulate_data,
    aec_simulate_collate_fn,
    aec_simulator_fn,
    aec_infer_simulator_fn,
)
from core.utils import Config, dist_file_get
from core.dataset.preprocess import PREPROCESS


@PREPROCESS.register_module()
class LoadSimulatorData:
    """Signal data loading"""

    def __init__(
        self,
        simulate_type='simulate',
        simulator_config_path='',
        wav_key='waveform',
        cached_name='train',
    ):
        '''
        simulate_type:
            'simulate' : simulator entry for joint training.
            'simulate_bf' : simulator entry for bf training.
            'simulate_ssl' : simulator entry for ssl training.
        simulator_config_path:
            simulator config path, support local, hdfs and etc.
        wav_key:
            waveform save name in train dataset
        cached_name:
            save name used in DataManger to load data from data source
            if mem_shared is set True(default) in simulator config,
            the cached_name in train and valid transform should be diff
        '''
        cfg = Config.fromfile(simulator_config_path).cfg_dict
        self._cfg = cfg
        self._data_manager = DataManager(cfg, cached_name=cached_name)
        self._max_speech_length = int(cfg.general.max_speech_time * cfg.general.sampling_rate)
        self.simulate_type = simulate_type
        self.simulate_fn = getattr(self, simulate_type)
        self.gen_cfg = self._cfg.general
        self.mic_num = self.gen_cfg.mic_num
        self.use_outside_data = self.gen_cfg.get('use_outside_data', 'false')
        self.noise_disturb_and_flag = self.gen_cfg.noise_disturb_and_flag
        self.wav_key = wav_key

    def set_random_seed(self, seed=None):
        """Set random seed.

        Args:
            seed: random seed, if None, use config seed.
        """
        if seed is None:
            seed = self._cfg.general.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def parse_block(self, cfg_key):
        """parse signal module
        one type of signal may have multi choices
        this func will choice with diffrent prob
        """
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

    def simulate(self, item):
        """Simulator entry for joint training.

        Args:
            item: dict contains outside data

        Return:
            item: include data simulated.
        """
        disturb_flag = random.random() < self.gen_cfg.disturb_ratio
        if self.noise_disturb_and_flag:
            noise_flag = random.random() < self.gen_cfg.noise_ratio
        else:
            noise_flag = (random.random() < self.gen_cfg.noise_ratio) and (not disturb_flag)

        clean_cfg_key = self.parse_block('clean')
        clean_data_out_dict = self._data_manager.get_audio_and_rir(
            waveform=item[self.wav_key],
            cfg_key=clean_cfg_key,
            use_outside_data=self.use_outside_data,
        )
        mix_snr = [0, 0]  # set to 0
        clean_direction = clean_data_out_dict['rir_direction']
        length = clean_data_out_dict['length']
        disturb_data_out_dict = None
        if disturb_flag:
            disturb_cfg_key = self.parse_block('disturb')
            disturb_data_out_dict = self._data_manager.get_audio_and_rir(
                ref_direction=[clean_direction], cfg_key=disturb_cfg_key, length=length
            )
            mix_snr[0] = np.random.uniform(
                self._cfg[disturb_cfg_key]['snr_min'], self._cfg[disturb_cfg_key]['snr_max']
            )
        noise_data_out_dict = None
        if noise_flag:
            noise_cfg_key = self.parse_block('noise')
            noise_data_out_dict = self._data_manager.get_audio_and_rir(
                cfg_key=noise_cfg_key, length=length
            )
            mix_snr[1] = np.random.uniform(
                self._cfg[noise_cfg_key]['snr_min'], self._cfg[noise_cfg_key]['snr_max']
            )
        item['clean'] = clean_data_out_dict
        item['disturb'] = disturb_data_out_dict
        item['noise'] = noise_data_out_dict
        item['length'] = length
        item['snr'] = mix_snr
        item['simu_channel_num'] = self.gen_cfg.mic_num
        item['direction'] = clean_direction

        return item

    def simulate_ssl(self, item):
        """Simulator entry for ssl training.

        Args:
            item: dict contains outside data

        Return:
            item: include data simulated.
        """
        disturb_flag = random.random() < self.gen_cfg.disturb_ratio
        if self.noise_disturb_and_flag:
            noise_flag = random.random() < self.gen_cfg.noise_ratio
        else:
            noise_flag = (random.random() < self.gen_cfg.noise_ratio) and (not disturb_flag)
        wnoise_flag = random.random() < self.gen_cfg.colornoise_ratio

        src_num_total = 0
        # generate src1
        cfg_key = self.parse_block('clean')
        src_num_total += self.parse_src_num(cfg_key)

        clean_data_out_dict = self._data_manager.get_audio_and_rir(
            waveform=item[self.wav_key],
            use_outside_data=self.use_outside_data,
            cfg_key=cfg_key,
        )
        length = clean_data_out_dict['length']
        clean_direction = clean_data_out_dict['rir_direction']
        disturb_direction = -1
        snr_list = [-1, -1, -1]
        # generate src2
        disturb_data_out_dict = None
        if disturb_flag:
            cfg_key = self.parse_block('disturb')
            src_num_total += self.parse_src_num(cfg_key)
            disturb_data_out_dict = self._data_manager.get_audio_and_rir(
                ref_direction=[clean_direction], cfg_key=cfg_key, length=length
            )
            disturb_direction = disturb_data_out_dict['rir_direction']
            disturb_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            snr_list[0] = disturb_snr
        # generate noise
        noise_data_out_dict = None
        if noise_flag:
            cfg_key = self.parse_block('noise')
            noise_data_out_dict = self._data_manager.get_audio_and_rir(
                ref_direction=[clean_direction], cfg_key=cfg_key, length=length
            )
            noise_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'], self._cfg[cfg_key]['snr_max']
            )
            snr_list[1] = noise_snr
        # generate wnoise
        wnoise_data_out_dict = None
        if wnoise_flag:
            cfg_key = self.parse_block('colornoise')
            wnoise_data_out_dict = self._data_manager.get_audio_and_rir(
                cfg_key=cfg_key, length=length
            )
            wnoise_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            snr_list[2] = wnoise_snr

        item['clean'] = clean_data_out_dict
        item['disturb'] = disturb_data_out_dict
        item['noise'] = noise_data_out_dict
        item['wnoise'] = wnoise_data_out_dict

        item['direction'] = np.array([clean_direction, disturb_direction])[None, :]  # 1, length
        item['source_num'] = float(src_num_total)
        item['noise_flag'] = float(noise_flag)
        item['length'] = int(length)
        item['snr'] = snr_list
        return item

    def simulate_bf(self, item):
        """Simulator entry for bf training.

        Args:
            item: input dict.

        Return:
            item: include data simulated.
        """
        gen_cfg = self._cfg.general
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
        clean_data_out_dict = self._data_manager.get_audio_and_rir(
            item[self.wav_key],
            use_outside_data=self.use_outside_data,
            cfg_key=cfg_key,
        )
        length = clean_data_out_dict['length']
        clean_direction = clean_data_out_dict['rir_direction']
        clean_direction_use = clean_data_out_dict['rir_direction_perturb']

        direction_all[0] = clean_direction_use
        direction_all[1] = clean_direction

        clean_data_out_dict2 = None
        if src_num > 1:
            clean_data_out_dict2 = self._data_manager.generate_audio_use_direct_rir(
                waveform=clean_data_out_dict['audio_data'][0],  # mic*length -> length
                ref_direction=[clean_direction_use],
                use_outside_data='true',
                rir_type='withref',
                cfg_key=cfg_key,
                src_idx=1,
            )
            if isinstance(self._cfg[cfg_key]['src_condition'][1]['length'], str):
                length2 = eval(self._cfg[cfg_key]['src_condition'][1]['length'])
            length2 = int(length2)
            clean_direction2 = clean_data_out_dict2['rir_direction']
            clean_data_out_dict2['length'] = length2
            direction_all[2] = clean_direction2
            assert abs(clean_direction2 - clean_direction_use) <= 25.0

        # generate src2
        disturb_data_out_dict = None
        if disturb_flag:
            cfg_key = self.parse_block('disturb')
            src_num_total += self.parse_src_num(cfg_key)
            disturb_data_out_dict = self._data_manager.get_audio_and_rir(
                ref_direction=[clean_direction_use], cfg_key=cfg_key, length=length
            )
            disturb_direction = disturb_data_out_dict['rir_direction']
            disturb_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            direction_all[3] = disturb_direction
            snr_list[0] = disturb_snr
        noise_data_out_dict = None
        # generate noise
        if noise_flag:
            cfg_key = self.parse_block('noise')
            noise_data_out_dict = self._data_manager.get_audio_and_rir(
                cfg_key=cfg_key, length=length
            )
            noise_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'], self._cfg[cfg_key]['snr_max']
            )
            snr_list[1] = noise_snr
        # generate wnoise
        wnoise_data_out_dict = None
        if wnoise_flag:
            cfg_key = self.parse_block('colornoise')
            wnoise_data_out_dict = self._data_manager.get_audio_and_rir(
                cfg_key=cfg_key, length=length
            )
            wnoise_snr = np.random.uniform(
                self._cfg[cfg_key]['snr_min'],
                self._cfg[cfg_key]['snr_max'],
            )
            # snr_list[2] = disturb_snr
            snr_list[2] = wnoise_snr  # check here is ok?

        # add key to item
        item['clean'] = clean_data_out_dict
        item['clean2'] = clean_data_out_dict2
        item['disturb'] = disturb_data_out_dict
        item['noise'] = noise_data_out_dict
        item['wnoise'] = wnoise_data_out_dict
        item['snr'] = snr_list

        item['direction_all'] = np.array(direction_all)[None, :]
        item['clean_direction_use'] = float(clean_direction_use)
        item['source_num'] = float(src_num_total)
        item['noise_flag'] = float(noise_flag)
        item['length'] = int(length)
        return item

    def simulate_denoise(self, item):
        '''Simulator entry for denoise training.

        Args:
            item: input dict.

        Return:
            item: include data simulated.'''
        gen_cfg = self._cfg.general
        # colornoise flag
        wnoise_flag = random.random() < gen_cfg.colornoise_ratio
        early_reverb_signal = gen_cfg.get('early_reverb_signal', 'use_zero')
        clean_flag = random.random() < gen_cfg.get('clean_ratio', 1.0)
        noise_flag = (not clean_flag) or random.random() < gen_cfg.get('noise_ratio', 1.0)

        mix_snr = [0, 0]

        noise_cfg_key = self.parse_block('noise')
        rir_cfg_key = noise_cfg_key  # use noise rir condition
        rir, _ = self._data_manager.load_multi_source_rir(cfg_key=rir_cfg_key)

        # load speech
        clean_data_out_dict = None
        length = None
        clean_cfg_key = self.parse_block('clean')
        if clean_flag:
            clean_data, length, _ = self._data_manager.load_audio_data(
                outside_data=item[self.wav_key],
                cfg_key=clean_cfg_key,
                length=self._max_speech_length,
                random_clip=True,
            )
            audio_data = [clean_data] * self.mic_num
            audio_data = np.stack(audio_data, axis=0)  # mic_num, length

            clean_data_out_dict = dict(
                audio_data=audio_data,  # (mic_num, length)
                length=length,  # int
                rir_data=None,  # (mic_num, rir_length)
                rir_direction=-1,  # float
                rir_direction_perturb=-1,  # float
                rir_type='direct',  # just do fftconv
            )
            if self._cfg[clean_cfg_key]['rir_condition']:  # config rir_condition
                clean_data_out_dict['rir_data'] = rir[0]
            elif not self._cfg[clean_cfg_key]['rir_condition']:  # no rir
                clean_data_out_dict['rir_type'] = 'norir'  # no rir

        # load noise
        noise_data_out_dict = None
        sign_clip = 0
        if noise_flag:
            noise_data, length, noise_key = self._data_manager.load_audio_data(
                cfg_key=noise_cfg_key, length=length, random_clip=True
            )
            audio_data = [noise_data] * self.mic_num
            audio_data = np.stack(noise_data, axis=0)  # mic_num, length
            noise_data_out_dict = dict(
                audio_data=audio_data,  # (mic_num, length)
                length=length,  # int
                rir_data=None,  # (mic_num, rir_length)
                rir_direction=-1,  # float
                rir_direction_perturb=-1,  # float
                rir_type='direct',  # just do fftconv
            )
            mix_snr[0] = np.random.uniform(
                self._cfg[noise_cfg_key]['snr_min'], self._cfg[noise_cfg_key]['snr_max']
            )
            if 'clip' in noise_key:
                sign_clip = 1
            if self.mic_num < rir[1].shape[0]:
                noise_data_out_dict['rir_data'] = rir[1][0:1, ...]
            else:
                noise_data_out_dict['rir_data'] = rir[1]

        # load colornoise
        wnoise_data_out_dict = None
        wnoise_cfg_key = self.parse_block('colornoise')
        if wnoise_flag:
            wnoise_num = random.randint(1, 3)
            wnoise_data_out_dict = []
            for _ in range(wnoise_num):
                random_data = random.random()
                wnoise_data_dict = self._data_manager.get_audio_and_rir(
                    cfg_key=wnoise_cfg_key, length=length
                )
                wnoise_data_dict['random_data'] = random_data
                wnoise_data_out_dict.append(wnoise_data_dict)
            mix_snr[1] = (
                random.random()
                * (self._cfg[wnoise_cfg_key]['snr_max'] - self._cfg[wnoise_cfg_key]['snr_min'])
                + self._cfg[wnoise_cfg_key]['snr_min']
            )

        item['clean'] = clean_data_out_dict
        item['noise'] = noise_data_out_dict
        item['wnoise'] = wnoise_data_out_dict
        item['snr'] = mix_snr
        item['length'] = length
        item['sign_clip'] = sign_clip
        if early_reverb_signal == 'use_zero' or (not clean_flag):
            item['speech'] = np.zeros(shape=[length], dtype=np.float32)
            item['early_reverb_signal'] = np.zeros(
                shape=[length], dtype=np.float32
            )  # (num_samples,)
        elif early_reverb_signal == 'use_clean':
            item['speech'] = clean_data
            item['early_reverb_signal'] = clean_data  # (num_samples,)
        if noise_flag:
            noisy_noreverb = item['speech'] + noise_data[:160000]
        else:
            noisy_noreverb = item['speech']
        item['noisy_noreverb'] = noisy_noreverb  # (num_samples,)

        return item

    def simulate_denoise_valid(self, item):
        ''' '''
        if item is None or 'speech' not in item:
            return item
        length = self._max_speech_length

        clean = self._data_manager.fix_audio_length(
            waveform=item['speech'], length=length
        )  # (length,)

        noise = self._data_manager.fix_audio_length(
            waveform=item['noise'], position=item['position'], length=length
        )  # (length,)

        rir_cfg_key = self.parse_block('clean')
        rir_data, index_rir = self._data_manager.load_multi_source_rir(cfg_key=rir_cfg_key)
        clean_data = np.stack([clean] * self.mic_num, axis=0)  # (mic, length,)
        clean_data_out_dict = dict(
            audio_data=clean_data,  # (mic_num, length)
            length=length,  # int
            rir_data=rir_data[0],  # (mic_num, rir_length)
            index_rir=index_rir[0],
            rir_type='denoise_rir',  # just do fftconv
        )
        noise_data = np.stack([noise] * self.mic_num, axis=0)
        noise_data_out_dict = dict(
            audio_data=noise_data,  # (mic_num, length)
            length=length,  # int
            rir_data=rir_data[1],  # (mic_num, rir_length)
            index_rir=index_rir[1],
            rir_type='denoise_rir',  # just do fftconv
        )
        early_reverb_signal = dict(
            audio_data=clean,  # (length,)
            rir_data=rir_data[0][0, 0 : index_rir[0] + 800],
            index_rir=index_rir[0],
            rir_type='denoise_rir',
        )
        rir_ss = np.zeros([length], dtype=np.float32)
        rir_ss[index_rir[0]] = rir_data[0][0, index_rir[0]]
        speech_data_out_dict = dict(
            audio_data=clean,  # (length,)
            rir_data=rir_ss,
            index_rir=index_rir[0],
            rir_type='denoise_rir',
        )
        item['clean'] = clean_data_out_dict
        item['noise'] = noise_data_out_dict
        item['speech'] = speech_data_out_dict
        item['early_reverb_signal'] = early_reverb_signal
        # noisy_noreverb = clean + noise[:160000]
        noisy_noreverb = noise[:160000]
        item['noisy_noreverb'] = noisy_noreverb  # (num_samples,)
        item['length'] = length
        return item

    def simulate_aec(self, item):
        return aec_load_simulate_data(self, item)

    def __call__(self, item, **kwargs):
        '''
        call func
        '''
        local_process_num = kwargs.get('local_process_num')
        local_pid = kwargs.get('local_pid')
        self._data_manager.fetcher_reader_init(local_process_num, local_pid)
        return self.simulate_fn(item)


@PREPROCESS.register_module()
class SimulatorWaveformCollate:
    '''collate simulator waveforms'''

    def __init__(self, simulate_type='simulate', simulator_config_path=''):
        '''init'''
        self.simulate_type = simulate_type
        cfg = Config.fromfile(simulator_config_path).cfg_dict
        self._cfg = cfg
        self.mic_num = self._cfg.general.mic_num
        self.simulator_fn = getattr(self, simulate_type)

    def collate_audio_with_direct_rir(self, key, batch_data):
        ''' '''
        mic_num = self.mic_num
        max_audio_length, max_rir_length = 0, 0
        batch_idxs = []
        for bid, item in enumerate(batch_data):
            if item[key] is not None and item[key]['rir_type'] == 'direct':
                batch_idxs.append(bid)
                max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
                max_rir_length = max(max_rir_length, item[key]['rir_data'].shape[-1])
        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(batch_idxs=batch_idxs, audio_data=None, rir_data=None, mask=None)
        audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
        rir_data_tensor = torch.zeros(bsz, mic_num, max_rir_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)

        for tid, bid in enumerate(batch_idxs):
            audio_data = batch_data[bid][key]['audio_data']
            rir_data = batch_data[bid][key]['rir_data']
            audio_data_tensor[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            rir_data_tensor[tid, :, 0 : rir_data.shape[-1]] = torch.from_numpy(rir_data)
            mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1
        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            rir_data=rir_data_tensor,
            mask=mask_data_tensor,
        )

    def collate_audio_with_diffuse_rir(self, key, batch_data):
        ''' '''
        mic_num = self.mic_num
        max_audio_length, max_rir_length = 0, 0
        batch_idxs = []
        audio_data2_idxs = []

        for bid, item in enumerate(batch_data):
            if item[key] is not None and item[key]['rir_type'] == 'diffuse':
                max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
                max_rir_length = max(max_rir_length, item[key]['rir_data'].shape[-1])
                if item[key]['audio_data2'] is not None:
                    audio_data2_idxs.append(len(batch_idxs))
                batch_idxs.append(bid)
        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(
                batch_idxs=batch_idxs,
                audio_data=None,
                rir_data=None,
                batch_idxs2=audio_data2_idxs,
                audio_data2=None,
                rir_data2=None,
                mask=None,
            )
        audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
        rir_data_tensor = torch.zeros(bsz, mic_num, max_rir_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)
        audio_data_tensor2 = torch.zeros(
            len(audio_data2_idxs), mic_num, max_audio_length, dtype=torch.float32
        )
        rir_data_tensor2 = torch.zeros(
            len(audio_data2_idxs), mic_num, max_rir_length, dtype=torch.float32
        )
        mask_data_tensor2 = torch.zeros(bsz, max_audio_length, dtype=torch.float32)

        for tid, bid in enumerate(batch_idxs):
            audio_data = batch_data[bid][key]['audio_data']
            rir_data = batch_data[bid][key]['rir_data']
            audio_data_tensor[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            rir_data_tensor[tid] = torch.from_numpy(rir_data)
            mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1

        for tid, idx in enumerate(audio_data2_idxs):
            bid = batch_idxs[idx]
            audio_data = batch_data[bid][key]['audio_data2']
            rir_data = batch_data[bid][key]['rir_data2']
            audio_data_tensor2[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            rir_data_tensor2[tid, :, 0 : rir_data.shape[-1]] = torch.from_numpy(rir_data)
            mask_data_tensor2[tid, 0 : audio_data.shape[-1]] = 1

        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            rir_data=rir_data_tensor,
            mask=mask_data_tensor,
            batch_idxs2=audio_data2_idxs,
            audio_data2=audio_data_tensor2,
            rir_data2=rir_data_tensor2,
            mask2=mask_data_tensor2,
        )

    def collate_audio_with_diffuse_equ_rir(self, key, batch_data):
        '''collate'''
        mic_num = self.mic_num
        max_audio_length = 0
        batch_idxs = []
        for bid, item in enumerate(batch_data):
            if item[key] is not None and item[key]['rir_type'] == 'diffuse_equation':
                batch_idxs.append(bid)
                max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(batch_idxs=batch_idxs, audio_data=None, rir_data=None, mask=None)
        audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)

        rir_data_tensor = []
        for tid, bid in enumerate(batch_idxs):
            audio_data = batch_data[bid][key]['audio_data']
            rir_data = batch_data[bid][key]['rir_data']
            audio_data_tensor[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            rir_data_tensor.append(rir_data)
            mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1
        rir_data_tensor = torch.from_numpy(np.stack(rir_data_tensor, axis=0))
        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            rir_data=rir_data_tensor,
            mask=mask_data_tensor,
        )

    def collate_audio_with_denoise_rir(self, key, batch_data, mic_num=None):
        ''' '''
        if mic_num is None:
            mic_num = self.mic_num
        max_audio_length, max_rir_length = 0, 0
        batch_idxs = []
        for bid, item in enumerate(batch_data):
            if item[key] is not None and item[key]['rir_type'] == 'denoise_rir':
                batch_idxs.append(bid)
                max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
                max_rir_length = max(max_rir_length, item[key]['rir_data'].shape[-1])
        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(
                batch_idxs=batch_idxs, audio_data=None, rir_data=None, index_rir=None, mask=None
            )
        if mic_num <= 0:
            audio_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)
            rir_data_tensor = torch.zeros(bsz, max_rir_length, dtype=torch.float32)
        else:
            audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
            rir_data_tensor = torch.zeros(bsz, mic_num, max_rir_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)
        index_rir_list = []

        for tid, bid in enumerate(batch_idxs):
            audio_data = batch_data[bid][key]['audio_data']
            rir_data = batch_data[bid][key]['rir_data']
            index_rir = batch_data[bid][key]['index_rir']
            audio_data_tensor[tid, ..., 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            rir_data_tensor[tid, ..., 0 : rir_data.shape[-1]] = torch.from_numpy(rir_data)
            mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1
            index_rir_list.append(index_rir)
        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            rir_data=rir_data_tensor,
            index_rir=index_rir_list,
            mask=mask_data_tensor,
        )

    def collate_audio_with_colornoise(self, key, batch_data):
        ''' '''
        mic_num = self.mic_num
        max_audio_length = 0
        batch_idxs = []
        colornoise_type = ''
        random_datas = []
        for bid, item in enumerate(batch_data):
            if item[key] is not None:
                if isinstance(item[key], dict) and item[key]['rir_type'] in [
                    "correlated",
                    "uncorrelated",
                ]:
                    colornoise_type = item[key]['rir_type']
                    batch_idxs.append(bid)
                    if 'random_data' in item[key]:
                        random_datas.append(item[key]['random_data'])
                    else:
                        random_datas.append(1)
                    max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
                elif isinstance(item[key], list) and item[key][0]['rir_type'] in [
                    "correlated",
                    "uncorrelated",
                ]:
                    for val in item[key]:
                        colornoise_type = val['rir_type']
                        batch_idxs.append(bid)
                        if 'random_data' in val:
                            random_datas.append(val['random_data'])
                        else:
                            random_datas.append(1)
                        max_audio_length = max(max_audio_length, val['audio_data'].shape[-1])

        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(
                batch_idxs=batch_idxs, audio_data=None, rir_data=None, mask=None, random_data=None
            )

        audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)
        rir_data_tensor = []
        batch_idxs_set = set(batch_idxs)
        tid = 0
        for bid in batch_idxs_set:
            if isinstance(batch_data[bid][key], list):
                for item in batch_data[bid][key]:
                    audio_data = item['audio_data']
                    rir_data_tensor.append(item['rir_data'])
                    audio_data_tensor[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(
                        audio_data
                    )
                    mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1
                    tid += 1

        if colornoise_type == 'correlated':
            rir_data_tensor = torch.from_numpy(np.stack(rir_data_tensor, axis=0))
        elif colornoise_type == 'uncorrelated':
            rir_data_tensor = None
        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            rir_data=rir_data_tensor,
            mask=mask_data_tensor,
            random_data=torch.tensor(random_datas),
        )

    def collate_audio_without_rir(self, key, batch_data):
        '''collate audio only, no rir'''
        mic_num = self.mic_num
        max_audio_length = 0
        batch_idxs = []
        for bid, item in enumerate(batch_data):
            if item[key] is not None and item[key]['rir_type'] == 'norir':
                batch_idxs.append(bid)
                max_audio_length = max(max_audio_length, item[key]['audio_data'].shape[-1])
        bsz = len(batch_idxs)
        if bsz <= 0:
            return dict(batch_idxs=batch_idxs, audio_data=None, rir_data=None, mask=None)
        audio_data_tensor = torch.zeros(bsz, mic_num, max_audio_length, dtype=torch.float32)
        mask_data_tensor = torch.zeros(bsz, max_audio_length, dtype=torch.float32)

        for tid, bid in enumerate(batch_idxs):
            audio_data = batch_data[bid][key]['audio_data']
            audio_data_tensor[tid, :, 0 : audio_data.shape[-1]] = torch.from_numpy(audio_data)
            mask_data_tensor[tid, 0 : audio_data.shape[-1]] = 1
        return dict(
            batch_idxs=batch_idxs,
            audio_data=audio_data_tensor,
            mask=mask_data_tensor,
        )

    def collate_snr_list(self, batch_data):
        ''' '''
        snr = []
        for item in batch_data:
            snr.append(item['snr'])
        return torch.Tensor(snr)

    def collate_audio_and_rir(self, key, batch_data):
        '''collate audio and rir'''
        outdict = dict()
        outdict['direct'] = self.collate_audio_with_direct_rir(key, batch_data)
        outdict['diffuse'] = self.collate_audio_with_diffuse_rir(key, batch_data)
        outdict['diffuse_equation'] = self.collate_audio_with_diffuse_equ_rir(key, batch_data)
        outdict['norir'] = self.collate_audio_without_rir(key, batch_data)
        return outdict

    def simulate(self, batch_data, batch_out):
        '''simulator'''
        # collate clean
        batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)
        # collate disturb
        batch_out['disturb'] = self.collate_audio_and_rir('disturb', batch_data)
        # collate noise
        batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)

    def simulate_bf(self, batch_data, batch_out):
        '''simulator_bf'''
        # collate clean
        batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)
        # collate clean2
        batch_out['clean2'] = self.collate_audio_and_rir('clean2', batch_data)
        batch_out['lengths2'] = [
            item['clean2']['length'] for item in batch_data if item['clean2'] is not None
        ]
        # collate disturb
        batch_out['disturb'] = self.collate_audio_and_rir('disturb', batch_data)
        # collate noise
        batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)
        # collate wnoise
        batch_out['wnoise'] = self.collate_audio_with_colornoise('wnoise', batch_data)

    def simulate_ssl(self, batch_data, batch_out):
        # collate clean
        batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)

        # collate disturb
        batch_out['disturb'] = self.collate_audio_and_rir('disturb', batch_data)
        # collate noise
        batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)
        # collate wnoise
        batch_out['wnoise'] = self.collate_audio_with_colornoise('wnoise', batch_data)

    def simulate_denoise(self, batch_data, batch_out):
        # collate clean
        batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)
        # collate noise
        batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)
        # collate colornoise
        batch_out['wnoise'] = self.collate_audio_with_colornoise('wnoise', batch_data)

    def simulate_denoise_valid(self, batch_data, batch_out):
        ''''''
        batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)
        batch_out['clean']['denoise_rir'] = self.collate_audio_with_denoise_rir('clean', batch_data)
        # collate noise
        batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)
        batch_out['noise']['denoise_rir'] = self.collate_audio_with_denoise_rir('noise', batch_data)
        # collate noise
        batch_out['speech'] = dict()
        batch_out['speech']['denoise_rir'] = self.collate_audio_with_denoise_rir(
            'speech', batch_data, mic_num=0
        )
        # collate noise
        batch_out['early_reverb_signal'] = dict()
        batch_out['early_reverb_signal']['denoise_rir'] = self.collate_audio_with_denoise_rir(
            'early_reverb_signal', batch_data, mic_num=0
        )

    def simulate_aec(self, batch_data, batch_out):
        aec_simulate_collate_fn(self, batch_data, batch_out)

    def __call__(self, batch_data, batch_out):
        '''call func'''
        batch_out['snr'] = self.collate_snr_list(batch_data)
        bsz = len(batch_data)
        mic_num = self.mic_num
        max_length = max(item['length'] for item in batch_data)
        batch_out['batch_shape'] = (bsz, mic_num, max_length)
        batch_out['lengths'] = [item['length'] for item in batch_data]
        self.simulator_fn(batch_data, batch_out)


@PREPROCESS.register_module()
class SimulatorModule:
    """Simulator"""

    def __init__(self, simulate_type='simulate', simulator_config_path=''):
        self.simulate_type = simulate_type
        cfg = Config.fromfile(simulator_config_path).cfg_dict
        self._cfg = cfg
        self.simulate_type = simulate_type
        self.simulator_fn = getattr(self, simulate_type)
        self.diffuse_module = DiffuseNoise()

    def mix(self, speech, noise, snr, out_snr=False, use_vad=False, sign_clip=None):
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
        gen_cfg = self._cfg.general
        frame_size = gen_cfg.frame_size
        speech_t = speech[:, 0, ...]
        noise_t = noise[:, 0, ...]
        max_length = speech.shape[-1]
        n_frames = int(max_length / frame_size)
        speech_framed = speech_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)
        noise_framed = noise_t[..., : n_frames * frame_size].reshape(bsz, n_frames, frame_size)

        if use_vad:
            speech_vad = (torch.abs(speech_framed).mean(-1) > 10.0).float()
            noise_vad = (torch.abs(noise_framed).mean(-1) > 10.0).float()
            speech_framed *= speech_vad.unsqueeze(-1)
            noise_framed *= noise_vad.unsqueeze(-1)
            speech_power = torch.mean(torch.mean((speech_framed**2), dim=-1), dim=-1)
            noise_power = torch.mean(torch.mean((noise_framed**2), dim=-1), dim=-1)
        else:
            speech_power = torch.mean((speech_t**2), dim=-1)
            noise_power = torch.mean((noise_t**2), dim=-1)
        speech_power[speech_power == 0] = torch.mean(
            speech_power
        )  # using averge power when not adding clean
        noise_scale = torch.sqrt(
            speech_power / (noise_power + gen_cfg.min_limit) / torch.pow(10.0, snr / 10.0)
        ).reshape(-1, 1, 1)
        if out_snr:
            noise_scale = torch.sqrt(speech_power / (noise_power + gen_cfg.min_limit))
            if sign_clip is not None:
                noise_scale2 = (
                    torch.max(torch.abs(speech_t), dim=-1).values
                    / torch.max(torch.abs(noise_t), dim=-1).values
                )
                sign_no_clip = torch.ones_like(sign_clip, device=sign_clip.device) - sign_clip
                noise_scale = noise_scale * sign_no_clip + noise_scale2 * sign_clip
            noise_scale = noise_scale / torch.pow(10.0, snr / 10.0)
            noise_scale = noise_scale.reshape(-1, 1, 1)

        noise = noise * noise_scale
        noisy = speech + noise
        return noise, noisy, snr

    def rand_sync_agc_array_cpu(self, data):
        """Calculate agc ratio"""
        gen_cfg = self._cfg.general
        agc_value = torch.cuda.FloatTensor(data.shape[0]).uniform_(gen_cfg.agc_min, gen_cfg.agc_max)
        data_t = data[:, 0, :]
        cur_max = torch.abs(data_t).max(dim=-1).values
        agc_ratio = agc_value / (cur_max + gen_cfg.min_limit)
        return agc_ratio, agc_value

    def rand_agc_ratio(self, data):
        '''generate random agc ratio
        used in aec task
        '''
        gen_cfg = self._cfg.general
        bsz = data.shape[0]
        random_data = torch.rand(bsz, device=data.device)
        agc_value = random_data * (gen_cfg.agc_max - gen_cfg.agc_min) + gen_cfg.agc_min
        agc_ratio = (
            1
            / (torch.max(torch.max(torch.abs(data), dim=-1).values, dim=-1).values + 1e-3)
            * agc_value
        )
        agc_ratio = agc_ratio.reshape(-1, 1, 1)
        return agc_ratio

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
        bsz = noisy_data.shape[0]
        agc_ratio, agc_val = self.rand_sync_agc_array_cpu(noisy_data)
        # [bsz * mic_num * sampled_len ] * [bsz, 1, 1]
        agc_ratio = agc_ratio.reshape(noisy_data.shape[0], 1, 1)
        noisy_data = noisy_data * agc_ratio
        args = list(args)
        for i in range(len(args)):
            args[i] *= agc_ratio

        vol_cfg = self._cfg.volume_rand
        mic_num = self._cfg.general.mic_num
        amp_low = vol_cfg.amp_low
        amp_high = vol_cfg.amp_high
        rand_rto = (
            torch.clamp(
                torch.randn(bsz * mic_num, device=noisy_data.device).reshape(bsz, mic_num, 1)
                * vol_cfg.variance,
                -1,
                1,
            )
            * 0.5
            + 0.5
        )
        amp_use = rand_rto * (amp_high - amp_low) + amp_low
        noisy_data *= amp_use
        return noisy_data, args, agc_val

    def rand_sync_agc_n(self, *datas, agc_values=None, min_limit=None):
        ''' '''
        gen_cfg = self._cfg.general
        if min_limit is None:
            min_limit = gen_cfg.min_limit
        if agc_values is None:
            agc_values = torch.cuda.FloatTensor(datas[0].shape[0]).uniform_(
                gen_cfg.agc_min, gen_cfg.agc_max
            )
        max_in = torch.zeros(datas[0].shape[0], device=datas[0].device)  # bsz
        for data in datas:
            cur_max = data.max(dim=-1).values
            if data.ndim == 3:
                cur_max = cur_max.max(dim=-1).values
            max_in = torch.max(max_in, cur_max)

        agc_ratios = agc_values / (max_in + min_limit)
        agc_ratios = agc_ratios.reshape(datas[0].shape[0], 1, 1)
        datas = list(datas)
        for i in range(len(datas)):
            if datas[i].ndim == 2:
                datas[i] *= agc_ratios[:, :, 0]
            elif datas[i].ndim == 3:
                datas[i] *= agc_ratios
        return datas

    def fftconvolve(self, audio_data, rir, mask, index=None):
        '''do fftconvolve
        Args:
            audio_data(Tensor): input audio data, (bsz, mic_num, num_samples)
            rir_data(Tensor): input rir data, (bsz, mic_num, rir_length)
            mask(Tensor): input mask data, (bsz, num_samples)
        Return:
            audio(Tensor): mixed signals, (bsz, mic_num, num_samples)
        '''
        audios = fftconvolve(audio_data, rir, axes=-1)
        if index is None:
            audio = audios[..., : audio_data.shape[-1]]
        else:
            audio = torch.zeros_like(audio_data, device=audio_data.device)
            for bid in range(audio_data.shape[0]):
                audio[bid] = audios[bid, ..., index[bid] : index[bid] + audio_data.shape[-1]]
        if audio.ndim == 3:
            audio *= mask.unsqueeze(1)
        else:
            audio *= mask
        return audio

    def calculate_with_direct_rir(self, data):
        '''calculate with direct rir'''
        if data is None or data['batch_idxs'] is None or len(data['batch_idxs']) <= 0:
            return [], None
        batch_idxs = data['batch_idxs']
        audio_data = data['audio_data']
        rir_data = data['rir_data']
        mask = data['mask']
        audio = self.fftconvolve(audio_data, rir_data, mask)
        return batch_idxs, audio

    def calculate_with_denoise_rir(self, data):
        ''''''
        if data is None or data['batch_idxs'] is None or len(data['batch_idxs']) <= 0:
            return [], None
        batch_idxs = data['batch_idxs']
        audio_data = data['audio_data']
        rir_data = data['rir_data']
        index_rir = data['index_rir']
        mask = data['mask']
        audio = self.fftconvolve(audio_data, rir_data, mask, index=index_rir)
        return batch_idxs, audio

    def mix_truc_diffuse_noise(self, noise_data_diffuse):
        '''mix truc diffuse noise
        Args:
            noise_data_diffuse(Tensor): noise data generate with diffuse rir (bsz, mic_num, num_samples)

        Return:
            noise(Tensor): noise after mixed (bsz, mic_num, num_samples)
        '''
        # TODO(litianyu.y): check logic here

        energy = torch.mean(noise_data_diffuse**2, -1) + 1e-8
        energy = torch.sqrt(energy[:, 0] / energy[:, 1])
        noise_data_diffuse[:, 1] = torch.bmm(
            noise_data_diffuse[:, 1].unsqueeze(-1), energy.reshape(-1, 1, 1)
        )[..., 0]
        return noise_data_diffuse

    def calculate_with_diffuse_rir(self, data):
        '''calculate with diffuse rir'''
        if data is None or data['batch_idxs'] is None or len(data['batch_idxs']) <= 0:
            return [], None
        batch_idxs = data['batch_idxs']
        audio_data = data['audio_data']
        rir_data = data['rir_data']
        mask = data['mask']
        audio = self.fftconvolve(audio_data, rir_data, mask)
        if data['batch_idxs2'] is None or len(data['batch_idxs2']) <= 0:
            return batch_idxs, self.mix_truc_diffuse_noise(audio)
        batch_idxs2 = data['batch_idxs2']
        audio_data2 = data['audio_data2']
        rir_data2 = data['rir_data2']
        mask2 = data['mask2']
        audio2 = self.fftconvolve(audio_data2, rir_data2, mask2)
        for tid, bid in enumerate(batch_idxs2):
            audio[bid, :, : audio2[tid].shape[-1]] += audio2[tid]
        return batch_idxs, self.mix_truc_diffuse_noise(audio)

    def calucate_with_diffuse_equ_rir(self, data):
        '''calucate_with_diffuse_equ_rir'''
        if data is None or data['batch_idxs'] is None or len(data['batch_idxs']) <= 0:
            return [], None
        batch_idxs = data['batch_idxs']
        audio_data = data['audio_data']
        rir_data = data['rir_data']
        audio = self.diffuse_module.mix_signals(audio_data, rir_data)
        return batch_idxs, audio

    def calucate_without_rir(self, data):
        '''calculate without rir'''
        if data is None or data['batch_idxs'] is None or len(data['batch_idxs']) <= 0:
            return [], None
        batch_idxs = data['batch_idxs']
        audio_data = data['audio_data']
        return batch_idxs, audio_data

    def get_colornoise_data(self, data, bsz, mic_num, length, device):
        '''
        get colornoise data
        '''
        data_out = torch.zeros(bsz, mic_num, length, device=device)
        if data['rir_data'] is None:
            batch_idxs, audio = data['batch_idxs'], data['audio_data']
        else:
            batch_idxs, audio = self.calculate_with_direct_rir(data)
        random_data = data['random_data']
        if batch_idxs:
            for tid, bid in enumerate(batch_idxs):
                data_out[bid] += audio[tid] * random_data[tid]
        return data_out

    def get_audio_data(self, data, bsz, mic_num, length, device):
        '''
        calculate to get real audio_data
        '''
        data_out = torch.zeros(bsz, mic_num, length, device=device)
        batch_idxs, audio = self.calculate_with_direct_rir(data['direct'])
        if batch_idxs:
            for tid, bid in enumerate(batch_idxs):
                data_out[bid, :, : audio[tid].shape[-1]] = audio[tid]
        batch_idxs, audio = self.calculate_with_diffuse_rir(data['diffuse'])
        if batch_idxs:
            for tid, bid in enumerate(batch_idxs):
                data_out[bid, :, : audio[tid].shape[-1]] = audio[tid]
        batch_idxs, audio = self.calucate_with_diffuse_equ_rir(data['diffuse_equation'])
        if batch_idxs:
            for tid, bid in enumerate(batch_idxs):
                data_out[bid, :, : audio[tid].shape[-1]] = audio[tid]
        batch_idxs, audio = self.calucate_without_rir(data['norir'])
        if batch_idxs:
            for tid, bid in enumerate(batch_idxs):
                data_out[bid, :, : audio[tid].shape[-1]] = audio[tid]

        if 'denoise_rir' in data:
            batch_idxs, audio = self.calculate_with_denoise_rir(data['denoise_rir'])
            if batch_idxs:
                for tid, bid in enumerate(batch_idxs):
                    data_out[bid, ..., : audio[tid].shape[-1]] = audio[tid]
        return data_out

    def simulate(self, item):
        '''simulator calculate func
        There are three types of sound sources:
            clean: must exits, clean audio signal, bsz * mic_num * sampled_len
            disturb: may not exits, disturb audio signal, bsz * mic_num * sampled_len
            noise: may not exits, disturb audio signal, bsz * mic_num * sampled_len

        The calculation logic is:
            clean_data, disturb_data, noise_data will perform fft convolution operation,
            then will mix all three source signals,
            finally will perform  randomize_volume operation.
        '''
        snr = item.pop('snr')
        shape = item.pop('batch_shape')
        bsz, mic_num, length = shape

        # clean data
        clean_data = item.pop('clean')
        clean_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
        noisy_data = clean_data.clone()

        # disturb data
        disturb_data = item.pop('disturb')
        disturb_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
        # mix clean data and noisy data
        disturb_data, noisy_data, _ = self.mix(
            noisy_data,
            disturb_data,
            snr[:, 0],
        )

        # noise_data
        noise_data = item.pop('noise')
        noise_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)

        noise_data, noisy_data, _ = self.mix(
            noisy_data,
            noise_data,
            snr[:, 1],
        )

        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data,
            clean_data,
            disturb_data,
        )

        item['mc_waveform'] = noisy_data
        item['target_waveform'] = clean_data
        item['disturb_waveform'] = disturb_data
        return item

    def simulate_ssl(self, item):
        """Simulator entry for ssl training."""
        snr = item.pop('snr')
        shape = item.pop('batch_shape')
        bsz, mic_num, length = shape

        # clean data
        clean_data = item.pop('clean')
        clean_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
        noisy_data = clean_data.clone().detach()
        # disturb data
        disturb_data = item.pop('disturb')
        disturb_data = self.get_audio_data(disturb_data, bsz, mic_num, length, snr.device)
        # mix clean data and noisy data
        disturb_data, noisy_data, _ = self.mix(
            noisy_data,
            disturb_data,
            snr[:, 0],
        )
        # noise_data
        noise_data = item.pop('noise')
        noise_data = self.get_audio_data(noise_data, bsz, mic_num, length, snr.device)
        noise_data, noisy_data, _ = self.mix(
            noisy_data,
            noise_data,
            snr[:, 1],
        )
        # wnoise_data
        wnoise_data = item.pop('wnoise')
        wnoise_data = self.get_colornoise_data(wnoise_data, bsz, mic_num, length, snr.device)

        wnoise_data, noisy_data, _ = self.mix(
            noisy_data,
            wnoise_data,
            snr[:, 2],
        )
        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data,
            clean_data,
            disturb_data,
        )
        directional_data = torch.zeros(bsz, 2, length, device=clean_data.device)
        directional_data[:, 0, :] = clean_data[:, 0, :]
        directional_data[:, 1, :] = disturb_data[:, 0, :]
        item['mc_waveform'] = noisy_data
        item['directional_waveform'] = directional_data

        item['target_waveform'] = clean_data
        item['disturb_waveform'] = disturb_data
        item['noise_waveform'] = noise_data
        return item

    def concatenate_clean_data(self, clean, clean2, batch_idxs, lengths, length2s):
        '''concatenate clean data in simulate_bf'''
        if len(batch_idxs) <= 0 or clean2 is None:
            return clean
        for idx, bid in enumerate(batch_idxs):
            length2 = length2s[idx]
            length = lengths[bid]
            clean[bid, :, length - length2 :] = clean2[idx, :, length - length2 :]
        return clean

    def simulate_bf(self, item):
        """Simulator entry for bf training.

        Args:
            item: input dict.

        Return:
            item: include data simulated.
        """
        snr = item.pop('snr')
        shape = item.pop('batch_shape')
        bsz, mic_num, length = shape

        # clean data
        clean = item.pop('clean')
        clean_data = self.get_audio_data(clean, bsz, mic_num, length, snr.device)
        clean2 = item.pop('clean2')
        batch_idxs, clean_data2 = self.calculate_with_direct_rir(clean2['direct'])
        clean_data = self.concatenate_clean_data(
            clean=clean_data,
            clean2=clean_data2,
            batch_idxs=batch_idxs,
            lengths=item['lengths'],
            length2s=item['lengths2'],
        )
        noisy_data = clean_data.clone()

        # disturb data
        disturb = item.pop('disturb')
        disturb_data = self.get_audio_data(disturb, bsz, mic_num, length, snr.device)
        # mix clean data and noisy data
        disturb_data, noisy_data, _ = self.mix(
            noisy_data,
            disturb_data,
            snr[:, 0],
        )

        # noise_data
        noise = item.pop('noise')
        noise_data = self.get_audio_data(noise, bsz, mic_num, length, snr.device)
        noise_data, noisy_data, _ = self.mix(
            noisy_data,
            noise_data,
            snr[:, 1],
        )
        # wnoise_data
        wnoise = item.pop('wnoise')
        wnoise_data = self.get_colornoise_data(wnoise, bsz, mic_num, length, snr.device)
        wnoise_data, noisy_data, _ = self.mix(
            noisy_data,
            wnoise_data,
            snr[:, 2],
        )

        noisy_data, (clean_data, disturb_data), _ = self.randomize_volume(
            noisy_data,
            clean_data,
            disturb_data,
        )

        item['target_waveform'] = clean_data
        item['disturb_waveform'] = disturb_data
        item['mc_waveform'] = noisy_data
        item['target_sou_data'] = clean_data[:, 0, :]
        directional_data = torch.zeros(bsz, 2, length, device=clean_data.device)
        directional_data[:, 0, :] = clean_data[:, 0, :]
        directional_data[:, 1, :] = disturb_data[:, 0, :]
        item['directional_data'] = directional_data
        return item

    def simulate_denoise(self, item):
        '''
        simulate denoise
        '''
        snr = item.pop('snr')
        shape = item.pop('batch_shape')
        bsz, mic_num, length = shape

        # clean data
        clean_data = item.pop('clean')
        clean_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
        noisy_data = clean_data.clone()

        # noise_data
        noise_data = item.pop('noise')
        noise_data = self.get_audio_data(noise_data, bsz, mic_num, length, snr.device)
        noise_data, noisy_data, _ = self.mix(
            noisy_data,
            noise_data,
            snr[:, 0],
            True,
        )
        # wnoise_data
        wnoise = item.pop('wnoise')

        wnoise_data = self.get_colornoise_data(wnoise, bsz, mic_num, length, snr.device)
        wnoise_data, _, _ = self.mix(clean_data, wnoise_data, snr[:, 1], True)
        noise_data = noise_data + wnoise_data
        noisy_data = noisy_data + wnoise_data

        (
            early_reverb_signal,
            speech,
            noisy_data,
            noise_data,
            noisy_noreverb,
            clean_data,
        ) = self.rand_sync_agc_n(
            item['early_reverb_signal'],
            item['speech'],
            noisy_data,
            noise_data,
            item['noisy_noreverb'],
            clean_data,
            agc_values=None,
        )

        item['speech_waveform'] = clean_data[:, 0:1, :]  # simu mc speech
        item['noisy_waveform'] = noisy_data  # simu mc noisy
        return item

    def simulate_denoise_valid(self, item):
        '''
        simulate denoise
        '''
        snr = item.pop('snr')
        shape = item.pop('batch_shape')
        bsz, mic_num, length = shape

        # clean data
        clean_data = item.pop('clean')
        clean_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
        noisy_data = clean_data.clone()

        # noise_data
        noise_data = item.pop('noise')
        noise_data = self.get_audio_data(noise_data, bsz, mic_num, length, snr.device)
        noise_data, noisy_data, _ = self.mix(
            noisy_data, noise_data, snr, True, True, item['sign_clip']
        )
        # early_reverb_signal
        early_reverb_signal = item.pop('early_reverb_signal')
        _, early_reverb_signal = self.calculate_with_denoise_rir(early_reverb_signal['denoise_rir'])

        speech = item.pop('speech')
        _, speech = self.calculate_with_denoise_rir(speech['denoise_rir'])

        item['noisy_noreverb'] += speech

        (
            early_reverb_signal,
            speech,
            noisy_data,
            noise_data,
            noisy_noreverb,
            clean_data,
        ) = self.rand_sync_agc_n(
            early_reverb_signal,
            speech,
            noisy_data,
            noise_data,
            item['noisy_noreverb'],
            clean_data,
            agc_values=item['agc'],
            min_limit=1e-5,
        )
        item['speech_waveform'] = clean_data[:, 0:1, :]  # simu mc speech
        item['noisy_waveform'] = noisy_data  # simu mc noisy
        return item

    def simulate_aec(self, item):
        '''aec simulator fn'''
        if not hasattr(self, 'aec_clibs_init_flag'):
            aec_clibs_file = self._cfg.general.get(
                'aec_clibs_file',
                'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/litianyu.y/se/aec/data/clibs',
            )
            dist_file_get(remote_file=aec_clibs_file)
            self.aec_clibs_init_flag = True
        return aec_simulator_fn(self, item)

    def simulate_aec_infer(self, item):
        '''aec inference simulator fn'''
        return aec_infer_simulator_fn(self, item)

    def __call__(self, item):
        '''call func'''
        return self.simulator_fn(item)
