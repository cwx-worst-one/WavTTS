'''
rir processing
'''
import random
from collections import deque
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from packaging import version
from dataloader import FalconReader
from core.extensions import kaldi_add_rir
from core.utils import complex_multiply
from core.utils.dist_util import get_local_rank, get_local_size
from .wav import WavParser
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class RIRSimulator(nn.Module):
    """RIR Simulator"""

    def __init__(
        self,
        speech_rir_ratio,
        disturb_rir_ratio,
        disturb_ratio,
        noise_rir_ratio,
        noise_ratio,
        sir_max,
        sir_min,
        snr_max,
        snr_min,
        agc_max,
        agc_min,
        use_agc,
        normalize,
        target_dereverb=False,
        rir_cut_length=0,
        rir_decay_rate=-1,
        rir_direct_duration=0,
    ):
        super().__init__()
        self.speech_rir_ratio = speech_rir_ratio
        self.disturb_rir_ratio = disturb_rir_ratio
        self.noise_rir_ratio = noise_rir_ratio
        self.disturb_ratio = disturb_ratio
        self.noise_ratio = noise_ratio
        self.sir_max = sir_max
        self.sir_min = sir_min
        self.snr_max = snr_max
        self.snr_min = snr_min
        self.agc_max = agc_max
        self.agc_min = agc_min
        self.use_agc = use_agc
        self.normalize = normalize
        self.target_dereverb = target_dereverb
        self.rir_cut_length = rir_cut_length
        self.rir_decay_rate = rir_decay_rate
        self.rir_direct_duration = rir_direct_duration

    # pylint: disable=no-else-return
    @staticmethod
    def agc_scale(mic1, mic2, clean, agc_values):
        """agc scale"""
        if mic2 is not None:
            input_maxs, _ = torch.max(torch.abs(torch.cat((mic1, mic2, clean), dim=-1)), dim=1)
            agc_ratios = agc_values / input_maxs
            mic1_scaled = mic1 * torch.unsqueeze(agc_ratios, dim=1)
            mic2_scaled = mic2 * torch.unsqueeze(agc_ratios, dim=1)
            clean_scaled = clean * torch.unsqueeze(agc_ratios, dim=1)
            return mic1_scaled, mic2_scaled, clean_scaled
        else:
            input_maxs, _ = torch.max(torch.abs(torch.cat((mic1, clean), dim=-1)), dim=1)
            agc_ratios = agc_values / input_maxs
            mic1_scaled = mic1 * torch.unsqueeze(agc_ratios, dim=1)
            clean_scaled = clean * torch.unsqueeze(agc_ratios, dim=1)
            return mic1_scaled, clean_scaled, None

    # pylint: disable=invalid-name
    @staticmethod
    def _add_nonspeech(
        batch_input, batch_nonspeech, num_of_frames, frame_size, dB, add_flags, batch_size
    ):
        """add non speech"""
        batch_input_framed = batch_input[:, : num_of_frames * frame_size].view(
            batch_size, num_of_frames, frame_size
        )
        batch_nonspeech_framed = batch_nonspeech[:, : num_of_frames * frame_size].view(
            batch_size, num_of_frames, frame_size
        )
        input_powers, _ = torch.max(torch.mean((batch_input_framed**2), dim=2), dim=1)
        nonspeech_powers, _ = torch.max(torch.mean((batch_nonspeech_framed**2), dim=2), dim=1)
        nonspeech_scales = torch.sqrt(
            input_powers / (nonspeech_powers + 1e-5) / torch.pow(10.0, dB / 10)
        )
        batch_output = batch_input + batch_nonspeech * torch.unsqueeze(
            nonspeech_scales, dim=1
        ) * torch.unsqueeze(add_flags, dim=1)
        return batch_output

    # pylint: disable=invalid-name, no-else-return
    def add_nonspeech(
        self,
        batch_input_1,
        batch_input_2,
        batch_nonspeech_1,
        batch_nonspeech_2,
        frame_size,
        nonspeech_ratio,
        dB_max,
        dB_min,
    ):
        """add nonspeech"""
        # batch_input, batch_nonspeech: [batch, length]
        batch_size, length = batch_input_1.size()
        add_nonspeech_flags = (
            torch.rand(batch_size, device=batch_input_1.device, dtype=batch_input_1.dtype)
            < nonspeech_ratio
        ).to(batch_input_1.dtype)
        dBs = dB_min + (dB_max - dB_min) * torch.rand(
            batch_size, device=batch_input_1.device, dtype=batch_input_1.dtype
        )
        num_of_frames = int(length / frame_size)
        batch_output_1 = self._add_nonspeech(
            batch_input_1,
            batch_nonspeech_1,
            num_of_frames,
            frame_size,
            dBs,
            add_nonspeech_flags,
            batch_size,
        )

        if batch_input_2 is None:
            return batch_output_1
        else:
            batch_output_2 = self._add_nonspeech(
                batch_input_2,
                batch_nonspeech_2,
                num_of_frames,
                frame_size,
                dBs,
                add_nonspeech_flags,
                batch_size,
            )
            return batch_output_1, batch_output_2

    @staticmethod
    def fft_convolution(x, h, rir_ratio, mode="same"):
        """fft convolution"""
        batch_size = x.size()[0]
        norir_ids = torch.rand(batch_size, device=x.device) > rir_ratio
        xl = x.size()[-1]
        hl = h.size()[-1]
        length = xl + hl - 1
        x_padded = F.pad(x, [0, length - xl], mode='constant', value=0.0)
        h_padded = F.pad(h, [0, length - hl], mode='constant', value=0.0)
        if version.parse(torch.__version__) >= version.parse('1.7.0'):
            # fix bug in torch 1.8.1
            # docs: https://github.com/pytorch/pytorch/pull/63327/files
            if version.parse(torch.__version__) <= version.parse('1.10.0'):
                torch.backends.cuda.cufft_plan_cache.max_size = 0
            fft_x = torch.fft.rfft(x_padded)
            fft_h = torch.fft.rfft(h_padded)
            fft_h.data[norir_ids] = 1.0
            x_conv = torch.fft.irfftn(
                complex_multiply(fft_x, fft_h),
                dim=-1,
                s=[length],
            )
        else:
            fft_x = torch.rfft(x_padded, signal_ndim=1, normalized=False, onesided=True)
            fft_h = torch.rfft(h_padded, signal_ndim=1, normalized=False, onesided=True)
            fft_h.data[norir_ids] = 1.0
            x_conv = torch.irfft(
                complex_multiply(fft_x, fft_h),
                signal_ndim=1,
                normalized=False,
                onesided=True,
                signal_sizes=[length],
            )
        x_out = None
        if mode == "full":
            x_out = x_conv
        elif mode == "same":
            x_out = x_conv[:, :xl]
        return x_out

    # pylint: disable=no-else-return
    @torch.no_grad()
    def forward(self, clean, disturb, noise, rir):
        """forward"""
        # pylint:disable=too-many-branches
        batch_size = clean.size()[0]
        if disturb is not None:
            if self.target_dereverb:
                if self.rir_cut_length > 0 and self.rir_cut_length < rir.shape[-1]:
                    direct_rir = rir[:, 0, : self.rir_cut_length]
                elif self.rir_decay_rate > 0 and self.rir_decay_rate < 1:
                    direct_rir = rir[:, 0, :].clone()
                    rir_len = direct_rir.size()[-1]
                    max_idx = torch.argmax(direct_rir.abs(), dim=1)
                    for i in range(batch_size):
                        direct_duration = int(self.rir_direct_duration)
                        tmp = rir_len - max_idx[i] - direct_duration
                        if tmp > 0:
                            direct_rir[i, max_idx[i] + direct_duration :] = direct_rir[
                                i, max_idx[i] + direct_duration :
                            ] * (
                                self.rir_decay_rate
                                ** torch.arange(start=0, end=tmp, device=direct_rir.device)
                            )
                else:
                    direct_rir = None
                clean_direct_1 = self.fft_convolution(clean, direct_rir, self.speech_rir_ratio)
            clean_reverb_1 = self.fft_convolution(clean, rir[:, 0, :], self.speech_rir_ratio)
            clean_reverb_2 = self.fft_convolution(clean, rir[:, 1, :], self.speech_rir_ratio)
            disturb_reverb_1 = self.fft_convolution(disturb, rir[:, 2, :], self.disturb_rir_ratio)
            disturb_reverb_2 = self.fft_convolution(disturb, rir[:, 3, :], self.disturb_rir_ratio)
            mic1_out, mic2_out = self.add_nonspeech(
                clean_reverb_1,
                clean_reverb_2,
                disturb_reverb_1,
                disturb_reverb_2,
                240,
                self.disturb_ratio,
                self.sir_max,
                self.sir_min,
            )
            if noise is not None:
                mic1_out, mic2_out = self.add_nonspeech(
                    mic1_out,
                    mic2_out,
                    noise,
                    noise,
                    240,
                    self.noise_ratio,
                    self.snr_max,
                    self.snr_min,
                )
            if self.target_dereverb:
                if self.use_agc:
                    agc_values = self.agc_min + (self.agc_max - self.agc_min) * torch.rand(
                        batch_size, device=clean.device, dtype=clean.dtype
                    )
                    mic1_out, mic2_out, target = self.agc_scale(
                        mic1_out, mic2_out, clean_direct_1, agc_values
                    )
                else:
                    target = clean_direct_1
            else:
                if self.use_agc:
                    agc_values = self.agc_min + (self.agc_max - self.agc_min) * torch.rand(
                        batch_size, device=clean.device, dtype=clean.dtype
                    )
                    mic1_out, mic2_out, target = self.agc_scale(
                        mic1_out, mic2_out, clean_reverb_1, agc_values
                    )
                else:
                    target = clean_reverb_1
            mic1_out = torch.stack((mic1_out, mic2_out), dim=-1)  # B T channel
            return mic1_out, target
        else:
            rir_idx = random.randint(0, 3)
            clean_reverb_1 = self.fft_convolution(clean, rir[:, rir_idx, :], self.speech_rir_ratio)

            noise_reverb_1 = self.fft_convolution(noise, rir[:, rir_idx, :], self.noise_rir_ratio)
            mic1_out = self.add_nonspeech(
                clean_reverb_1,
                None,
                noise_reverb_1,
                noise_reverb_1,
                240,
                self.noise_ratio,
                self.snr_max,
                self.snr_min,
            )
            # TODO remove agc
            if self.use_agc:
                agc_values = self.agc_min + (self.agc_max - self.agc_min) * torch.rand(
                    batch_size, device=clean.device, dtype=clean.dtype
                )
            else:
                target = clean_reverb_1
            mic1_out, target, _ = self.agc_scale(mic1_out, None, clean_reverb_1, agc_values)
            return mic1_out, target  # B T


@PREPROCESS.register_module()
class AddRIR:
    """Add RIR"""

    # pylint: disable=line-too-long
    ###############################################################################################
    # param definition could be found https://bytedance.feishu.cn/docs/doccnQK6ym2rCkOqHyGxqGmSRoe#
    ###############################################################################################
    def __init__(
        self,
        key="waveform",
        dir_diff_threshold=20,
        rir_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/yxl.19/dataset/rir/rir100w",
        rir_shards=6,
        rir_prefix="sub",
        rir_chunk=1,
        noise_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/yxl.19/dataset/noise/dy500h",
        noise_shards=3,
        noise_prefix="sub",
        noise_rate=16000,
        rir_length=2048,
        snr_max=20,
        snr_min=5,
        sir_max=15,
        sir_min=-5,
        agc_max=25000,
        agc_min=5000,
        use_agc=True,
        speech_rir_ratio=1.0,
        disturb_ratio=1.0,
        disturb_rir_ratio=1.0,
        noise_ratio=0,
        noise_rir_ratio=0,
        mic_num=2,
        noise_chunk=1,
        reuse_num=120,
        normalize=False,
        device='cpu',
        batch_size=20,
        parrallel_chunk_num=2,
    ):

        """
        snr_max: max signal-to-noise ratio, 1-mic only
        snr_min: min signal-to-noise ratio, 1-mic only
        agc_max: both for 1-mic and 2-mic
        agc_min: both for 1-mic and 2-mic
        use_agc: both for 1-mic and 2-mic
        speech_rir_ratio: both for 1-mic and 2-mic
        disturb_rir_ratio: 2-mic only
        noise_ratio: 1-mic only
        noise_rir_ratio: 1-mic only
        """
        # pylint:disable=too-many-locals
        self.rir_reader = None
        self.noise_reader = None
        self.data_sim = None

        self.dir_diff_threshold = dir_diff_threshold
        self.rir_dir = rir_dir
        self.rir_shards = rir_shards
        self.rir_prefix = rir_prefix
        self.noise_dir = noise_dir
        self.noise_shards = noise_shards
        self.noise_prefix = noise_prefix
        self.noise_rate = noise_rate
        self.noise_ratio = noise_ratio
        self.normalize = normalize
        self.rir_length = rir_length
        self.snr_max = snr_max
        self.snr_min = snr_min
        self.sir_max = sir_max
        self.sir_min = sir_min
        self.agc_max = agc_max
        self.agc_min = agc_min
        self.use_agc = use_agc
        self.speech_rir_ratio = speech_rir_ratio
        self.disturb_ratio = disturb_ratio
        self.disturb_rir_ratio = disturb_rir_ratio
        self.noise_rir_ratio = noise_rir_ratio
        self.mic_num = mic_num
        self.rir_chunk = rir_chunk
        self.noise_chunk = noise_chunk
        self.reuse_num = reuse_num
        self.reused = reuse_num
        self.noise_data = None
        self.rir = None
        self.rir_direction = None
        self.key = key
        self.device = device
        self.batch = []
        self.max_length = 0
        self.process_batch = []
        self.batch_size = batch_size
        self.parrallel_chunk_num = parrallel_chunk_num
        self.rir_chunks = deque()  # chunks buf
        self.rir_datas = deque()  # data buf
        self.noise_chunks = deque()  # chunks buf
        self.noise_datas = deque()  # data buf

    def get_reader(self, data_type):
        """get {data_type} reader"""
        if data_type == 'noise':
            if self.noise_reader is not None:
                return
            noise_list = [
                '{}/{}{}'.format(self.noise_dir, self.noise_prefix, shard)
                for shard in range(self.noise_shards)
            ]
            self.noise_reader = FalconReader(
                noise_list,  # data_paths
                256,  # fd_cache_size
                4,  # io_thread_num
                5,  # io_retry
                "add_noise",  # unused
                get_local_size(),  # GPU num in one worker
                get_local_rank(),  # GPU idx in one worker
                self.noise_chunk,  # chunk_size
            )
        elif data_type == 'rir':
            if self.rir_reader is not None:
                return
            rir_list = [
                '{}/{}{}'.format(self.rir_dir, self.rir_prefix, shard)
                for shard in range(self.rir_shards)
            ]
            self.rir_reader = FalconReader(
                rir_list,  # rir list
                256,  # fd_cache_size
                4,  # io_thread_num
                5,  # io_retry
                "add_rir",  # unused
                get_local_size(),  # GPU num in one worker
                get_local_rank(),  # GPU idx in one worker
                self.rir_chunk,  # chunk_size
            )

    def update_buf(self, data_type):
        '''update needed data buf'''
        if data_type == 'noise':
            if not self.noise_chunks:
                self.get_reader('noise')
                shard = random.randint(0, self.noise_shards - 1)
                entry_nums = self.noise_reader.get_entry_num([shard], False)
                chunks_idxs = [i * self.noise_chunk for i in range(entry_nums // self.noise_chunk)]
                self.noise_chunks.extend(
                    [
                        chunks_idxs[i : i + self.parrallel_chunk_num]
                        for i in range(0, len(chunks_idxs), self.parrallel_chunk_num)
                    ]
                )
            chunks = self.noise_chunks.popleft()
            datas = sum(self.noise_reader.read_many(chunks), [])
            self.noise_datas.extend(datas)
        elif data_type == 'rir':
            if not self.rir_chunks:
                self.get_reader('rir')
                shard = random.randint(0, self.rir_shards - 1)
                entry_nums = self.rir_reader.get_entry_num([shard], False)
                chunks_idxs = [i * self.rir_chunk for i in range(entry_nums // self.rir_chunk)]
                self.rir_chunks.extend(
                    [
                        chunks_idxs[i : i + self.parrallel_chunk_num]
                        for i in range(0, len(chunks_idxs), self.parrallel_chunk_num)
                    ]
                )
            chunks = self.rir_chunks.popleft()
            datas = sum(self.rir_reader.read_many(chunks), [])
            self.rir_datas.extend(datas)

    def get_data(self, data_type):
        '''read from noise shard'''
        if data_type == 'noise':
            if not self.noise_datas:
                self.update_buf('noise')
            data = self.noise_datas.popleft()
            return data
        if data_type == 'rir':
            if not self.rir_datas:
                self.update_buf('rir')
            data = self.rir_datas.popleft()
            return data
        return None

    def get_noise(self):
        '''get noise data.'''
        while True:
            data = self.get_data('noise')
            noise_data = np.frombuffer(data, dtype='int16')
            noise_data = torch.from_numpy(noise_data.copy()).float()
            yield noise_data

    def get_rir(self):
        '''get noise data.'''
        while True:
            data = self.get_data('rir')
            rir = np.frombuffer(data, dtype='float32')
            rir = torch.from_numpy(rir.copy()).view(-1, 4, self.rir_length + 2)
            rir_direction = rir[:, 0, :2]
            rir = rir[:, :, 2:]
            flags = [
                abs(rir_direction[i][0] - rir_direction[i][1]) > self.dir_diff_threshold
                for i in range(rir_direction.shape[0])
            ]
            rir = rir[flags]
            rir_direction = rir_direction[flags]
            yield rir, rir_direction

    def cut_into_fixed_length(self, data, length, batch_size):
        """crop fixed length noise slice"""
        batch = []
        for _ in range(batch_size):
            self.reused += 1
            start_pos = random.randint(0, data.size(0) - length)
            batch.append(data[start_pos : start_pos + length])
        return torch.stack(batch)

    def pad_wav(self, batch_data, max_length):
        """padd to same length"""
        res = np.zeros((len(batch_data), max_length), dtype=batch_data[0][self.key].dtype)
        for index, item in enumerate(batch_data):
            res[index][: item[self.key].shape[1]] = item[self.key][0]
        return res

    def __call__(self, item, **_kwargs):
        """do rir"""
        # TODO: This is only for 2-mic, we will support 1-mic in next pr
        self.max_length = max(self.max_length, item[self.key].shape[1])
        self.batch.append(item)
        if len(self.batch) < self.batch_size:
            if self.process_batch:
                return self.process_batch.pop()
            return None
        wav_batch = self.pad_wav(self.batch, self.max_length)

        if self.reused > self.reuse_num:
            if self.noise_reader is None:
                self.noise_reader = self.get_noise()
                self.rir_reader = self.get_rir()
                self.data_sim = RIRSimulator(
                    speech_rir_ratio=self.speech_rir_ratio,
                    noise_ratio=self.noise_ratio,
                    noise_rir_ratio=self.noise_rir_ratio,
                    normalize=self.normalize,
                    disturb_ratio=self.disturb_ratio,
                    disturb_rir_ratio=self.disturb_rir_ratio,
                    sir_max=self.sir_max,
                    sir_min=self.sir_min,
                    snr_max=self.snr_max,
                    snr_min=self.snr_min,
                    agc_max=self.agc_max,
                    agc_min=self.agc_min,
                    use_agc=self.use_agc,
                )
                self.data_sim.eval()
                self.noise_data = next(self.noise_reader)
                self.rir, self.rir_direction = next(self.rir_reader)

            self.reused = 0

        batch_size, ori_len = wav_batch.shape
        noise_data = self.cut_into_fixed_length(self.noise_data, ori_len, batch_size)
        rir_idx = np.random.randint(0, len(self.rir), batch_size)
        rir_data = self.rir[rir_idx]
        dir_data = self.rir_direction[rir_idx]
        # TODO only support 1 mic and 2 mics
        # item[self.key] is still a numpy array
        if self.mic_num == 1:
            wav_data, target_data = self.data_sim(
                torch.from_numpy(wav_batch).to(self.device),
                None,
                noise_data.to(self.device),
                rir_data.to(self.device),
            )
        else:
            wav_data, target_data = self.data_sim(
                torch.from_numpy(wav_batch).to(self.device),
                noise_data.to(self.device),
                None,
                rir_data.to(self.device),
            )

        _, t, _ = wav_data.shape
        new_data = wav_data.permute(0, 2, 1).reshape(-1, t)
        for index in range(0, batch_size):
            # for now we return numpy
            # pylint: disable=line-too-long
            item = self.batch[index]
            item["wav_simu"] = (
                new_data[index * self.mic_num : (index + 1) * self.mic_num]
                .cpu()
                .numpy()
                .astype(np.float32)[:, : item[self.key].shape[1]]
            )
            # pylint: disable=line-too-long
            item["wav_simu_target"] = (
                target_data[index : index + 1]
                .cpu()
                .numpy()
                .astype(np.float32)[:, : item[self.key].shape[1]]
            )
            # pylint: disable=line-too-long
            item['direction'] = dir_data[index : index + 1].cpu().numpy().astype(np.float32)
        del wav_data, target_data
        if self.device == 'cuda':
            # clear cache and cufft plane cache
            torch.cuda.empty_cache()
            if _kwargs.get("local_rank", None) is not None:
                torch.backends.cuda.cufft_plan_cache[_kwargs['local_rank']].clear()
            else:
                torch.backends.cuda.cufft_plan_cache.clear()
        self.process_batch.extend(self.batch)
        self.batch = []
        self.max_length = 0
        if self.process_batch:
            return self.process_batch.pop()
        return None


@PREPROCESS.register_module()
class GetRirData:
    '''just get rir data, not do RIR.'''

    # pylint: disable=line-too-long
    def __init__(
        self,
        wav_key="waveform",
        noise_key='noise',
        rir_key='rir',
        rir_dir_key='direction',
        dir_diff_threshold=20,
        rir_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/yxl.19/dataset/rir/rir100w",
        rir_shards=6,
        rir_prefix="sub",
        rir_chunk=1,
        noise_dir="hdfs://haruna/home/byte_arnold_hl_speech_asr/user/yxl.19/dataset/noise/dy500h",
        noise_shards=3,
        noise_prefix="sub",
        noise_rate=16000,
        noise_chunk=1,
        rir_length=2048,
        reuse_num=120,
        direction_pertubation=False,
        parrallel_chunk_num=2,
    ):
        '''init.'''
        self.wav_key = wav_key
        self.noise_key = noise_key
        self.rir_key = rir_key
        self.rir_dir_key = rir_dir_key

        # for rir data
        self.dir_diff_threshold = dir_diff_threshold
        self.rir_dir = rir_dir
        self.rir_shards = rir_shards
        self.rir_prefix = rir_prefix
        self.rir_length = rir_length
        self.rir_reader = None
        self.rir_data = None
        self.rir_direction = None
        self.rir_reused = 0
        self.direction_pertubation = direction_pertubation
        self.rir_chunk = rir_chunk
        self.rir_chunks = deque()  # chunks buf
        self.rir_datas = deque()  # data buf

        # for noise
        self.noise_dir = noise_dir
        self.noise_shards = noise_shards
        self.noise_prefix = noise_prefix
        self.noise_rate = noise_rate
        self.noise_chunk = noise_chunk
        self.noise_reader = None
        self.noise_data = None
        self.noise_reused = 0
        self.reuse_num = reuse_num
        self.noise_chunks = deque()  # chunks buf
        self.noise_datas = deque()  # data buf

        self.parrallel_chunk_num = parrallel_chunk_num

    def get_reader(self, data_type):
        """get data_type reader"""
        if data_type == 'noise':
            if self.noise_reader is not None:
                return
            noise_list = [
                '{}/{}{}'.format(self.noise_dir, self.noise_prefix, shard)
                for shard in range(self.noise_shards)
            ]
            self.noise_reader = FalconReader(
                noise_list,
                256,  # fd_cache_size
                4,  # io_thread_num
                5,  # io_retry
                "add_noise",  # unused
                get_local_size(),  # GPU num in one worker
                get_local_rank(),  # GPU idx in one worker
                self.noise_chunk,  # chunk_size
            )
        elif data_type == 'rir':
            if self.rir_reader is not None:
                return
            rir_list = [
                '{}/{}{}'.format(self.rir_dir, self.rir_prefix, shard)
                for shard in range(self.rir_shards)
            ]
            self.rir_reader = FalconReader(
                rir_list,  # rir list
                256,  # fd_cache_size
                4,  # io_thread_num
                5,  # io_retry
                "add_rir",  # unused
                get_local_size(),  # GPU num in one worker
                get_local_rank(),  # GPU idx in one worker
                self.rir_chunk,  # chunk_size
            )

    def update_buf(self, data_type):
        '''update needed data buf'''
        if data_type == 'noise':
            if not self.noise_chunks:
                self.get_reader('noise')
                shard = random.randint(0, self.noise_shards - 1)
                entry_nums = self.noise_reader.get_entry_num([shard], False)
                chunks_idxs = [i * self.noise_chunk for i in range(entry_nums // self.noise_chunk)]
                self.noise_chunks.extend(
                    [
                        chunks_idxs[i : i + self.parrallel_chunk_num]
                        for i in range(0, len(chunks_idxs), self.parrallel_chunk_num)
                    ]
                )
            chunks = self.noise_chunks.popleft()
            datas = sum(self.noise_reader.read_many(chunks), [])
            self.noise_datas.extend(datas)
        elif data_type == 'rir':
            if not self.rir_chunks:
                self.get_reader('rir')
                shard = random.randint(0, self.rir_shards - 1)
                entry_nums = self.rir_reader.get_entry_num([shard], False)
                chunks_idxs = [i * self.rir_chunk for i in range(entry_nums // self.rir_chunk)]
                self.rir_chunks.extend(
                    [
                        chunks_idxs[i : i + self.parrallel_chunk_num]
                        for i in range(0, len(chunks_idxs), self.parrallel_chunk_num)
                    ]
                )
            chunks = self.rir_chunks.popleft()
            datas = sum(self.rir_reader.read_many(chunks), [])
            self.rir_datas.extend(datas)

    def get_data(self, data_type):
        '''read from noise shard'''
        if data_type == 'noise':
            if not self.noise_datas:
                self.update_buf('noise')
            data = self.noise_datas.popleft()
            noise_data = np.frombuffer(data, dtype='int16')
            return noise_data
        if data_type == 'rir':
            if not self.rir_datas:
                self.update_buf('rir')
            data = self.rir_datas.popleft()
            rir = np.frombuffer(data, dtype='float32').reshape(-1, 4, self.rir_length + 2)
            rir_direction = rir[:, 0, :2]
            rir = rir[:, :, 2:]
            flags = [
                abs(rir_direction[i][0] - rir_direction[i][1]) > self.dir_diff_threshold
                for i in range(rir_direction.shape[0])
            ]
            rir = rir[flags]
            rir_direction = rir_direction[flags]
            return rir, rir_direction
        return None

    def get_noise_data(self, org_len):
        '''get noise data.'''
        if self.noise_data is None:
            self.noise_data = self.get_data('noise')
            self.noise_reused = 0

        self.noise_reused += 1
        start_pos = random.randint(0, self.noise_data.shape[0] - org_len)
        data = self.noise_data[start_pos : start_pos + org_len]
        if self.noise_reused >= self.reuse_num:
            self.noise_data = None
        return data

    def get_rir_data(self):
        '''get noise data.'''
        while self.rir_data is None or len(self.rir_data) == 0:
            self.rir_data, self.rir_direction = self.get_data('rir')
            self.rir_reused = 0

        rir_idx = random.randint(0, len(self.rir_data) - 1)
        rir_data = self.rir_data[rir_idx]
        rir_dir = self.rir_direction[rir_idx]
        # rir direction shift augmentation
        if self.direction_pertubation:
            rand_distribution = np.clip(np.random.randn() * 0.5, -1.0, 1.0)
            rir_dir = rir_dir + rand_distribution * 10.0
            rir_dir = np.clip(rir_dir, a_min=0.0, a_max=180.0)
        self.rir_reused += 1
        if self.rir_reused >= self.reuse_num:
            self.rir_data, self.rir_direction = None, None
        return rir_data, rir_dir

    def __call__(self, item, **_kwargs):
        '''get the data.'''
        if item is None or (self.wav_key not in item):
            return item
        waveform = item[self.wav_key]
        _, org_len = waveform.shape

        noise_data = self.get_noise_data(org_len)
        rir_data, rir_dir = self.get_rir_data()

        item[self.noise_key] = noise_data
        item[self.rir_key] = rir_data
        item[self.rir_dir_key] = rir_dir
        return item


@PREPROCESS.register_module()
class RirDataCollate:
    '''collate data for rir.'''

    def __init__(
        self,
        wav_key="waveform",
        noise_key='noise',
        rir_key='rir',
        rir_dir_key='direction',
        rir_length=2048,
    ):
        '''init.'''
        self.wav_key = wav_key
        self.noise_key = noise_key
        self.rir_key = rir_key
        self.rir_length = rir_length
        self.rir_dir_key = rir_dir_key

    def __call__(self, bucket_list, batch_out):
        '''do collate for rir noise and data.'''
        if sum(self.rir_key in item for item in bucket_list) != len(bucket_list):
            return

        batch_waveform = batch_out[self.wav_key]
        bsz, max_samples = batch_waveform.shape
        pad_samples = batch_out.get('pad_samples', [0] * bsz)

        batch_noise = torch.zeros([bsz, max_samples])
        batch_rir_data = torch.zeros([bsz, 4, self.rir_length])
        batch_rir_dir = torch.zeros([bsz, 2])

        for i, item in enumerate(bucket_list):
            noise = item[self.noise_key]
            rir_data = item[self.rir_key]
            rir_dir = item[self.rir_dir_key]
            pad_sample = pad_samples[i]

            noise_len = noise.shape[0]
            batch_noise[i, pad_sample : pad_sample + noise_len] = torch.from_numpy(noise)
            batch_rir_data[i] = torch.from_numpy(rir_data)
            batch_rir_dir[i] = torch.from_numpy(rir_dir)

        batch_out[self.noise_key] = batch_noise
        batch_out[self.rir_key] = batch_rir_data
        batch_out[self.rir_dir_key] = batch_rir_dir


@PREPROCESS.register_module()
class GPURir:
    '''RIR for GPU.'''

    def __init__(
        self,
        wav_key="waveform",
        noise_key='noise',
        rir_key='rir',
        target_wav_key="target_waveform",
        rir_dir_key="direction",
        snr_max=20,
        snr_min=5,
        sir_max=20,
        sir_min=5,
        agc_max=25000,
        agc_min=5000,
        use_agc=True,
        simu_ratio=1.0,
        speech_rir_ratio=1.0,
        disturb_ratio=1.0,
        disturb_rir_ratio=1.0,
        noise_ratio=1.0,
        noise_rir_ratio=1.0,
        mic_num=2,
        normalize=False,
        target_dereverb=False,
        rir_cut_length=800,
        rir_decay_rate=0.995,
        rir_direct_duration=480,
    ):
        '''init.'''
        self.wav_key = wav_key
        self.noise_key = noise_key
        self.rir_key = rir_key
        self.target_wav_key = target_wav_key
        self.rir_dir_key = rir_dir_key
        self.mic_num = mic_num
        self.simu_ratio = simu_ratio
        self.data_sim = RIRSimulator(
            speech_rir_ratio=speech_rir_ratio,
            noise_ratio=noise_ratio,
            noise_rir_ratio=noise_rir_ratio,
            normalize=normalize,
            disturb_ratio=disturb_ratio,
            disturb_rir_ratio=disturb_rir_ratio,
            sir_max=sir_max,
            sir_min=sir_min,
            snr_max=snr_max,
            snr_min=snr_min,
            agc_max=agc_max,
            agc_min=agc_min,
            use_agc=use_agc,
            target_dereverb=target_dereverb,
            rir_cut_length=rir_cut_length,
            rir_decay_rate=rir_decay_rate,
            rir_direct_duration=rir_direct_duration,
        )

    def __call__(self, batch_data, **_kwargs):
        """do rir"""
        if random.uniform(0, 1) > self.simu_ratio:
            return batch_data

        wav_batch = batch_data[self.wav_key]
        noise_data = batch_data[self.noise_key]
        rir_data = batch_data[self.rir_key]

        self.data_sim.eval()
        if self.mic_num == 1:
            wav_data, _ = self.data_sim(wav_batch, None, noise_data, rir_data)
        else:
            wav_data, target_data = self.data_sim(wav_batch, noise_data, None, rir_data)
            _, t, _ = wav_data.shape
            wav_data = wav_data.permute(0, 2, 1).reshape(-1, t)
            batch_data[self.target_wav_key] = target_data
        new_data = wav_data.short()

        batch_data[self.wav_key] = new_data

        return batch_data


@PREPROCESS.register_module()
class KaldiAddRir(AddRIR):
    """kaldi version add rir to wav"""

    def __init__(
        self,
        rir_dir='hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/litianyu.y/data/sid/shuffled_simulated_rirs/',
        rir_shards=16,
        rir_prefix='reverb',
        rir_chunk=30,
        parrallel_chunk_num=20,
        aug_key='augmentation_type',
        rir_type='reverb',
        key='waveform',
        rir_key=None,
    ):
        super().__init__(
            self,
            rir_dir=rir_dir,
            rir_shards=rir_shards,
            rir_prefix=rir_prefix,
            rir_chunk=rir_chunk,
            parrallel_chunk_num=parrallel_chunk_num,
        )

        self.aug_key = aug_key
        self.key = key
        self.rir_key = rir_key if rir_key is not None else key
        self.rir_type = rir_type
        self.wav_parser = WavParser(min_len=0, max_len=float('inf'))

    def get_rir(self):
        rir_data = None
        while True:
            rir_data = self.get_data('rir')
            rir_data = pickle.loads(rir_data)
            rir_data = self.wav_parser(rir_data)
            if rir_data is None or self.rir_key not in rir_data:
                continue
            break
        return rir_data[self.rir_key]

    def __call__(self, item, **_kwargs):
        if self.aug_key not in item:
            return item
        if not item[self.aug_key] == self.rir_type:
            return item
        rir_data = self.get_rir()
        item[self.key] = kaldi_add_rir(item[self.key], rir_data)
        return item
