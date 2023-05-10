''' Feature extractor of se. '''

import torch

try:
    import torch.fft
except:
    pass
from torch import nn
import torch.nn.functional as F
import numpy as np
from packaging import version
import random
from core.utils.se.fixbeam import fixbeam
from torch_stft import STFT

FOLAT_TO_INT_SCALE = 32768.0

FB_COEFF_HIGH_DELAY = [
    4.082482904638630e-001 * 0.7071067812,
    -3.004261596374099e-017,
    -5.282181998520569e-001,
    8.225260197957544e-018,
    2.399888152683055e-001,
    -5.733259464062257e-017,
    -1.209426510735405e-002,
    6.365366430137442e-018,
    1.846294773620827e-005,
    -4.145515408357529e-018,
    6.120179006378732e-004,
    -1.425483598645417e-017,
    5.499197872127104e-004,
    3.338200835541673e-017,
    3.330512243432287e-004,
    -3.991445470685697e-018,
    1.174539197140093e-004,
    8.134817722064635e-018,
    1.271175953190277e-003,
    1.061081289889354e-016,
    5.904772354521236e-004,
    4.261935782680764e-018,
    2.547449951754077e-004,
    -1.989653107221370e-018,
    1.439335536944256e-004,
    1.032861020696383e-017,
    8.379821248584243e-005,
    2.637946926618362e-018,
    6.459175728329905e-005,
    -4.337403675070662e-018,
    8.917856010668035e-004,
    5.921434018767376e-018,
]

FB_COEFF_LOW_DELAY = [
    4.082482904638630e-001 * 0.7071067812,
    4.968401743061806e-001,
    2.066691714121471e-001,
    -2.238048346354209e-001,
    -3.262082283276860e-001,
    -1.420211070438243e-001,
    -3.909320533828466e-002,
    -1.924553538772505e-002,
    -1.436529416070727e-002,
    -1.113334827822719e-002,
    -6.884275038178902e-003,
    -5.017088528479356e-003,
    -3.134903302460024e-003,
    -1.835919935877842e-003,
    -3.491272159574006e-003,
    -3.198290675026846e-003,
    -4.600789153950826e-003,
    -3.942995493097604e-003,
    -3.806208429521692e-003,
    -4.414552153450705e-003,
    -3.855268368743612e-003,
    -2.544286412324216e-003,
    -1.607660917815359e-003,
    -3.162033065507703e-004,
]


def inverse_dct(data_in, out_size):
    '''inverse dct transform.'''
    data_out = np.zeros(out_size)
    scale = np.sqrt(2.0 / out_size)
    for i in range(out_size):
        tmp = 0.0
        for j in range(data_in.shape[0]):
            tmp += data_in[j] * np.cos(np.pi * j * (2 * i + 1) / 2.0 / out_size)
        data_out[i] = tmp * scale
    return data_out


def get_fb_win(frame_len, delay_version='high'):
    '''generate the filter band window.'''
    subband_num = frame_len * 2
    win_size = subband_num * 3
    if delay_version == 'high':
        coeff = np.array(FB_COEFF_HIGH_DELAY, dtype=np.float32)
    elif delay_version == 'low':
        coeff = np.array(FB_COEFF_LOW_DELAY, dtype=np.float32)
    else:
        coeff = np.array(FB_COEFF_HIGH_DELAY, dtype=np.float32)
    scale_factor = np.sqrt(subband_num)
    subband_win = inverse_dct(coeff, win_size)
    subband_win *= scale_factor
    return subband_win[::-1].copy()


def subband_analyze(audio, win, frame_size, over_sample_ratio):
    '''subband analysis operation.
    audio: [Batch, max_length]
    win: [filter_size, ]
    '''
    with torch.no_grad():
        num_bands = frame_size * over_sample_ratio
        filter_size = win.size(0)
        num_blocks = int(filter_size / num_bands)
        audio_paded = (
            F.pad(
                audio,
                [filter_size - frame_size, filter_size - frame_size],
                mode='constant',
                value=0,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        framed_audio = F.unfold(audio_paded, kernel_size=(1, filter_size), stride=(1, frame_size))
        framed_audio = torch.transpose(framed_audio, 1, 2)
        filterd_frames = framed_audio * win
        x3 = filterd_frames[:, :, 0:num_bands]
        for i in range(1, num_blocks):
            x3 = x3 + filterd_frames[:, :, num_bands * i : num_bands * (i + 1)]
        if version.parse(torch.__version__) >= version.parse('1.7.0'):
            output = torch.fft.rfft(x3)
            output = torch.stack((output.real, output.imag), dim=-1)
        else:
            output = torch.rfft(x3, signal_ndim=1, normalized=False, onesided=True)
        return output


def subband_analyze_withgrad(audio, win, frame_size, over_sample_ratio):
    '''subband analysis operation.
    audio: [Batch, max_length]
    win: [filter_size, ]
    '''
    num_bands = frame_size * over_sample_ratio
    filter_size = win.size(0)
    num_blocks = int(filter_size / num_bands)
    audio_paded = (
        F.pad(
            audio,
            [filter_size - frame_size, filter_size - frame_size],
            mode='constant',
            value=0,
        )
        .unsqueeze(1)
        .unsqueeze(1)
    )
    framed_audio = F.unfold(audio_paded, kernel_size=(1, filter_size), stride=(1, frame_size))
    framed_audio = torch.transpose(framed_audio, 1, 2)
    filterd_frames = framed_audio * win
    x3 = filterd_frames[:, :, 0:num_bands]
    for i in range(1, num_blocks):
        x3 = x3 + filterd_frames[:, :, num_bands * i : num_bands * (i + 1)]
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        output = torch.fft.rfft(x3)
        output = torch.stack((output.real, output.imag), dim=-1)
    else:
        output = torch.rfft(x3, signal_ndim=1, normalized=False, onesided=True)
    return output


def subband_compose(subband_feat, win, frame_size, over_sample_ratio):
    '''subband compose operation.
    subband_feat: [Batch, num_frames, frame_size + 1, 2]
    win: [filter_size, ]
    '''
    num_bands = frame_size * over_sample_ratio
    filter_size = win.size(0)
    num_blocks = int(filter_size / num_bands)
    num_frames = subband_feat.size(1)
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        x2 = torch.fft.irfftn(
            torch.complex(subband_feat[..., 0], subband_feat[..., 1]), dim=-1, s=[num_bands]
        )
    else:
        x2 = torch.irfft(
            subband_feat, signal_ndim=1, normalized=False, onesided=True, signal_sizes=[num_bands]
        )
    x2 = x2.repeat([1, 1, num_blocks])
    win = win.view(1, 1, filter_size)
    x2 = x2 * win
    x2 = torch.transpose(x2, 1, 2)
    res = F.fold(
        x2,
        output_size=[1, frame_size * (num_frames + num_blocks * over_sample_ratio - 1)],
        kernel_size=(1, filter_size),
        stride=(1, frame_size),
    )
    return res.squeeze()


def energy_phase(in_subband):
    subband_real = in_subband[..., 0]
    subband_imag = in_subband[..., 1]
    subband_energy = subband_real**2 + subband_imag**2
    subband_mag = torch.sqrt(subband_real**2 + subband_imag**2)
    subband_phase = in_subband / (subband_mag.unsqueeze(-1) + 1e-12)
    return subband_energy, subband_phase


class BaseSEFeatExtractor(nn.Module):
    '''base se feature extractor.'''

    def __init__(self, args):
        super().__init__()
        self.win_filter = torch.from_numpy(
            get_fb_win(args.frame_length, delay_version=args.delay_version)
        ).float()
        self.win_filter.detach_()
        self.frame_length = args.frame_length
        self.oversample_ratio = args.oversample_ratio
        self.sample_rate = args.sampling_rate
        if args.delay_version == 'low':
            self.fb_delay = 1
        else:
            self.fb_delay = 5

    def forward(self, batch):
        '''extract feats'''
        return batch

    def subband_analyze_transform(self, input_):
        '''Subband analyze attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_analyze(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_

    def subband_compose_transform(self, input_):
        '''Subband compose attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_compose(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_


class AecSubbandFeatExtractor(nn.Module):
    '''aec feature extractor.'''

    def __init__(self, args):
        super().__init__()
        self.win_filter = torch.from_numpy(
            get_fb_win(args.frame_length, delay_version=args.delay_version)
        ).float()
        self.win_filter.detach_()
        self.frame_length = args.frame_length
        self.oversample_ratio = args.oversample_ratio
        self.sample_rate = args.sampling_rate
        if args.delay_version == 'low':
            self.fb_delay = 1
        else:
            self.fb_delay = 5

    def forward(self, batch):
        '''aec feature extractor forward, input: batch data output: feats struct'''
        with torch.no_grad():
            if batch.__contains__('speech'):
                nearend_subband = self.subband_analyze_transform(batch["speech"])
                speech_stft = nearend_subband[:, : -self.fb_delay, :, :] / FOLAT_TO_INT_SCALE
            else:
                clean_r = clean_i = 0
            aec_subband = self.subband_analyze_transform(batch["aec"])
            fir_stft = aec_subband[:, : -self.fb_delay, :, :] / FOLAT_TO_INT_SCALE

            ref_subband = self.subband_analyze_transform(batch["ref_tde"])
            ref_stft = ref_subband[:, : -self.fb_delay, :, :] / FOLAT_TO_INT_SCALE

            mic_subband = self.subband_analyze_transform(batch["mic"])
            mic_stft = mic_subband[:, : -self.fb_delay, :, :] / FOLAT_TO_INT_SCALE

            if batch.__contains__('speech'):
                out = {
                    'fir_stft': fir_stft,
                    'speech_stft': speech_stft,
                    'mic_stft': mic_stft,
                    'ref_stft': ref_stft,
                    'fb_delay': self.fb_delay,
                    'clean': batch["speech"],
                    'aec': batch["aec"],
                    'ref': batch["ref_tde"],
                }
            else:
                out = {
                    'fir_stft': fir_stft,
                    'ref_stft': ref_stft,
                    'mic_stft': mic_stft,
                    'speech_stft': None,
                }
            return out

    def subband_analyze_transform(self, input_):
        '''Subband analyze attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_analyze(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_

    def subband_analyze_transform_withgrad(self, input_):
        '''Subband analyze attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_analyze_withgrad(
            input_, win_filter, self.frame_length, self.oversample_ratio
        )
        return output_

    def subband_compose_transform(self, input_):
        '''Subband compose attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_compose(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_


class SSLSubbandFeatExtractor_linear(BaseSEFeatExtractor):
    '''ssl feature extractor.'''

    def __init__(self, args):
        super().__init__(args)
        self.mic_num = args.mic_num
        self.disable_freq = args.disable_freq
        self.block_frame = args.block_frame
        self.uniform_type = args.uniform_type
        self.zone_num = args.zone_num
        self.max_source_num = args.max_source_num
        self.gauss_smooth = args.gauss_smooth

    def forward(self, batch):
        with torch.no_grad():
            device = batch['mc_waveform'].device
            noisy_ori = batch['mc_waveform'].clone()
            noisy = batch['mc_waveform']
            n_batch = noisy.shape[0]
            assert noisy.shape == noisy_ori.shape
            assert len(noisy.shape) == 3
            assert noisy.shape[1] == self.mic_num

            noisy0_subband = self.subband_analyze_transform(noisy_ori[:, 0, :])
            block_num = noisy0_subband.shape[1] // self.block_frame
            freq_num = noisy0_subband.shape[2]
            # net input
            block_input = torch.zeros(
                (n_batch, block_num, self.mic_num * 2, self.block_frame, freq_num)
            ).to(device)
            for i in range(self.mic_num):
                spectra = self.subband_analyze_transform(noisy[:, i, :])
                block_input[:, :, i * 2 : i * 2 + 2, :, :] = (
                    spectra[:, : block_num * self.block_frame, :, :]
                    .reshape(n_batch, block_num, self.block_frame, freq_num, 2)
                    .permute(0, 1, 4, 2, 3)
                )
            if self.disable_freq["enable"]:
                if self.disable_freq["freq_bin_range"] is not None:
                    assert (
                        self.disable_freq["freq_bin_range"][0]
                        < self.disable_freq["freq_bin_range"][1]
                    ), "freq_bin_range has problem."
                    assert (
                        self.disable_freq["freq_bin_range"][0] >= 0
                        and self.disable_freq["freq_bin_range"][1] <= freq_num
                    ), "freq_bin_range has problem."
                    block_input[
                        ...,
                        self.disable_freq["freq_bin_range"][0] : self.disable_freq[
                            "freq_bin_range"
                        ][1],
                    ] *= 0.01
                else:
                    assert (
                        self.disable_freq["freq_range"] is not None
                    ), "freq_range shouldn't be none"
                    assert self.disable_freq["freq_range"][0] >= 0 and self.disable_freq[
                        "freq_range"
                    ][1] <= (self.sample_rate / 2), "freq_range has problem."
                    freq_bin_min = int(
                        self.disable_freq["freq_range"][0] / (self.sample_rate / 2) * freq_num
                    )
                    freq_bin_max = int(
                        self.disable_freq["freq_range"][1] / (self.sample_rate / 2) * freq_num
                    )
                    assert freq_bin_min >= 0 and freq_bin_max <= freq_num, "freq range has problem"
                    block_input[..., freq_bin_min:freq_bin_max] *= 0.01

            if not batch.__contains__('directional_waveform'):
                out = {'block_input': block_input, 'noisy0_subband': noisy0_subband}
                return out
            noisy_energy, _ = energy_phase(noisy0_subband)
            noisy_energy_block = (
                noisy_energy[:, : block_num * self.block_frame, :]
                .reshape(n_batch, block_num, self.block_frame, freq_num)
                .mean(dim=(2, 3))
            )
            weight_block = noisy_energy_block / noisy_energy_block.mean(dim=-1, keepdim=True)
            weight_block = torch.gt(weight_block, 1e-2).cuda().float()
            sou_num = batch["source_num"]
            target_angle = batch['direction']
            location_data = batch['directional_waveform']
            assert batch['direction'].shape[1] == self.max_source_num
            assert batch['directional_waveform'].shape[1] == self.max_source_num

            if self.uniform_type == "rad":
                zone_idx = torch.clamp(
                    (target_angle // (180.0 / self.zone_num)), 0, self.zone_num - 1
                ).int()
            elif self.uniform_type == "cos":
                if self.zone_num % 2 == 0:
                    zone_idx = torch.clamp(
                        self.zone_num // 2
                        - torch.ceil(
                            torch.cos(target_angle / 180.0 * np.pi) / (2.0 / self.zone_num)
                        ),
                        0.0,
                        self.zone_num - 1,
                    ).int()
                else:
                    zone_idx = torch.clamp(
                        self.zone_num // 2
                        - torch.round(
                            torch.cos(target_angle / 180.0 * np.pi) / (2.0 / (self.zone_num - 1))
                        ),
                        0.0,
                        self.zone_num - 1,
                    ).int()
            target_vec_block = torch.zeros(n_batch, self.max_source_num, block_num).to(device)
            for cnt in range(self.max_source_num):
                tmp_subband = self.subband_analyze_transform(location_data[:, cnt])
                tmp_energy, _ = energy_phase(tmp_subband)
                tmp_energy_block = (
                    tmp_energy[:, : block_num * self.block_frame]
                    .reshape(n_batch, block_num, self.block_frame, freq_num)
                    .mean(dim=(2, 3))
                )
                target_vec_block[:, cnt] = tmp_energy_block / (1e-6 + noisy_energy_block)
            loc_vec = torch.zeros(n_batch, block_num, self.zone_num).to(device)
            for i in range(n_batch):
                loc_vec[i, :, zone_idx[i, 0]] = target_vec_block[i, 0, :]
                if sou_num[i] > 1:
                    for j in range(int(sou_num[i].item()) - 1):
                        assert target_angle[i, j + 1] > -1
                        loc_vec[i, :, zone_idx[i, j + 1]] += target_vec_block[i, j + 1]
            loc_vec = torch.gt(loc_vec, 0.05).to(device).float()  # B T 40
            loc_vec = loc_vec * weight_block.unsqueeze(-1)
            for i in range(n_batch):
                assert torch.sum(loc_vec[i]) > 0.5

            if self.gauss_smooth["enable"]:
                gauss_ref = torch.exp(
                    -torch.arange(-self.zone_num + 1, self.zone_num).float() ** 2
                    / (self.gauss_smooth["gauss_const"]) ** 2
                ).to(device)
                loc_vec_reserve = loc_vec.clone()
                for i in range(self.zone_num):
                    nonzero_index = torch.nonzero(loc_vec_reserve[:, :, i], as_tuple=True)
                    tmp = gauss_ref[self.zone_num - 1 - i : self.zone_num - 1 - i + self.zone_num]
                    loc_vec[nonzero_index[0], nonzero_index[1], :] = torch.max(
                        tmp, loc_vec[nonzero_index[0], nonzero_index[1], :]
                    )
            out = {
                'block_input': block_input,
                'weight_block': weight_block,
                'noisy0_subband': noisy0_subband,
                'noisy_time': noisy,
                'loc_vec': loc_vec,
                'block_ratio': target_vec_block,
                "source_num": batch['source_num'],
                "diffuse_flag": batch['noise_flag'],
                "data_info": batch['direction'],
            }
            return out


class SSLSubbandFeatExtractor_cicular(BaseSEFeatExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.mic_num = args.mic_num
        self.disable_freq = args.disable_freq
        self.block_frame = args.block_frame
        self.uniform_type = args.uniform_type
        self.zone_num = args.zone_num
        self.max_source_num = args.max_source_num
        self.gauss_smooth = args.gauss_smooth

    def forward(self, batch):
        with torch.no_grad():
            device = batch['mc_waveform'].device
            noisy_ori = batch['mc_waveform'].clone()
            noisy = batch['mc_waveform']
            n_batch = noisy.shape[0]
            assert noisy.shape == noisy_ori.shape
            assert len(noisy.shape) == 3
            assert noisy.shape[1] == self.mic_num

            noisy0_subband = self.subband_analyze_transform(noisy_ori[:, 0, :])
            block_num = noisy0_subband.shape[1] // self.block_frame
            freq_num = noisy0_subband.shape[2]
            # net input
            block_input = torch.zeros(
                (n_batch, block_num, self.mic_num * 2, self.block_frame, freq_num)
            ).to(device)
            for i in range(self.mic_num):
                spectra = self.subband_analyze_transform(noisy[:, i, :])
                block_input[:, :, i * 2 : i * 2 + 2, :, :] = (
                    spectra[:, : block_num * self.block_frame, :, :]
                    .reshape(n_batch, block_num, self.block_frame, freq_num, 2)
                    .permute(0, 1, 4, 2, 3)
                )
            if self.disable_freq["enable"]:
                if self.disable_freq["freq_bin_range"] is not None:
                    assert (
                        self.disable_freq["freq_bin_range"][0]
                        < self.disable_freq["freq_bin_range"][1]
                    ), "freq_bin_range has problem."
                    assert (
                        self.disable_freq["freq_bin_range"][0] >= 0
                        and self.disable_freq["freq_bin_range"][1] <= freq_num
                    ), "freq_bin_range has problem."
                    block_input[
                        ...,
                        self.disable_freq["freq_bin_range"][0] : self.disable_freq[
                            "freq_bin_range"
                        ][1],
                    ] *= 0.01
                else:
                    assert (
                        self.disable_freq["freq_range"] is not None
                    ), "freq_range shouldn't be none"
                    assert self.disable_freq["freq_range"][0] >= 0 and self.disable_freq[
                        "freq_range"
                    ][1] <= (self.sample_rate / 2), "freq_range has problem."
                    freq_bin_min = int(
                        self.disable_freq["freq_range"][0] / (self.sample_rate / 2) * freq_num
                    )
                    freq_bin_max = int(
                        self.disable_freq["freq_range"][1] / (self.sample_rate / 2) * freq_num
                    )
                    assert freq_bin_min >= 0 and freq_bin_max <= freq_num, "freq range has problem"
                    block_input[..., freq_bin_min:freq_bin_max] *= 0.01
            if not batch.__contains__('directional_waveform'):
                out = {'block_input': block_input, 'noisy0_subband': noisy0_subband}
                return out

            noisy_energy, _ = energy_phase(noisy0_subband)
            noisy_energy_block = (
                noisy_energy[:, : block_num * self.block_frame, :]
                .reshape(n_batch, block_num, self.block_frame, freq_num)
                .mean(dim=(2, 3))
            )
            weight_block = noisy_energy_block / noisy_energy_block.mean(dim=-1, keepdim=True)
            weight_block = torch.gt(weight_block, 1e-2).cuda().float()
            sou_num = batch["source_num"]
            target_angle = batch['direction']
            location_data = batch['directional_waveform']
            assert batch['direction'].shape[1] == self.max_source_num
            assert batch['directional_waveform'].shape[1] == self.max_source_num

            if self.uniform_type == "rad":
                zone_idx = torch.clamp(
                    (target_angle // (360.0 / self.zone_num)), 0, self.zone_num - 1
                ).int()
            elif self.uniform_type == "cos":
                raise ValueError("Not supported")
            target_vec_block = torch.zeros(n_batch, self.max_source_num, block_num).to(device)
            for cnt in range(self.max_source_num):
                tmp_subband = self.subband_analyze_transform(location_data[:, cnt])
                tmp_energy, _ = energy_phase(tmp_subband)
                tmp_energy_block = (
                    tmp_energy[:, : block_num * self.block_frame]
                    .reshape(n_batch, block_num, self.block_frame, freq_num)
                    .mean(dim=(2, 3))
                )
                target_vec_block[:, cnt] = tmp_energy_block / (1e-6 + noisy_energy_block)
            loc_vec = torch.zeros(n_batch, block_num, self.zone_num).to(device)
            for i in range(n_batch):
                loc_vec[i, :, zone_idx[i, 0]] = target_vec_block[i, 0, :]
                if sou_num[i] > 1:
                    for j in range(int(sou_num[i].item()) - 1):
                        assert target_angle[i, j + 1] > -1
                        loc_vec[i, :, zone_idx[i, j + 1]] += target_vec_block[i, j + 1]
            loc_vec = torch.gt(loc_vec, 0.05).to(device).float()  # B T 40
            loc_vec = loc_vec * weight_block.unsqueeze(-1)
            for i in range(n_batch):
                assert torch.sum(loc_vec[i]) > 0.5

            if self.gauss_smooth["enable"]:
                assert self.zone_num % 2 == 0
                ref_idx = (
                    abs(torch.arange(0, self.zone_num) - self.zone_num / 2) - self.zone_num / 2
                )
                ref_idx = ref_idx.repeat(2)
                gauss_ref = torch.exp(
                    -ref_idx.float() ** 2 / (self.gauss_smooth["gauss_const"]) ** 2
                ).to(device)
                loc_vec_reserve = loc_vec.clone()
                for i in range(self.zone_num):
                    nonzero_index = torch.nonzero(loc_vec_reserve[:, :, i], as_tuple=True)
                    tmp = gauss_ref[self.zone_num - i : self.zone_num - i + self.zone_num]
                    loc_vec[nonzero_index[0], nonzero_index[1], :] = torch.max(
                        tmp, loc_vec[nonzero_index[0], nonzero_index[1], :]
                    )
            out = {
                'block_input': block_input,
                'weight_block': weight_block,
                'noisy0_subband': noisy0_subband,
                'noisy_time': noisy,
                'loc_vec': loc_vec,
                'block_ratio': target_vec_block,
                "source_num": batch['source_num'],
                "diffuse_flag": batch['noise_flag'],
                "data_info": batch['direction'],
            }
            return out


class BFSubbandFeatExtractor(BaseSEFeatExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.mic_num = args.mic_num
        self.array_type = args.array_type
        self.mic_space = args.mic_space
        self.mask_type = args.mask_type
        self.net_in = args.net_in

    def forward(self, batch):
        with torch.no_grad():
            device = batch['mc_waveform'].device
            noisy_ori = batch['mc_waveform'].clone()
            noisy = batch['mc_waveform']
            n_batch = noisy.shape[0]
            assert noisy.shape == noisy_ori.shape
            assert len(noisy.shape) == 3
            assert noisy.shape[1] == self.mic_num
            noisy_subband = []
            noisy_energy = []
            for i in range(self.mic_num):
                tmp = self.subband_analyze_transform(noisy[:, i, :])
                noisy_subband.append(tmp)
                noisy_energy.append(energy_phase(tmp)[0])
            noisy_subband = torch.stack(noisy_subband, dim=1)  # B, mic_num, T, F, 2
            noisy_energy = torch.stack(noisy_energy, dim=1)  # B , mic_num, T, F
            noisy0_time = self.subband_compose_transform(noisy_subband[:, 0, ...])

            """ ===============net input================ """
            target_direction = batch['clean_direction_use']  # B
            noisy_steer_mic, fixbeam_subband = fixbeam(
                noisy_subband, target_direction, self.array_type, self.mic_space
            )  # B T F 2
            noisy_energy_ch_mean = torch.mean(noisy_energy, dim=1)  # B T F
            noisy_energy_freq_mean = torch.mean(noisy_energy_ch_mean, dim=2, keepdim=True)  # B T 1
            noisy_energy_time_mean = torch.mean(
                noisy_energy_freq_mean, dim=1, keepdim=True
            )  # B 1 1
            weight = noisy_energy_freq_mean / (noisy_energy_time_mean + 1e-12)
            weight = torch.gt(weight, 1e-2).cuda().float()  # B T 1
            weight_time = weight.transpose(1, 2).repeat(
                [1, self.win_filter.shape[0], 1]
            )  # B filter_size T
            out_size = int((weight_time.shape[-1] - 1) * self.frame_length + weight_time.shape[1])
            weight_time = F.fold(
                weight_time,
                output_size=[1, out_size],
                kernel_size=(1, weight_time.shape[1]),
                stride=(1, self.frame_length),
            )

            if not batch.__contains__('target_sou_data'):
                out = {
                    'mask_weight': weight,
                    'noisy_subband': noisy_subband,
                    'fixbeam_subband': fixbeam_subband,
                    'noisy0_time': noisy0_time,
                }
                out['spectra_in'] = eval(self.net_in)
                if self.mask_type == "mask_clean_fixbeam":
                    out['base'] = fixbeam_subband
                return out

            """ real mask """
            clean = batch['target_sou_data']  # B, len
            clean_subband_ori = self.subband_analyze_transform(clean)
            clean_energy, _ = energy_phase(clean_subband_ori)
            fixbeam_energy, _ = energy_phase(fixbeam_subband)

            mask_clean_fixbeam = torch.sqrt(clean_energy / (fixbeam_energy + 1e-12))
            mask_clean_fixbeam = torch.clamp(mask_clean_fixbeam, 0.0, 1.0)
            clean_subband = fixbeam_subband * mask_clean_fixbeam.unsqueeze(-1)
            clean_time = self.subband_compose_transform(clean_subband)

            out = {
                'mask_weight': weight,
                'noisy_subband': noisy_subband,
                'fixbeam_subband': fixbeam_subband,
                'target_spectra': clean_subband_ori,
                'noisy0_time': noisy0_time,
                'target_time': clean_time,
            }
            out['spectra_in'] = eval(self.net_in)
            out['target_mask'] = eval(self.mask_type)
            if self.mask_type == "mask_clean_fixbeam":
                out['base'] = fixbeam_subband
            return out


class DenoiseSubbandFeatExtractor(nn.Module):
    '''denoise feature extractor.'''

    def __init__(self, args):
        super().__init__()
        self.win_filter = torch.from_numpy(
            get_fb_win(args.frame_length, delay_version=args.delay_version)
        ).float()
        self.win_filter.detach_()
        self.frame_length = args.frame_length
        self.oversample_ratio = args.oversample_ratio
        self.sample_rate = args.sampling_rate
        if args.delay_version == 'low':
            self.fb_delay = 1
        else:
            self.fb_delay = 5

    def forward(self, batch):
        '''denoise feature extractor forward, input: batch data output: feats struct'''
        with torch.no_grad():
            if "speech_waveform" in batch.keys():
                clean_wav = batch["speech_waveform"]
                batch_size, num_mic, length = clean_wav.shape
                clean_wav = clean_wav.reshape([-1, length])
                nearend_subband = self.subband_analyze_transform(clean_wav)
                nearend_subband = (
                    nearend_subband[:, :-5, :, :] / FOLAT_TO_INT_SCALE
                )  # int to float, convert to 0~1
                shape = nearend_subband.shape
                nearend_subband = nearend_subband.reshape(
                    [batch_size, num_mic, shape[1], shape[2], shape[3]]
                )
                clean_r = nearend_subband[:, :, :, :, 0]
                clean_i = nearend_subband[:, :, :, :, 1]

            noisy_wav = batch["noisy_waveform"]
            batch_size, num_mic, length = noisy_wav.shape
            noisy_wav = noisy_wav.reshape([-1, length])
            noisy_subband = self.subband_analyze_transform(noisy_wav)
            noisy_subband = noisy_subband[:, :-5, :, :] / FOLAT_TO_INT_SCALE
            shape = noisy_subband.shape
            noisy_subband = noisy_subband.reshape(
                [batch_size, num_mic, shape[1], shape[2], shape[3]]
            )
            noisy_r = noisy_subband[:, :, :, :, 0]
            noisy_i = noisy_subband[:, :, :, :, 1]

            out = {
                'noisy_r': noisy_r,
                'noisy_i': noisy_i,
                'fb_delay': self.fb_delay,
                'noisy': batch["noisy_waveform"],
            }
            if "speech_waveform" in batch.keys():
                out['clean_r'] = clean_r
                out['clean_i'] = clean_i
                out['clean'] = batch["speech_waveform"]
            return out

    def subband_analyze_transform(self, input_):
        '''Subband analyze attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_analyze(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_

    def subband_compose_transform(self, input_):
        '''Subband compose attached to feat extractor.'''
        win_filter = self.win_filter.to(input_.device)
        output_ = subband_compose(input_, win_filter, self.frame_length, self.oversample_ratio)
        return output_


class DenoiseStftFeatExtractor(nn.Module):
    '''denoise feature extractor using stft.'''

    def __init__(self, args):
        super().__init__()
        self.frame_length = args.frame_length
        self.frame_shift = args.frame_shift
        self.fft_size = args.fft_size
        self.win_type = args.win_type
        self.stft = STFT(
            filter_length=self.fft_size,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            window=self.win_type,
        ).cuda()

    def forward(self, batch):
        '''denoise feature extractor forward, input: batch data output: feats struct'''
        with torch.no_grad():
            if "speech_waveform" in batch.keys():
                clean_wav = batch["speech_waveform"]
                batch_size, num_mic, length = clean_wav.shape
                clean_wav = clean_wav.reshape([-1, length])
                clean_mag, clean_phase = self.stft_transform(clean_wav / FOLAT_TO_INT_SCALE)
                shape = clean_mag.shape
                clean_mag = clean_mag.reshape([batch_size, num_mic, shape[1], shape[2]])
                clean_phase = clean_phase.reshape([batch_size, num_mic, shape[1], shape[2]])
                clean_r = clean_mag * torch.cos(clean_phase)
                clean_i = clean_mag * torch.sin(clean_phase)

            noisy_wav = batch["noisy_waveform"]
            batch_size, num_mic, length = noisy_wav.shape
            noisy_wav = noisy_wav.reshape([-1, length])
            noisy_mag, noisy_phase = self.stft_transform(noisy_wav / FOLAT_TO_INT_SCALE)
            shape = noisy_mag.shape
            noisy_mag = noisy_mag.reshape([batch_size, num_mic, shape[1], shape[2]])
            noisy_phase = noisy_phase.reshape([batch_size, num_mic, shape[1], shape[2]])
            noisy_r = noisy_mag * torch.cos(noisy_phase)
            noisy_i = noisy_mag * torch.sin(noisy_phase)

            out = {
                'noisy_r': noisy_r.permute(0, 1, 3, 2).contiguous(),
                'noisy_i': noisy_i.permute(0, 1, 3, 2).contiguous(),
                'noisy': batch["noisy_waveform"],
            }
            if "speech_waveform" in batch.keys():
                out['clean_r'] = clean_r.permute(0, 1, 3, 2).contiguous()
                out['clean_i'] = clean_i.permute(0, 1, 3, 2).contiguous()
                out['clean'] = batch["speech_waveform"]
            return out

    def stft_transform(self, input_):
        '''stft analyze attached to feat extractor.'''
        x_spec = self.stft.transform(input_)
        return x_spec

    def inverse_stft(self, x_real, x_imag):
        '''inverse stft analyze attached to feat extractor.'''
        x_mag = torch.sqrt(x_real**2 + x_imag**2 + 1.0e-8).permute(0, 2, 1).contiguous()
        x_pahse = torch.atan2(x_imag + 1.0e-8, x_real + 1.0e-8).permute(0, 2, 1).contiguous()
        x_t = self.stft.inverse(x_mag, x_pahse)
        return x_t
