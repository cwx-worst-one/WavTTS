''' Solutions of SE tasks '''

import os
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.io import wavfile
import torch
from torch import nn


from core.solutions.base_solution import BaseSolution, register_solution
from core.models.se.feat_extractor import *
from core.models.se.model_aec import *
from core.models.se.model_ssl import *
from core.models.se.model_bf import *
from core.models.se.model_denoise import *
from core.criterions.se_criterion import *
from core.utils import hdfs_put, FalconDict
from core.solutions.inference import (
    BaseInfer,
    INFERS,
)


class AECModelExporter(nn.Module):
    '''export aec model to onnx'''

    def __init__(self, args):
        super().__init__()
        self.aec_net = eval(args.aec_net_type)(args)

    def forward(self, xx):
        '''
        Forward for BaseAECSolution module
        '''
        mask = self.aec_net(xx)
        return mask


@register_solution("BaseAECSolution")
class BaseAECSolution(BaseSolution):
    """
    Model AEC:
    """

    def __init__(self, args):
        """
        args:
        feat_extractor: feature extractor name of solution
        aec_net_type: model name of solution
        criterion_type: criterion name of solution
        """
        super().__init__()
        self.feat_extractor = eval(args.feat_extractor_type)(args.feat_extractor)
        self.aec_net = eval(args.aec_net_type)(args.aec_net)
        self.criterion_module = eval(args.criterion_type)(args.criterion)
        self.args = args
        self.float_to_int_scale = 32768

    def forward(self, batch_data):
        """
        Args:
            input:
                mic (Batch x Time): input mic signals
                ref (Batch x Time): input ref signals
                ref_tde (Batch x Time): input ref signals
                aec (Batch x Time): input aec residual signals
                speech (Batch x Time): target clean speech signals
            output:
                loss (Batch): loss of aec training
        """

        feat = self.feat_extractor(batch_data)
        output = self.aec_net(feat)

        enh = self.feat_extractor.subband_compose_transform(output["ehn_stft"])
        if enh.ndim == 1:
            enh = enh.unsqueeze(0)
        enh = (
            self.float_to_int_scale
            * enh[:, self.feat_extractor.frame_length * self.feat_extractor.fb_delay :]
        )

        # for stft consistency
        enh_stft = self.feat_extractor.subband_analyze_transform_withgrad(enh)
        enh_stft = enh_stft[:, : -self.feat_extractor.fb_delay, :, :] / self.float_to_int_scale
        output["ehn_stft"] = enh_stft

        loss = self.criterion_module(output)

        forward_out = OrderedDict()
        forward_out["backward_loss"] = loss
        # forward_out["enh"] = enh
        return forward_out

    def inference(self, batch_data):
        """
        Args:
            input:
                mic (Batch x Time): input mic signals
                ref (Batch x Time): input ref signals
                aec (Batch x Time): input aec residual signals
                speech (Batch x Time): target clean speech signals
            output:
                enh (Batch x Time): enh signals
        """
        feat = self.feat_extractor(batch_data)
        output = self.aec_net(feat)

        enh = self.feat_extractor.subband_compose_transform(output["ehn_stft"])
        if enh.ndim == 1:
            enh = enh.unsqueeze(0)
        enh = (
            self.float_to_int_scale
            * enh[:, self.feat_extractor.frame_length * self.feat_extractor.fb_delay :]
        )
        forward_out = OrderedDict()
        forward_out["enh_wav"] = enh
        forward_out['uttid'] = batch_data['uttid']
        if 'speech' in batch_data:
            forward_out['pesq'] = compute_pesq2(
                batch_data['speech'][0, :].cpu().numpy(), enh[0].cpu().numpy()
            )
        else:
            forward_out['reduce_db'] = echo_reduce_db(batch_data['mic'][0], enh[0])
        if self.args.inference.get("save_enh_wav", False):
            self.save(self.args, forward_out)
        return forward_out

    def save(self, args, data_dict):
        '''save enh_wav'''
        inference_cfg = args.inference
        wav = data_dict['enh_wav'].cpu().numpy()[0]  # choose first batch
        wav = wav[np.newaxis, :]  # add channel dim
        uttid = data_dict['uttid'][0]
        sample_rate = data_dict.get('sample_rate', 16000)
        save_name = os.path.join(inference_cfg.save_dir, f"{uttid}")
        wavfile.write(save_name, sample_rate, wav.transpose(1, 0).astype(np.int16))
        if inference_cfg.get('remote_save_dir', None):
            hdfs_put(save_name, inference_cfg.remote_save_dir)


@register_solution("BaseSSLSolution")
class BaseSSLSolution(BaseSolution):
    """
    Model SSL:
    """

    def __init__(self, args):
        """
        args:
        feat_extractor: feature extractor name of solution
        ssl_net_type: model name of solution
        criterion_type: criterion name of solution
        """
        super().__init__()
        self.feat_extractor = eval(args.feat_extractor_type)(args.feat_extractor)
        self.ssl_net = eval(args.ssl_net_type)(args.ssl_net)
        self.criterion_module = eval(args.criterion_type)(args.criterion)
        self.args = args

    def forward(self, batch_data):
        """
        batch_data: raw data for feature extraction
        """
        feat = self.feat_extractor(batch_data)
        est_loc_vec = self.ssl_net(feat)
        loss = self.criterion_module(est_loc_vec, feat["loc_vec"])
        forward_out = OrderedDict()
        forward_out["backward_loss"] = loss
        # data_out = dict(
        #     est_loc_vec=est_loc_vec,
        #     loc_vec=feat['loc_vec'],
        #     noisy0_subband=feat['noisy0_subband']
        # )
        return forward_out

    def angle_dis(self, a, b):
        """calculate angle distance between two np.ndarrays"""
        array_type = self.args.inference.array_type
        if array_type == "circular":
            tmp1 = (a - b) % 360
            tmp2 = (b - a) % 360
            dis = np.where(tmp1 < tmp2, tmp1, tmp2)
        elif array_type == 'linear':
            dis = np.abs(a - b)
        return dis

    def inference(self, batch_data):
        """
        batch_data: raw data for feature extraction
        """
        self.inference_type = self.args.inference.type
        self.infer_save_dir = self.args.inference.save_dir
        self.infer_remote_save_dir = self.args.inference.remote_save_dir
        if self.inference_type == 'save_probability':
            return self.inference_save_probability(batch_data)
        if self.inference_type == 'save_location_accuracy':
            self.zone_num = self.args.inference.get('zone_num', None)
            self.angle_config = self.args.inference.get('angle_config', None)
            assert self.zone_num is not None
            assert self.angle_config is not None
            self.eval_angle_bins = self.args.inference.get('eval_angle_bins', [0, 15, 370])
            self.accuracy_max_angle_diff = self.args.inference.get('accuracy_threshold', 15)
            return self.inference_location_accuracy(batch_data)
        return 0

    def inference_location_accuracy(self, batch_data):
        """
        inference function to evaluate localization accuracy
        """

        self.info_out = dict(bins=self.eval_angle_bins)
        nbatch = batch_data['mc_waveform'].shape[0]
        vad_info = batch_data['vad']
        target_angle = batch_data['direction'].detach().cpu().numpy()
        # parse vad info
        frame_vad_info = (
            F.pad(
                vad_info,
                (
                    self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                    self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                ),
                mode='constant',
                value=0.0,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        frame_vad_info = F.unfold(
            frame_vad_info,
            kernel_size=(1, self.feat_extractor.win_filter.shape[0]),
            stride=(1, self.feat_extractor.frame_length),
        )  # [1 filtersize T']
        frame_vad_info = torch.sum(frame_vad_info, dim=1)
        frame_vad_info = torch.gt(
            frame_vad_info, self.feat_extractor.win_filter.shape[0] // 2
        ).float()
        block_vad_info = frame_vad_info[
            :,
            : int(
                frame_vad_info.shape[-1]
                // self.feat_extractor.block_frame
                * self.feat_extractor.block_frame
            ),
        ]
        block_vad_info = torch.sum(
            block_vad_info.reshape(block_vad_info.shape[0], -1, self.feat_extractor.block_frame),
            dim=-1,
        )
        block_vad_info = (
            torch.gt(block_vad_info, self.feat_extractor.block_frame // 2).detach().cpu().numpy()
        )  # [B T'']
        # parse angle info
        zone2angle = np.linspace(
            start=self.angle_config.angle_min,
            stop=self.angle_config.angle_max,
            num=self.zone_num,
            endpoint=False,
        )
        angle_interval = (self.angle_config.angle_max - self.angle_config.angle_min) / self.zone_num
        zone2angle = zone2angle + angle_interval / 2.0
        # inference
        feat = self.feat_extractor(batch_data)
        est_loc_vec = self.ssl_net(feat)  # [B T zone_num]
        est_loc_zone_idx = torch.argmax(est_loc_vec, dim=-1).detach().cpu().numpy()
        est_loc_angle = zone2angle[est_loc_zone_idx]
        est_angle_diff = self.angle_dis(est_loc_angle, target_angle[:, None])  # [B T]
        for i in range(nbatch):
            target_angle_str = f"{target_angle[i]:.2f}"
            if target_angle_str not in self.info_out:
                self.info_out[target_angle_str] = dict(
                    diff_bins=np.zeros(len(self.eval_angle_bins) - 1, dtype=np.int64),
                    total_cnt=0,
                    accurate_cnt=0,
                )
            est_angle_diff_valid = est_angle_diff[i, block_vad_info[i]]
            res, _ = np.histogram(est_angle_diff_valid, bins=self.eval_angle_bins)
            acc_cnt = np.sum(est_angle_diff_valid <= self.accuracy_max_angle_diff)
            self.info_out[target_angle_str]['diff_bins'] = (
                res + self.info_out[target_angle_str]['diff_bins']
            )
            self.info_out[target_angle_str]['total_cnt'] = (
                est_angle_diff_valid.shape[-1] + self.info_out[target_angle_str]['total_cnt']
            )
            self.info_out[target_angle_str]['accurate_cnt'] = (
                acc_cnt + self.info_out[target_angle_str]['accurate_cnt']
            )
        return self.info_out

    def inference_save_probability(self, batch_data):
        """
        inference function to save probability of each zone
        batch_data: raw data for feature extraction
        """
        feat = self.feat_extractor(batch_data)
        est_loc_vec = self.ssl_net(feat)
        for i in range(est_loc_vec.shape[0]):
            data_name = batch_data['wavname'][i].split('.wav')[0]
            filepath = os.path.join(self.infer_save_dir, data_name + '.txt')
            with open(filepath, "w", encoding='utf-8') as f:
                np.savetxt(f, est_loc_vec[i].detach().cpu().numpy().T, fmt="%.4f", delimiter="\t")
            if self.infer_remote_save_dir:
                hdfs_put(filepath, self.infer_remote_save_dir)


@register_solution("BaseBFSolution")
class BaseBFSolution(BaseSolution):
    """
    Model BF:
    """

    def __init__(self, args):
        """
        args:
        feat_extractor: feature extractor name of solution
        aec_net_type: model name of solution
        criterion_type: criterion name of solution
        """
        super().__init__()
        self.feat_extractor = eval(args.feat_extractor_type)(args.feat_extractor)
        self.bf_net = eval(args.bf_net_type)(args.bf_net)
        self.criterion_module = eval(args.criterion_type)(args.criterion)
        self.args = args
        self.invalid_length = int(
            self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length
        )

    def forward(self, batch_data):
        """
        batch_data: raw data for feature extraction
        """
        feat = self.feat_extractor(batch_data)
        out_batch = self.bf_net(feat)
        pre_mask = out_batch['pre_mask']
        pre_spectra = out_batch['pre_spectra']
        pre_time = self.feat_extractor.subband_compose_transform(pre_spectra)
        critera_ipt_dict = dict(
            mask_ipt=pre_mask,
            mask_target=feat['target_mask'],
            ipt_time=pre_time,
            target_time=feat['target_time'],
            noisy_time=feat['noisy0_time'],
            mask_weight_in=feat['mask_weight'],
        )
        loss_list = self.criterion_module(**critera_ipt_dict)
        forward_out = OrderedDict()
        forward_out['backward_loss'] = loss_list[0]
        forward_out['cur_mse_loss'] = loss_list[1]
        forward_out['cur_relmse_loss'] = loss_list[2]
        forward_out['cur_sisnr_loss'] = loss_list[3]
        forward_out['cur_sisnr_imp'] = loss_list[4]

        # data_out = dict(
        #     noisy0_time=feat['noisy0_time'],
        #     target_time=feat['target_time'],
        #     pre_time=pre_time,
        #     target_spec=feat['target_spectra'],
        #     noisy_spec=feat['noisy_subband'][:, 0, ...],
        #     pre_spec=pre_spectra,
        # )
        return forward_out

    def inference(self, batch_data):
        """
        batch_data: raw data for feature extraction
        """
        self.inference_type = self.args.inference.type
        self.infer_save_dir = self.args.inference.save_dir
        self.infer_remote_save_dir = self.args.inference.remote_save_dir
        if self.inference_type == 'save_wav':
            self.fix_gain_on = self.args.inference.get('fix_gain_on', False)
            self.target_level = self.args.inference.get('target_level', 32767)
            return self.inference_save_wav(batch_data)
        if self.inference_type == 'spatial_response_and_sisnr':
            self.src_step = self.args.inference.src_direction_step
            if not hasattr(self.args.inference, 'net_direction_config'):
                self.net_direction_in_data = True
            else:
                self.net_direction_in_data = False
                self.net_direction_config = self.args.inference.net_direction_config
                self.net_direction = torch.linspace(**self.net_direction_config)
            return self.inference_sr_and_sisnr(batch_data)
        return 0

    def inference_save_wav(self, batch_data):
        """
        inference function to enhance noisy wav and save enhanced wav
        Args:
            batch_data: raw data for feature extraction
        """
        wav_length = batch_data['mc_waveform'].shape[-1]
        batch_data['clean_direction_use'] = batch_data['target_direction']
        feat = self.feat_extractor(batch_data)
        out_batch = self.bf_net(feat)
        pre_spectra = out_batch['pre_spectra']
        pre_time = self.feat_extractor.subband_compose_transform(pre_spectra)
        pre_time = pre_time[:, self.invalid_length : self.invalid_length + wav_length]
        pre_time = pre_time.detach().cpu().numpy()
        if self.fix_gain_on:
            pre_time = (
                pre_time
                / (np.max(np.abs(pre_time), axis=-1, keepdims=True) + 1e-6)
                * self.target_level
            )
        for i in range(pre_time.shape[0]):
            data_name = batch_data['wavname'][i].split('.wav')[0]
            filepath = os.path.join(self.infer_save_dir, data_name + '.wav')
            wavfile.write(
                filepath,
                self.feat_extractor.sample_rate,
                pre_time[i].astype(np.int16),
            )
            if self.infer_remote_save_dir:
                hdfs_put(filepath, self.infer_remote_save_dir)

    def inference_sr_and_sisnr(self, batch_data):
        """
        inference function to output beam pattern response and sisnr improvements
        Args:
            batch_data: raw data for feature extraction
        """
        # pylint:disable=too-many-branches
        wav_length = batch_data['mc_waveform'].shape[-1]
        nbatch = batch_data['mc_waveform'].shape[0]
        self.response_energy = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        target_data = batch_data.get('target_data', None)
        ori_sc_spectra = self.feat_extractor.subband_analyze_transform(
            batch_data['mc_waveform'][:, 0, :]
        )
        ori_sc_energy, _ = energy_phase(ori_sc_spectra)  # [B, T, F]
        if target_data is not None:
            sisnr_mean_in = sisnr_noreduce(batch_data['mc_waveform'][:, 0, :], target_data)
        if self.net_direction_in_data:
            batch_data['clean_direction_use'] = batch_data['net_direction'].float()
            feat = self.feat_extractor(batch_data)
            out_batch = self.bf_net(feat)
            pre_spectra = out_batch['pre_spectra']
            pre_energy, _ = energy_phase(pre_spectra)
            pre_time = self.feat_extractor.subband_compose_transform(pre_spectra)
            pre_time = pre_time[:, self.invalid_length : self.invalid_length + wav_length]
            if target_data is not None:
                sisnr_mean_out = sisnr_noreduce(pre_time, target_data)
                sisnr_mean_imp = sisnr_mean_out - sisnr_mean_in
            else:
                sisnr_mean_imp = None
            # parse vad info
            ovlp_frame_vad_info = (
                F.pad(
                    batch_data['ovlp_vad'],
                    (
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                    ),
                    mode='constant',
                    value=0.0,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )
            ovlp_frame_vad_info = F.unfold(
                ovlp_frame_vad_info,
                kernel_size=(1, self.feat_extractor.win_filter.shape[0]),
                stride=(1, self.feat_extractor.frame_length),
            )  # [1 filtersize T']
            ovlp_frame_vad_info = torch.sum(ovlp_frame_vad_info, dim=1)
            ovlp_frame_vad_info = torch.gt(
                ovlp_frame_vad_info, self.feat_extractor.win_filter.shape[0] // 2
            ).float()
            nonovlp_frame_vad_info = (
                F.pad(
                    batch_data['nonovlp_vad'],
                    (
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                    ),
                    mode='constant',
                    value=0.0,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )
            nonovlp_frame_vad_info = F.unfold(
                nonovlp_frame_vad_info,
                kernel_size=(1, self.feat_extractor.win_filter.shape[0]),
                stride=(1, self.feat_extractor.frame_length),
            )  # [1 filtersize T']
            nonovlp_frame_vad_info = torch.sum(nonovlp_frame_vad_info, dim=1)
            nonovlp_frame_vad_info = torch.gt(
                nonovlp_frame_vad_info, self.feat_extractor.win_filter.shape[0] // 2
            ).float()
            for i in range(pre_energy.shape[0]):
                sou_direction = batch_data['target_direction'][i].item()
                net_direction = batch_data['net_direction'][i].item()
                sou_direction = sou_direction // self.src_step * self.src_step
                if sisnr_mean_imp is not None:
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'sisnr_imp'
                    ] += sisnr_mean_imp[i].item()
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'sisnr_count'
                    ] += 1
                else:
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'sisnr_imp'
                    ] = sisnr_mean_imp
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'sisnr_count'
                    ] = 0
                # nonovlp mode
                nonovlp_vad_sum = torch.sum(batch_data['nonovlp_vad'][i, :]).item()
                ovlp_vad_sum = torch.sum(batch_data['ovlp_vad'][i, :]).item()
                if nonovlp_vad_sum > 0:
                    ori_sc_energy_nonovlp = torch.sum(
                        ori_sc_energy[i, :, :] * nonovlp_frame_vad_info[i, :, None], dim=0
                    ) / torch.sum(nonovlp_frame_vad_info[i])
                    pre_energy_nonovlp = torch.sum(
                        pre_energy[i, :, :] * nonovlp_frame_vad_info[i, :, None], dim=0
                    ) / torch.sum(nonovlp_frame_vad_info[i])
                    energy_res_nonovlp = torch.mean(
                        pre_energy_nonovlp / (ori_sc_energy_nonovlp + 1e-7)
                    ).item()
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'energy_response_nonovlp'
                    ] += energy_res_nonovlp
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'count_nonovlp'
                    ] += 1

                if ovlp_vad_sum > 0:
                    ori_sc_energy_ovlp = torch.sum(
                        ori_sc_energy[i, :, :] * ovlp_frame_vad_info[i, :, None], dim=0
                    ) / torch.sum(ovlp_frame_vad_info[i])
                    pre_energy_ovlp = torch.sum(
                        pre_energy[i, :, :] * ovlp_frame_vad_info[i, :, None], dim=0
                    ) / torch.sum(ovlp_frame_vad_info[i])
                    energy_res_ovlp = torch.mean(
                        pre_energy_ovlp / (ori_sc_energy_ovlp + 1e-7)
                    ).item()
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'energy_response_ovlp'
                    ] += energy_res_ovlp
                    self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                        'count_ovlp'
                    ] += 1
        else:
            # parse vad info
            ovlp_frame_vad_info = (
                F.pad(
                    batch_data['ovlp_vad'],
                    (
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                    ),
                    mode='constant',
                    value=0.0,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )
            ovlp_frame_vad_info = F.unfold(
                ovlp_frame_vad_info,
                kernel_size=(1, self.feat_extractor.win_filter.shape[0]),
                stride=(1, self.feat_extractor.frame_length),
            )  # [1 filtersize T']
            ovlp_frame_vad_info = torch.sum(ovlp_frame_vad_info, dim=1)
            ovlp_frame_vad_info = torch.gt(
                ovlp_frame_vad_info, self.feat_extractor.win_filter.shape[0] // 2
            ).float()
            nonovlp_frame_vad_info = (
                F.pad(
                    batch_data['nonovlp_vad'],
                    (
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                        self.feat_extractor.win_filter.shape[0] - self.feat_extractor.frame_length,
                    ),
                    mode='constant',
                    value=0.0,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )
            nonovlp_frame_vad_info = F.unfold(
                nonovlp_frame_vad_info,
                kernel_size=(1, self.feat_extractor.win_filter.shape[0]),
                stride=(1, self.feat_extractor.frame_length),
            )  # [1 filtersize T']
            nonovlp_frame_vad_info = torch.sum(nonovlp_frame_vad_info, dim=1)
            nonovlp_frame_vad_info = torch.gt(
                nonovlp_frame_vad_info, self.feat_extractor.win_filter.shape[0] // 2
            ).float()
            for _, net_direction in enumerate(self.net_direction):
                batch_data['clean_direction_use'] = net_direction * torch.ones(
                    nbatch, dtype=torch.float32
                )
                feat = self.feat_extractor(batch_data)
                out_batch = self.bf_net(feat)
                pre_spectra = out_batch['pre_spectra']
                pre_energy, _ = energy_phase(pre_spectra)
                pre_time = self.feat_extractor.subband_compose_transform(pre_spectra)
                pre_time = pre_time[:, self.invalid_length : self.invalid_length + wav_length]
                if target_data is not None:
                    sisnr_mean_out = sisnr_noreduce(pre_time, target_data)
                    sisnr_mean_imp = sisnr_mean_out - sisnr_mean_in
                else:
                    sisnr_mean_imp = None
                for i in range(pre_energy.shape[0]):
                    sou_direction = batch_data['target_direction'][i].item()
                    sou_direction = sou_direction // self.src_step * self.src_step
                    if sisnr_mean_imp is not None:
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'sisnr_imp'
                        ] += sisnr_mean_imp[i].item()
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'sisnr_count'
                        ] += 1
                    else:
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'sisnr_imp'
                        ] = sisnr_mean_imp
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'sisnr_count'
                        ] = 0
                    # nonovlp mode
                    nonovlp_vad_sum = torch.sum(batch_data['nonovlp_vad'][i, :]).item()
                    ovlp_vad_sum = torch.sum(batch_data['ovlp_vad'][i, :]).item()
                    if nonovlp_vad_sum > 0:
                        ori_sc_energy_nonovlp = torch.sum(
                            ori_sc_energy[i, :, :] * nonovlp_frame_vad_info[i, :, None], dim=0
                        ) / torch.sum(nonovlp_frame_vad_info[i])
                        pre_energy_nonovlp = torch.sum(
                            pre_energy[i, :, :] * nonovlp_frame_vad_info[i, :, None], dim=0
                        ) / torch.sum(nonovlp_frame_vad_info[i])
                        energy_res_nonovlp = torch.mean(
                            pre_energy_nonovlp / (ori_sc_energy_nonovlp + 1e-7)
                        ).item()
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'energy_response_nonovlp'
                        ] += energy_res_nonovlp
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'count_nonovlp'
                        ] += 1
                    if ovlp_vad_sum > 0:
                        ori_sc_energy_ovlp = torch.sum(
                            ori_sc_energy[i, :, :] * ovlp_frame_vad_info[i, :, None], dim=0
                        ) / torch.sum(ovlp_frame_vad_info[i])
                        pre_energy_ovlp = torch.sum(
                            pre_energy[i, :, :] * ovlp_frame_vad_info[i, :, None], dim=0
                        ) / torch.sum(ovlp_frame_vad_info[i])
                        energy_res_ovlp = torch.mean(
                            pre_energy_ovlp / (ori_sc_energy_ovlp + 1e-7)
                        ).item()
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'energy_response_ovlp'
                        ] += energy_res_ovlp
                        self.response_energy[f"{net_direction:.2f}"][f"{sou_direction:.2f}"][
                            'count_ovlp'
                        ] += 1
        return self.response_energy


class DenoiseModelExporter(BaseInfer):
    '''export denoise model to onnx'''

    NAME = 'denoise'
    INPUTS = [
        FalconDict(name='noisy_i', type=torch.float32, shape=['B', 'C', 'T', 'H']),
        FalconDict(name='noisy_r', type=torch.float32, shape=['B', 'C', 'T', 'H']),
    ]
    OUTPUTS = [
        FalconDict(name='enh_i', type=torch.float32, shape=['B', 'T', 'H']),
        FalconDict(name='enh_r', type=torch.float32, shape=['B', 'T', 'H']),
    ]

    def __init__(self, model, **kwargs):
        '''init'''
        super().__init__(kwargs)
        self.model = model

    def forward(self, noisy_i, noisy_r):
        '''
        Forward for BaseDenoisrSolution module
        '''
        feat = OrderedDict()
        feat['noisy_i'] = noisy_i
        feat['noisy_r'] = noisy_r
        output = self.model.denoise_net(feat)
        enh_r = output["enh_r"]
        enh_i = output["enh_i"]
        return enh_i, enh_r

    def sample_inputs(self):
        '''generate sample inputs'''
        if self.model.feat_extractor.frame_length > 160:
            feature_dim = self.model.feat_extractor.frame_length // 2 + 1
        else:
            feature_dim = 161
        batch_dim = 5
        channel_dim = 6
        time_dim = 800
        noisy_i = self._generate_input_data(
            0, dynamic_axis=[batch_dim, channel_dim, time_dim, feature_dim], low=-16, high=16
        )
        noisy_r = self._generate_input_data(
            1, dynamic_axis=[batch_dim, channel_dim, time_dim, feature_dim], low=-16, high=16
        )

        return noisy_i, noisy_r


@register_solution("BaseDenoiseSolution")
class BaseDenoiseSolution(BaseSolution):
    """
    Model Denoise:
    """

    def __init__(self, args):
        """
        args:
        feat_extractor: feature extractor name of solution
        aec_net_type: model name of solution
        criterion_type: criterion name of solution
        """
        super().__init__()
        self.feat_extractor = eval(args.feat_extractor_type)(args.feat_extractor)
        self.denoise_net = eval(args.denoise_net_type)(args.denoise_net).cuda()
        self.criterion_module = eval(args.criterion_type)(args.criterion)
        self.args = args

    def forward(self, batch_data):
        """
        Args:
            input:
                noisy (Batch x Channel x Time): input noisy signals
                clean (Batch x Channel x Time): target clean speech signals
            output:
                loss (Batch): loss of denoise training
        """
        feat = self.feat_extractor(batch_data)
        output = self.denoise_net(feat)
        enh_r = output["enh_r"]
        enh_i = output["enh_i"]

        if self.args.feat_extractor_type == "DenoiseStftFeatExtractor":
            enh = self.feat_extractor.inverse_stft(enh_r, enh_i)
            clean_r = feat["clean_r"][:, 0, :, :]
            clean_i = feat["clean_i"][:, 0, :, :]
            clean = self.feat_extractor.inverse_stft(clean_r, clean_i)
        else:
            ri_out = torch.cat((enh_r.unsqueeze(-1), enh_i.unsqueeze(-1)), -1)
            enh = self.feat_extractor.subband_compose_transform(ri_out)
            clean = self.feat_extractor.subband_compose_transform(ri_out)
            if enh.ndim == 1:
                enh = enh.unsqueeze(0)
            enh = enh[:, self.feat_extractor.frame_length * self.feat_extractor.fb_delay :]
            ri_out_clean = torch.cat(
                (
                    feat["clean_r"][:, 0, :, :].unsqueeze(-1),
                    feat["clean_i"][:, 0, :, :].unsqueeze(-1),
                ),
                -1,
            )
            clean = self.feat_extractor.subband_compose_transform(ri_out_clean)
            if clean.ndim == 1:
                clean = enh.unsqueeze(0)
            clean = clean[:, self.feat_extractor.frame_length * self.feat_extractor.fb_delay :]

        output["source"] = clean
        output["estimate_source"] = enh
        output["clean_r"] = feat["clean_r"].squeeze(1)
        output["clean_i"] = feat["clean_i"].squeeze(1)

        # cal loss
        loss = self.criterion_module(output)

        forward_out = OrderedDict()
        forward_out["backward_loss"] = loss
        return forward_out

    def inference(self, batch_data):
        """
        Args:
            input:
                noisy (Batch x Channel x Time): input noisy signals
                clean (Batch x Channel x Time): target clean speech signals
            output:
                loss (Batch): loss of denoise training
        """
        feat = self.feat_extractor(batch_data)
        output = self.denoise_net(feat)
        enh_r = output["enh_r"]
        enh_i = output["enh_i"]

        if self.args.feat_extractor_type == "DenoiseStftFeatExtractor":
            enh = 32768 * self.feat_extractor.inverse_stft(enh_r, enh_i)
        else:
            ri_out = torch.cat((enh_r.unsqueeze(-1), enh_i.unsqueeze(-1)), -1)
            enh = self.feat_extractor.subband_compose_transform(ri_out)
            if enh.ndim == 1:
                enh = enh.unsqueeze(0)
            enh = 32768 * enh[:, self.feat_extractor.frame_length * self.feat_extractor.fb_delay :]

        forward_out = OrderedDict()
        forward_out["enh_wav"] = enh
        forward_out['uttid'] = batch_data['uttid']
        if self.args.inference.get('cal_pesq', True):
            forward_out['pesq'] = compute_pesq2(
                batch_data['clean_waveform'][0, 0, :].cpu().numpy(), enh[0].cpu().numpy()
            )
        if self.args.inference.get('cal_sisnr', True):
            forward_out['sisnr'] = (
                -si_snr(
                    enh,
                    batch_data['clean_waveform'][:, 0, :],
                )
                .cpu()
                .numpy()
            )

        if self.args.inference.get("save_enh_wav", False):
            self.save(self.args, forward_out)
        return forward_out

    def save(self, args, data_dict):
        '''save enh_wav'''
        inference_cfg = args.inference
        wav = data_dict['enh_wav'].cpu().numpy()[0]  # choose first batch
        wav = wav[np.newaxis, :]  # add channel dim
        uttid = data_dict['uttid'][0]
        sample_rate = data_dict.get('sample_rate', 16000)
        save_name = os.path.join(inference_cfg.save_dir, f"{uttid}")
        wavfile.write(save_name, sample_rate, wav.transpose(1, 0).astype(np.int16))
        if inference_cfg.get('remote_save_dir', None):
            hdfs_put(save_name, inference_cfg.remote_save_dir)

    def register_infers(self):
        '''register infers'''
        infers = [DenoiseModelExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)
