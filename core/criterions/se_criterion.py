"""
designed speech enhancement loss caculators
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# pylint: disable='import-error'
from torch_stft import STFT
from librosa.filters import mel as librosa_mel_fn

try:
    from pypesq import pesq
except ImportError:
    pesq = None


def l2norm(mat, keepdim=False):
    '''Calculate l2 norm.'''
    return torch.norm(mat, dim=-1, keepdim=keepdim)


def l2_norm(s1, s2):
    '''calculate l2_norm'''
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


# pylint: disable='redefined-outer-name'
def sisnr_noreduce(ipt, target):
    '''Scale Invariant signal noise ratio(Sisnr) loss.
    ipt(se): input time domain signal [B, T]
    target(st): target time domain signal [B, T]
    EQUATION:
        proj = <st, se>st/||st||^2
        err = se - proj
        sisnr = norm(proj) / norm(err)
    '''
    ipt_len = ipt.shape[1]
    target_len = target.shape[1]
    if ipt_len < target_len:
        target = target[:, :ipt_len]
    elif ipt_len > target_len:
        ipt = ipt[:, :target_len]

    epsilon_e = 1e-7
    x_zm = ipt - torch.mean(ipt, dim=-1, keepdim=True)
    s_zm = target - torch.mean(target, dim=-1, keepdim=True)
    t_zm = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        * s_zm
        / (l2norm(s_zm, keepdim=True) ** 2 + epsilon_e)
    )
    si_snr = 20 * torch.log10(epsilon_e + l2norm(t_zm) / (l2norm(x_zm - t_zm) + epsilon_e))
    return si_snr


def sisnr(ipt, target):
    '''Scale Invariant signal noise ratio(Sisnr) loss.
    ipt: input time domain signal [B, T]
    target: target time domain signal [B, T]
    '''
    ipt_len = ipt.shape[1]
    target_len = target.shape[1]
    if ipt_len < target_len:
        target = target[:, :ipt_len]
    elif ipt_len > target_len:
        ipt = ipt[:, :target_len]

    epsilon_e = 1e-7
    x_zm = ipt - torch.mean(ipt, dim=-1, keepdim=True)
    s_zm = target - torch.mean(target, dim=-1, keepdim=True)
    t_zm = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        * s_zm
        / (l2norm(s_zm, keepdim=True) ** 2 + epsilon_e)
    )
    loss_si_snr = 20 * torch.log10(epsilon_e + l2norm(t_zm) / (l2norm(x_zm - t_zm) + epsilon_e))
    return -1.0 * torch.sum(loss_si_snr) / len(loss_si_snr)


# pylint: disable='redefined-outer-name'
def si_snr_utterance(ipt, target, wav_mask=None):
    """
    compute utterance level sisnr loss
    """
    amp_scale_max = 1.5
    amp_scale_min = 1 / amp_scale_max
    assert ipt.size() == target.size()
    epsilon = 1e-8
    if wav_mask is None:
        wav_mask = torch.ones_like(ipt)

    ipt_mean = torch.sum(ipt * wav_mask, dim=1, keepdim=True) / wav_mask.sum(dim=1, keepdim=True)
    target_mean = torch.sum(target * wav_mask, dim=1, keepdim=True) / wav_mask.sum(
        dim=1, keepdim=True
    )
    s_estimate = ipt - ipt_mean
    s_target = target - target_mean

    pair_wise_dot = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(s_target**2, dim=-1, keepdim=True) + epsilon

    scale = pair_wise_dot / s_target_energy
    scale = torch.clamp(scale, min=amp_scale_min, max=amp_scale_max)

    pair_wise_proj = scale * s_target
    e_noise = s_estimate - pair_wise_proj
    si_snr = (torch.sum((pair_wise_proj * wav_mask) ** 2, dim=-1) + epsilon) / (
        torch.sum((e_noise * wav_mask) ** 2, dim=-1) + epsilon
    )

    si_snr = 10 * torch.log10(si_snr + epsilon)
    si_snr = 0 - si_snr

    return si_snr


# pylint: disable='invalid-name'
def compute_pesq2(clean_signal, noisy_signal, sr=16000):
    '''
    calucate denoise signal diff
    '''
    length = min(clean_signal.shape[-1], noisy_signal.shape[-1])
    return pesq(clean_signal[..., :length] / 32768.0, noisy_signal[..., :length] / 32768.0, sr)


def si_snr(s1, s2, eps=1e-8):
    '''args: s1, s2: [B, T]'''
    min_length = min(s1.shape[-1], s2.shape[-1])
    s1 = s1[..., :min_length]
    s2 = s2[..., :min_length]
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def echo_reduce_db(sample_origin, sample_enh, eps=1e-10):
    '''calucate reduce score'''
    reduce_db = 10 * torch.log10(torch.sum(sample_origin**2) / (torch.sum(sample_enh**2) + eps))
    return reduce_db


class MixSisnrMelMseAsyn(nn.Module):
    """Mix loss of sisnr and mel mse async.

    Example::

        # criterion
        criterion_type='MixSisnrMelMseAsyn',
        criterion=dict(
            alpha=2,
            lamda=0,
            fs=16000,
            n_fft=256,
            n_mels=80,
            fmin=0
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = args.alpha
        self.lamda = args.lamda
        mel_basis = librosa_mel_fn(sr=args.fs, n_fft=args.n_fft, n_mels=args.n_mels, fmin=args.fmin)
        mel_basis = torch.from_numpy(mel_basis).float().cuda()
        self.register_buffer("mel_basis", mel_basis)

    def mix_sisnr_mel_mse_asyn(self, clean, enh, pred_spec, target_spec, irm):
        """Calculate the Sinr+MSE loss for variable length dataset.

        Args:

            - clean: clean time domain signal [B, T]
            - enh: enhanced time domain signal [B, T]
            - pred_spec: predicted freq domain signal [B, T, F]
            - target_spec: target freq domain signal [B, T, F]
            - irm: ideal ratio mask [B, T, F]
        """
        alpha = self.alpha
        with torch.no_grad():
            binary_mask = torch.gt(irm, -0.5).cuda().float()

        penality = target_spec > pred_spec
        x_f = (target_spec - pred_spec) * binary_mask
        g_f = alpha * penality * x_f + ~penality * x_f
        loss_mse = (g_f**2).sum() / (binary_mask.sum() + 1e-10)
        loss_sisnr = sisnr(enh, clean)
        loss = self.lamda * loss_mse + loss_sisnr
        return loss.mean()

    def forward(self, feat, mask, enh):
        """Calculate mix_sisnr_mel_mse_asyn loss

        Args:

            - feat: input feats
            - mask: output model predicted freq domain mask
            - enh: output model predicted enhanced time signal

        Returns:

            - mix_sisnr_mel_mse_asyn:Sinr+MSE loss
        """
        ehn_r = mask * feat["fir_r"]
        ehn_i = mask * feat["fir_i"]
        magnitude = torch.sqrt(ehn_r**2 + ehn_i**2 + 1e-9)
        mel_output = torch.matmul(self.mel_basis, magnitude.permute(0, 2, 1)) + 1e-9
        log_mel_spec_enh = torch.log(mel_output)

        magnitudec = torch.sqrt(feat["clean_r"] ** 2 + feat["clean_i"] ** 2 + 1e-9)
        mel_outputc = torch.matmul(self.mel_basis, magnitudec)
        log_mel_spec_clean = torch.log(mel_outputc + 1e-9)
        irm = torch.ones(log_mel_spec_clean.shape).to(log_mel_spec_clean.device)
        irm[(mel_output < 1.0e-10) & (mel_outputc < 1.0e-10)] = -1
        return self.mix_sisnr_mel_mse_asyn(
            feat['clean'], enh, log_mel_spec_enh, log_mel_spec_clean, irm
        )


class NnbeamHybridMseSisnr(nn.Module):
    '''loss used for nnbeam training, which is a mix of sisnr and mse.'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight_mse = args.get("weight_mse", 1.0)
        self.weight_rmse = args.get("weight_rmse", 1.0)
        self.weight_sisnr = args.get("weight_sisnr", 1.0)
        self.weight_sisnri = args.get("weight_sisnri", 1.0)

    def forward(self, batch_data):
        '''Calculate nnbeam loss'''
        epsilon_e = 1e-7

        ipt_wav = batch_data['target_waveform_pred']
        target_wav = batch_data['target_waveform']
        noisy_wav = batch_data['noisy_waveform']
        ipt_mask = batch_data['target_mask_pred']
        target_mask = batch_data['target_mask']
        wav_mask = batch_data['wav_mask']
        if wav_mask.ndim == 3:
            wav_mask = wav_mask[:, :, 0]
        spec_mask = batch_data['src_mask']
        if spec_mask.shape[1] < target_mask.shape[1]:
            spec_mask = F.pad(
                spec_mask, [0, target_mask.shape[1] - spec_mask.shape[1]], mode="constant", value=0
            )
        if wav_mask.shape[1] < target_wav.shape[1]:
            wav_mask = F.pad(
                wav_mask, [0, target_wav.shape[1] - wav_mask.shape[1]], mode="constant", value=0
            )
        elif wav_mask.shape[1] > target_wav.shape[1]:
            wav_mask = wav_mask[:, : target_wav.shape[1]]
        mask_weight_in = None
        if 'mask_weight' in batch_data:
            mask_weight_in = batch_data["mask_weight"]
        with torch.no_grad():
            mask_over = torch.sign(ipt_mask - target_mask) > 0
            mask_under = torch.sign(ipt_mask - target_mask) <= 0
            freq_len = ipt_mask.shape[-1]
            mask_weight = (
                (
                    torch.cos(
                        torch.arange(0, freq_len).to(ipt_mask.device) / (freq_len * 2.5) * np.pi
                    )
                    * 0.4
                    + 0.6
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if mask_weight_in is not None:
                mask_weight = mask_weight * (mask_weight_in + 0.1)

        mask_mse = torch.abs(ipt_mask - target_mask) * mask_weight
        mask_mse = mask_mse * mask_over + mask_mse * mask_under
        mask_mse = ((mask_mse**2).mean(-1) * spec_mask).sum() / ((spec_mask).sum() + epsilon_e)
        # mask_mse = mask_mse * self.weight_mse

        mask_rmse = torch.abs(ipt_mask - target_mask) * mask_weight
        mask_rmse = (
            ((mask_rmse * spec_mask.unsqueeze(-1)) ** 2)
            / ((target_mask * spec_mask.unsqueeze(-1)) ** 2 + 0.15)
        ).mean(-1).sum() / (spec_mask.sum() + epsilon_e)
        # mask_rmse = mask_rmse * self.weight_rmse

        sisnro = torch.mean(si_snr_utterance(ipt_wav, target_wav, wav_mask))
        # sisnro = sisnro * self.weight_sisnr

        sisnri = torch.mean(
            si_snr_utterance(noisy_wav, target_wav, wav_mask)
            - si_snr_utterance(ipt_wav, target_wav, wav_mask)
        )
        # sisnri = sisnri * self.weight_sisnri

        src_mask = batch_data['src_mask']
        frame_size = src_mask.float().sum()

        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['frame_size'] = frame_size
        forward_out["loss"] = (
            self.weight_mse * mask_mse
            + self.weight_rmse * mask_rmse
            + self.weight_sisnr * sisnro
            + self.weight_sisnri * sisnri
        )
        forward_out["mse_loss"] = mask_mse
        forward_out["rmse_loss"] = mask_rmse
        forward_out["sisnr_loss"] = sisnro
        forward_out["sisnri_loss"] = sisnri
        forward_out["backward_loss"] = forward_out["loss"]
        return forward_out


class NnbeamRnntHybrid(nn.Module):
    '''Nnbeam loss used for hybrid training with Rnnt.'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight_mse = args.get("weight_mse", 1.0)
        self.weight_rmse = args.get("weight_rmse", 1.0)
        self.weight_sisnr = args.get("weight_sisnr", 1.0)
        self.weight_sisnri = args.get("weight_sisnri", 1.0)
        self.weight_rnnt = args.get("weight_rnnt", 1.0)

    def forward(self, batch_data, asr_res):
        """forward"""
        epsilon_e = 1e-7

        ipt_wav = batch_data['target_waveform_pred']
        target_wav = batch_data['target_waveform']
        noisy_wav = batch_data['noisy_waveform']
        ipt_mask = batch_data['target_mask_pred']
        target_mask = batch_data['target_mask']
        wav_mask = batch_data['wav_mask']
        if wav_mask.ndim == 3:
            wav_mask = wav_mask[:, :, 0]
        spec_mask = batch_data['src_mask']
        if spec_mask.shape[1] < target_mask.shape[1]:
            spec_mask = F.pad(
                spec_mask, [0, target_mask.shape[1] - spec_mask.shape[1]], mode="constant", value=0
            )
        if wav_mask.shape[1] < target_wav.shape[1]:
            wav_mask = F.pad(
                wav_mask, [0, target_wav.shape[1] - wav_mask.shape[1]], mode="constant", value=0
            )
        elif wav_mask.shape[1] > target_wav.shape[1]:
            wav_mask = wav_mask[:, : target_wav.shape[1]]
        mask_weight_in = None
        if 'mask_weight' in batch_data:
            mask_weight_in = batch_data["mask_weight"]
        with torch.no_grad():
            mask_over = torch.sign(ipt_mask - target_mask) > 0
            mask_under = torch.sign(ipt_mask - target_mask) <= 0
            freq_len = ipt_mask.shape[-1]
            mask_weight = (
                (
                    torch.cos(
                        torch.arange(0, freq_len).to(ipt_mask.device) / (freq_len * 2.5) * np.pi
                    )
                    * 0.4
                    + 0.6
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if mask_weight_in is not None:
                mask_weight = mask_weight * (mask_weight_in + 0.1)

        mask_mse = torch.abs(ipt_mask - target_mask) * mask_weight
        mask_mse = mask_mse * mask_over + mask_mse * mask_under
        mask_mse = ((mask_mse**2).mean(-1) * spec_mask).sum() / ((spec_mask).sum() + epsilon_e)
        # mask_mse = mask_mse * self.weight_mse

        mask_rmse = torch.abs(ipt_mask - target_mask) * mask_weight
        mask_rmse = (
            ((mask_rmse * spec_mask.unsqueeze(-1)) ** 2)
            / ((target_mask * spec_mask.unsqueeze(-1)) ** 2 + 0.15)
        ).mean(-1).sum() / ((spec_mask).sum() + epsilon_e)
        # mask_rmse = mask_rmse * self.weight_rmse

        sisnro = torch.mean(si_snr_utterance(ipt_wav, target_wav, wav_mask))
        # sisnr = sisnr * self.weight_sisnr

        sisnri = torch.mean(
            si_snr_utterance(noisy_wav, target_wav, wav_mask)
            - si_snr_utterance(ipt_wav, target_wav, wav_mask)
        )
        # sisnri = sisnri * self.weight_sisnri

        src_mask = batch_data['src_mask']
        frame_size = src_mask.float().sum()

        forward_out = OrderedDict()
        forward_out['nll_loss'] = asr_res['nll_loss']
        forward_out['rnnt_loss'] = asr_res['loss']
        forward_out['acc'] = asr_res['acc']
        forward_out['tgt_size'] = asr_res['tgt_size']
        forward_out['cer'] = asr_res['cer']
        forward_out['dist'] = asr_res['dist']
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['frame_size'] = frame_size
        forward_out["loss"] = (
            self.weight_mse * mask_mse
            + self.weight_rmse * mask_rmse
            + self.weight_sisnr * sisnro
            + self.weight_sisnri * sisnri
            + self.weight_rnnt * forward_out['rnnt_loss']
        )
        forward_out["mse_loss"] = mask_mse
        forward_out["rmse_loss"] = mask_rmse
        forward_out["sisnr_loss"] = sisnro
        forward_out["sisnri_loss"] = sisnri
        forward_out["backward_loss"] = forward_out["loss"]
        return forward_out


class SSL(nn.Module):
    """criterion for ssl training

    Example::

        criterion_type='SSL',
        criterion=dict(
        criterion_subtype="mse"
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.criterion_subtype = args.criterion_subtype

    @staticmethod
    def loss_function_bce(pre_vec, tgt_vec):
        """function of bce loss

        Args:

            - pre_vec: output vector
            - tgt_vec: target vector
        """
        loss_func = nn.BCELoss()
        loss = loss_func(pre_vec, tgt_vec)
        return loss

    @staticmethod
    def loss_function_mse(pre_vec, tgt_vec):
        """function of mse loss

        Args:

            - pre_vec: output vector
            - tgt_vec: target vector
        """
        loss = torch.mean((pre_vec - tgt_vec) ** 2)
        return loss

    def forward(self, pre_vec, tgt_vec):
        """forward process

        Args:

            - pre_vec: output vector
            - tgt_vec: target vector
        """
        loss_func = 'loss_function_' + self.criterion_subtype
        loss_func = getattr(self, loss_func)
        return loss_func(pre_vec, tgt_vec)


class BFMseRelMseSisnr(nn.Module):
    """criterion for ssl training

    Example::

        criterion_type='BFMseRelMseSisnr',
        criterion=dict(
        mask_coeff = 40.0,
        rel_mask_coeff = 25.0,
        sisnr_coeff = 0.0,
        mask_under_penalty = 1.1,
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.eps = 1e-7
        self.mask_coeff = args.mask_coeff
        self.rel_mask_coeff = args.rel_mask_coeff
        self.sisnr_coeff = args.sisnr_coeff
        self.mask_under_penalty = args.mask_under_penalty

    def forward(
        self,
        mask_ipt=None,
        mask_target=None,
        ipt_time=None,
        target_time=None,
        noisy_time=None,
        mask_weight_in=None,
        sample_mask=None,
    ):
        """forward process

        Args:

        - mask_target: feat['target_mask']
        - ipt_time: pre_time,sisnro is linked to ipt_time
        - target_time: feat['target_time'],sisnro and sisnr_before is linked to target_time
        - noisy_time: feat['noisy0_time'],sisnr_before is linked to noisy_yime
        - mask_weight_in: feat['mask_weight'],use to count mask_weight
        """
        with torch.no_grad():
            binary_mask = torch.gt(mask_target, -2.0).float()
            mask_over = torch.sign(mask_ipt - mask_target) > 0
            mask_under = torch.sign(mask_ipt - mask_target) <= 0
            freq_len = mask_ipt.shape[-1]
            mask_weight = (
                (
                    torch.cos(
                        torch.arange(0, freq_len).to(mask_ipt.device) / (freq_len * 2.5) * np.pi
                    )
                    * 0.4
                    + 0.6
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if mask_weight_in is not None:
                mask_weight = mask_weight * (mask_weight_in + 0.1)
            if sample_mask is None:
                sample_mask = torch.ones(mask_target.shape[0], device=mask_target.device)

        mask_mse = torch.abs(mask_ipt - mask_target) * mask_weight
        loss_weight = (
            mask_mse * mask_over.to(mask_mse.device)
            + mask_mse * mask_under.to(mask_mse.device) * self.mask_under_penalty
        )
        loss_mask = (loss_weight**2).sum((1, 2)) / ((binary_mask).sum((1, 2)) + self.eps)
        relative_err = mask_mse
        loss_relative_mask = torch.mean(
            (relative_err**2) / ((mask_target * binary_mask) ** 2 + 0.15), dim=(1, 2)
        )
        loss_mask = torch.sum(loss_mask * sample_mask) / torch.sum(sample_mask)
        loss_relative_mask = torch.sum(loss_relative_mask * sample_mask) / torch.sum(sample_mask)

        sisnro = si_snr_utterance(ipt_time, target_time)
        sisnr_before = si_snr_utterance(noisy_time, target_time)
        sisnr_imp = torch.sum((sisnr_before - sisnro) * sample_mask) / torch.sum(sample_mask)
        sisnro = torch.sum(sisnro * sample_mask) / torch.sum(sample_mask)

        loss = (
            loss_mask * self.mask_coeff
            + loss_relative_mask * self.rel_mask_coeff
            + sisnro * self.sisnr_coeff
        )
        return loss, loss_mask, loss_relative_mask, sisnro, sisnr_imp


class DenoisePcsloss(nn.Module):
    """criterion for denoise training

    Example::

        criterion_type='DenoisePcsloss',
        criterion=dict(
        compress_coeff = 0.5,
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.eps = 1e-8
        self.compress_coeff = args.compress_coeff

    def forward(
        self,
        output,
    ):
        """forward process

        Args:
        - target_r: clean reat part
        - target_i: clean imaginary part
        - pred_r: enhance real part
        - pred_i: enhance imaginary part
        - loss = |MAG_target - MAG_pred| + (Complex_target - Complex_pred)**2
        - Magnitude is compressed by compress_coeff.
        """
        target_r = output["clean_r"]
        target_i = output["clean_i"]
        pred_r = output["enh_r"]
        pred_i = output["enh_i"]
        target_mag = torch.sqrt(target_r**2 + target_i**2 + self.eps)
        target_mag = torch.pow(target_mag + self.eps, self.compress_coeff)
        target_phase = torch.atan2(target_i + self.eps, target_r + self.eps)
        target_reg_r = target_mag * torch.cos(target_phase)
        target_reg_i = target_mag * torch.sin(target_phase)

        pred_mag = torch.sqrt(pred_r**2 + pred_i**2 + self.eps)
        pred_mag = torch.pow(pred_mag + self.eps, self.compress_coeff)
        pred_phase = torch.atan2(pred_i + self.eps, pred_r + self.eps)
        pred_reg_r = pred_mag * torch.cos(pred_phase)
        pred_reg_i = pred_mag * torch.sin(pred_phase)

        loss_mag = (target_mag - pred_mag) ** 2
        loss_phase = (pred_reg_r - target_reg_r) ** 2 + (pred_reg_i - target_reg_i) ** 2
        loss = loss_mag + loss_phase
        loss = torch.mean(loss)
        return loss


class FrequencyMse(nn.Module):
    """criterion for denoise training

    Example::

        criterion_type='FrequencyMse',
        criterion=dict(
        weight = 0.3,
        beta = 0.5
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.eps = 1e-8
        self.weight = args.weight
        self.beta = args.beta
        self.trans = STFT(filter_length=1024, hop_length=160, win_length=1024, window='hann').to(
            'cuda'
        )

    def forward(self, output):
        """forward process

        Args:

        - source: clean wav: [B, T]
        - estimate_source: estimate wav: [B, T]
        - loss = |MAG_target - MAG_pred| ** beta + (Complex_target - Complex_pred)**2
        - Magnitude is compressed by beta.
        """
        source = output["source"]
        estimate_source = output["estimate_source"]

        source_mag, source_pha = self.trans.transform(source[:, :] + self.eps)
        estimate_mag, estimate_pha = self.trans.transform(estimate_source[:, :] + self.eps)

        source_r = ((source_mag + self.eps) ** self.beta) * torch.cos(source_pha)
        source_i = ((source_mag + self.eps) ** self.beta) * torch.sin(source_pha)
        estimate_source_r = ((estimate_mag + self.eps) ** self.beta) * torch.cos(estimate_pha)
        estimate_source_i = ((estimate_mag + self.eps) ** self.beta) * torch.sin(estimate_pha)

        loss_mag = 0.0
        loss_cpx = 0.0
        loss_mag += torch.mean(
            (
                (source_mag[:, 1:, :] + self.eps) ** self.beta
                - (estimate_mag[:, 1:, :] + self.eps) ** self.beta
            )
            ** 2.0
        )
        loss_cpx += torch.mean(
            ((source_r[:, 1:, :] - estimate_source_r[:, 1:, :]) ** 2.0)
            + ((source_i[:, 1:, :] - estimate_source_i[:, 1:, :]) ** 2.0)
        )

        loss = (1 - self.weight) * loss_mag + self.weight * loss_cpx
        return loss


class EchoAwareLoss(nn.Module):
    """criterion for denoise training

    Example::
        criterion_type='EchoAwareLoss',
        criterion=dict(
            compress_coeff = 0.3,
        ),
    """

    def __init__(self, args):
        super().__init__()
        self.eps = 1e-8
        self.compress_coeff = args.compress_coeff  # 0.3 by defualt

    def forward(self, output):
        """forward process

        Args:

        - speech_stft: clean wav stft: [B, T, F, 2]
        - ehn_stft: estimate wav stft: [B, T, F, 2]
        - loss = w_echo*|MAG_target - MAG_pred|**compress_coeff + (Complex_target-Complex_pred)**2
        """
        speech_pow = output['speech_stft'].pow(2).sum(dim=-1)
        echo_pow = (output['mic_stft'] - output['speech_stft']).pow(2).sum(dim=-1)
        w_echo = echo_pow / (echo_pow + speech_pow + self.eps)

        compressed_speech_mag = speech_pow.clamp(min=self.eps).pow(0.5 * self.compress_coeff)
        compressed_est_mag = (
            output['ehn_stft'].pow(2).sum(dim=-1).clamp(min=self.eps).pow(0.5 * self.compress_coeff)
        )

        speech_im = output['speech_stft'][..., 1]
        speech_re = output['speech_stft'][..., 0]
        mask = torch.sign(speech_re)
        mask = torch.where(mask == 0, mask.new_ones(mask.shape), mask)
        speech_re = speech_re.abs().clamp(min=self.eps) * mask
        speech_pha = torch.atan2(speech_im, speech_re)
        est_im = output['ehn_stft'][..., 1]
        est_re = output['ehn_stft'][..., 0]
        mask = torch.sign(est_re)
        mask = torch.where(mask == 0, mask.new_ones(mask.shape), mask)
        est_re = est_re.abs().clamp(min=self.eps) * mask
        est_pha = torch.atan2(est_im, est_re)

        compressed_speech_stft = torch.stack(
            [
                torch.cos(speech_pha) * compressed_speech_mag,
                torch.sin(speech_pha) * compressed_speech_mag,
            ],
            dim=-1,
        )
        compressed_est_stft = torch.stack(
            [torch.cos(est_pha) * compressed_est_mag, torch.sin(est_pha) * compressed_est_mag],
            dim=-1,
        )

        l_mag = (compressed_speech_mag - compressed_est_mag).pow(2)
        l_pha = (compressed_speech_stft - compressed_est_stft).pow(2).sum(-1)

        loss = (l_mag * (1 + w_echo) + l_pha).mean()
        return loss
