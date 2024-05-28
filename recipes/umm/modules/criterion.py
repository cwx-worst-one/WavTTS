from typing import Dict, Optional, Union

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from samantha.criterion.spectral_loss import (
    MagnitudeSTFTLoss,
    MultiScaleSTFTLoss,
    SpectralConvergengeLoss,
    apply_reduction,
)


class STFTLoss(nn.Module):
    """STFT loss module.
    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).
    """

    def __init__(
        self,
        w_spectral_convergence: float = 1.0,
        w_lin_mag: float = 1.0,
        reduction: str = "mean",
        mag_distance: Optional[str] = "L1",
    ):
        super().__init__()
        self.spec_conv_loss = SpectralConvergengeLoss()
        self.mag_stft_loss = MagnitudeSTFTLoss(
            distance=mag_distance, reduction=reduction
        )
        self.w_spectral_convergence = w_spectral_convergence
        self.w_lin_mag = w_lin_mag
        self.reduction = reduction

    def forward(
        self, mag_x: torch.Tensor, mag_y: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        spec_mag_loss = (
            self.spec_conv_loss(mag_x, mag_y) if self.w_spectral_convergence else 0.0
        )
        lin_mag_loss = self.mag_stft_loss(mag_x, mag_y) if self.w_lin_mag else 0.0

        loss = (self.w_spectral_convergence * spec_mag_loss) + (
            self.w_lin_mag * lin_mag_loss
        )

        loss = apply_reduction(loss, reduction=self.reduction)
        return {
            "stft_loss": loss,
            "spec_mag_loss": spec_mag_loss,
            "lin_mag_loss": lin_mag_loss,
        }


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous().float()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return {"loss": loss.mean()}

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return {"loss": loss}


class MOSTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnnt_loss_fn = torchaudio.transforms.RNNTLoss()
        self.rq_loss_fn = MaskedCrossEntropy()
        self.stft_loss_fn = STFTLoss()

    def forward(
        self,
        rq_logits,
        rq_ids,
        rnnt_logits,
        text_ids,
        rnnt_logit_lengths,
        text_lengths,
        recon_feature,
        feature,
    ):
        rnnt_logits = rnnt_logits.contiguous().float()
        text_ids = text_ids.contiguous()
        recon_feature = recon_feature.contiguous().float()
        feature = feature.contiguous().float()

        # BEST-RQ cross entropy
        loss_dict = self.rq_loss_fn(rq_logits, rq_ids)

        # RNN-T loss
        rnnt_loss = self.rnnt_loss_fn(
            rnnt_logits, text_ids.int(), rnnt_logit_lengths.int(), text_lengths.int()
        )
        loss_dict["rnnt_loss"] = rnnt_loss
        # Spec
        spec_loss = self.stft_loss_fn.float()(recon_feature, feature)
        loss_dict.update(spec_loss)
        return loss_dict


class BestRQMelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rq_loss_fn = MaskedCrossEntropy()
        self.stft_loss_fn = STFTLoss()

    def forward(self, rq_logits, rq_ids, recon_feature, feature):
        recon_feature = recon_feature.contiguous().float()
        feature = feature.contiguous().float()

        # BEST-RQ cross entropy
        loss_dict = self.rq_loss_fn(rq_logits, rq_ids)

        # Spec
        spec_loss = self.stft_loss_fn.float()(recon_feature, feature)
        loss_dict.update(spec_loss)
        return loss_dict


class BestRQMelCTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss()
        self.stft_loss_fn = STFTLoss()

    def forward(self, recon_feature, feature, logits, text_ids):
        recon_feature = recon_feature.contiguous().float()
        feature = feature.contiguous().float()

        # Spec
        loss_dict = self.stft_loss_fn.float()(recon_feature, feature)

        # CTC
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["ctc_loss"] = ctc_loss

        return loss_dict


class Stage2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss()
        self.stft_loss_fn = STFTLoss()
        self.chroma_loss_fn = STFTLoss()

    def forward(
        self, recon_feature, feature, logits, text_ids, recon_chroma=None, chroma=None
    ):
        recon_feature = recon_feature.contiguous().float()
        feature = feature.contiguous().float()

        # Spec
        loss_dict = self.stft_loss_fn.float()(recon_feature, feature)

        if recon_chroma is not None and chroma is not None:
            recon_chroma = recon_chroma.contiguous().float()
            chroma = chroma.contiguous().float()
            chroma_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)
            loss_dict.update(
                chroma_stft_loss=chroma_loss["stft_loss"],
                chroma_spec_mag_loss=loss_dict["spec_mag_loss"],
                chroma_lin_mag_loss=chroma_loss["lin_mag_loss"],
            )

        # CTC
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["ctc_loss"] = ctc_loss

        return loss_dict


class UMMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.mel_loss_fn = STFTLoss()
        if config.add_chroma:
            self.chroma_loss_fn = STFTLoss()
        self.config = config

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, ctc_logits, text_ids, recon_mel, mel, recon_chroma, chroma):
        loss_dict = {}

        # Mel
        recon_mel = recon_mel.contiguous().float()
        mel = mel.contiguous().float()
        mel_loss = self.mel_loss_fn.float()(recon_mel, mel)
        loss_dict["loss_mel"] = mel_loss["stft_loss"]

        # Chroma
        if self.config.add_chroma:
            recon_chroma = recon_chroma.contiguous().float()
            chroma = chroma.contiguous().float()
            chroma_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)
            loss_dict["loss_chroma"] = chroma_loss["stft_loss"]
        if self.config.get("add_ctc", True):
            # CTC
            ctc_logits = ctc_logits.contiguous().float()
            input_lengths = torch.full(
                (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
            )
            labels_mask = text_ids > 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = text_ids.masked_select(labels_mask)

            # CTCLoss doesn't support fp16
            log_probs = F.log_softmax(
                ctc_logits, dim=-1, dtype=torch.float32
            ).transpose(
                0, 1
            )  # [N, T, C] -> [T, N, C]
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = self.ctc_loss_fn(
                    log_probs, flattened_targets, input_lengths, target_lengths
                )
            loss_dict["loss_ctc"] = ctc_loss

        return loss_dict


class UMMLossV2(nn.Module):
    """Refactored version of UMMLoss() for UMM Stage2 training with 2 MSS heads."""

    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.mel_loss_fn = STFTLoss()
        if config.add_chroma:
            self.chroma_loss_fn = STFTLoss()
        self.config = config

    def compute_spectrogram_loss(self, recon_spec, gt_spec, loss_fn):
        """Compute generalized spectrogram loss."""
        recon_spec = recon_spec.contiguous().float()
        gt_spec = gt_spec.contiguous().float()
        loss_spec = loss_fn.float()(recon_spec, gt_spec)
        return loss_spec["stft_loss"]

    def compute_mel_loss(self, recon_mel, mel):
        """Compute mel spectrogram loss."""
        return self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn)

    def compute_chroma_loss(self, recon_chroma, chroma):
        """Compute chroma spectrogram loss."""
        return self.compute_spectrogram_loss(recon_chroma, chroma, self.chroma_loss_fn)

    def compute_ctc_loss(self, ctc_logits, text_ids):
        """Compute CTC loss."""
        ctc_logits = ctc_logits.contiguous().float()
        input_lengths = torch.full(
            (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
        )
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        return ctc_loss

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, ctc_logits, text_ids, recon_mel, mel, recon_chroma, chroma):
        loss_dict = {
            "loss_mel": self.compute_mel_loss(recon_mel, mel),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        return loss_dict


class UMMLossMSS(UMMLossV2):
    """Identical to UMMLoss() but with 2 additional losses for MSS vocal and instrumental mel reconstruction."""

    def __init__(self, config):
        super().__init__(config)
        self.mel_vocal_loss_fn = STFTLoss()
        self.mel_inst_loss_fn = STFTLoss()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_mel,
        mel,
        recon_chroma,
        chroma,
        recon_mel_vocal,
        mel_vocal,
        recon_mel_inst,
        mel_inst,
    ):
        loss_dict = {
            "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn),
            "loss_mel_vocal": self.compute_spectrogram_loss(
                recon_mel_vocal, mel_vocal, self.mel_vocal_loss_fn
            ),
            "loss_mel_inst": self.compute_spectrogram_loss(
                recon_mel_inst, mel_inst, self.mel_inst_loss_fn
            ),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        return loss_dict


class UMMLossPitchSupervised(UMMLossV2):
    """
    Loss for Mel, CTC, Chroma, F0 and VUV.
    Intended for MAY2024 Pitch Loss ConvUMM models.
    """

    # @hanoihantrakul 27MAY2024
    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_mel,
        mel,
        recon_chroma,
        chroma,
        recon_f0,
        f0,
        recon_vuv,
        vuv,
    ):
        loss_dict = {
            "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        if self.config.add_supervised_pitch:
            loss_dict.update(f0_loss=pitch_utils.compute_f0_loss(recon_f0, f0, vuv))
            loss_dict.update(vuv_loss=pitch_utils.compute_vuv_loss(recon_vuv, vuv))
        return loss_dict


class UMMLossPitchPerceptual(UMMLossV2):
    """
    Loss for Mel, CTC, Chroma, Perceptual Pitch Loss (PPL)
    Intended for MAY2024 Pitch Loss ConvUMM models.
    """

    # @hanoihantrakul 28MAY2024
    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_mel,
        mel,
        recon_chroma,
        chroma,
        recon_pitch_h,
        pitch_h,
    ):
        loss_dict = {
            "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        if self.config.add_perceptual_pitch:
            loss_dict.update(
                perceptual_pitch_loss=pitch_utils.compute_perceptual_pitch_loss(
                    recon_pitch_h, pitch_h
                )
            )
        return loss_dict


class UMMLossPitchSupervisedPerceptual(UMMLossV2):
    """
    Loss for Mel, CTC, Chroma, Supervised Pitch Loss (SPL) and Perceptual Pitch Loss (PPL)
    Intended for MAY2024 Pitch Loss ConvUMM models.
    """

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_mel,
        mel,
        recon_chroma,
        chroma,
        recon_f0,
        f0,
        recon_vuv,
        vuv,
        recon_pitch_h,
        pitch_h,
    ):
        loss_dict = {
            "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        if self.config.add_supervised_pitch:
            loss_dict.update(f0_loss=pitch_utils.compute_f0_loss(recon_f0, f0, vuv))
            loss_dict.update(vuv_loss=pitch_utils.compute_vuv_loss(recon_vuv, vuv))
        if self.config.add_perceptual_pitch:
            loss_dict.update(
                perceptual_pitch_loss=pitch_utils.compute_perceptual_pitch_loss(
                    recon_pitch_h, pitch_h
                )
            )
        return loss_dict


class UMMLossMSSPitchSupervised(UMMLossV2):
    """
    Loss for Mel, CTC, Chroma, F0 and VUV.
    Intended for DEC2023 MSS models.
    """

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_mel,
        mel,
        recon_chroma,
        chroma,
        recon_f0,
        f0,
        recon_vuv,
        vuv,
    ):
        loss_dict = {
            "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn),
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids),
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        if self.config.add_pitch:
            loss_dict.update(f0_loss=pitch_utils.compute_f0_loss(recon_f0, f0, vuv))
            loss_dict.update(vuv_loss=pitch_utils.compute_vuv_loss(recon_vuv, vuv))
        return loss_dict


class UMMLossMSSPitchSupervisedPerceptual(UMMLossV2):
    """Loss for CTC, Chroma, F0, VUV and perceptual pitch loss."""

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        ctc_logits,
        text_ids,
        recon_chroma,
        chroma,
        recon_pitch_h,
        pitch_h,
        recon_f0,
        f0,
        recon_vuv,
        vuv,
    ):
        loss_dict = {
            # no mel in this configuration! Replace with perceptual loss
            # "loss_mel": self.compute_spectrogram_loss(recon_mel, mel, self.mel_loss_fn)
            "loss_ctc": self.compute_ctc_loss(ctc_logits, text_ids)
        }
        if self.config.add_chroma:
            loss_dict.update(loss_chroma=self.compute_chroma_loss(recon_chroma, chroma))
        if self.config.add_pitch:
            loss_dict.update(f0_loss=pitch_utils.compute_f0_loss(recon_f0, f0, vuv))
            loss_dict.update(vuv_loss=pitch_utils.compute_vuv_loss(recon_vuv, vuv))
        if self.config.add_perceptual_pitch:
            loss_dict.update(
                perceptual_pitch_loss=pitch_utils.compute_perceptual_pitch_loss(
                    recon_pitch_h, pitch_h
                )
            )
        return loss_dict


class Stage3ARLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.mel_loss_fn = STFTLoss()
        self.chroma_loss_fn = STFTLoss()
        self.ar_loss_fn = MaskedCrossEntropy()

    def forward(
        self, ctc_logits, ctc_ids, mel_out, mel, chroma_out, chroma, ar_logits, ar_ids
    ):
        mel_out = mel_out.contiguous().float()
        mel = mel.contiguous().float()
        chroma_out = chroma_out.contiguous().float()
        chroma = chroma.contiguous().float()

        loss_dict = {}

        # Spec
        mel_loss = self.mel_loss_fn.float()(mel_out, mel)
        chroma_loss = self.chroma_loss_fn.float()(chroma_out, chroma)
        loss_dict["loss_mel"] = mel_loss["stft_loss"]
        loss_dict["loss_chroma"] = chroma_loss["stft_loss"]

        # CTC
        ctc_logits = ctc_logits.contiguous().float()
        input_lengths = torch.full(
            (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
        )
        labels_mask = ctc_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = ctc_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["loss_ctc"] = ctc_loss

        # AR
        ar_loss = self.ar_loss_fn(ar_logits, ar_ids)
        loss_dict["loss_ar"] = ar_loss["loss"]

        return loss_dict


class CTCLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )

    def forward(self, ctc_logits, text_ids):
        loss_dict = {}
        # CTC
        ctc_logits = ctc_logits.contiguous().float()
        input_lengths = torch.full(
            (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
        )
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["loss_ctc"] = loss

        return loss_dict


class VocoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_stft_loss_fn = MultiScaleSTFTLoss(w_phase=1.0)

    def forward(self, recon_wav, wav):
        recon_wav = recon_wav.contiguous().float()
        wav = wav.contiguous().float()
        loss_dict = {}
        # Recon loss
        loss_dict["multi_stft_loss"] = self.multi_stft_loss_fn(recon_wav, wav)
        return loss_dict


class MKIILoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            reduction=config.ctc_loss_reduction, zero_infinity=config.ctc_zero_infinity
        )
        self.mel_loss_fn = STFTLoss()
        self.chroma_loss_fn = STFTLoss()
        self.config = config

    def forward(self, logits, text_ids, recon_chroma, chroma, recon_mel, mel):
        logits = logits.contiguous().float()
        recon_chroma = recon_chroma.contiguous().float()
        chroma = chroma.contiguous().float()
        recon_mel = recon_mel.contiguous().float()
        mel = mel.contiguous().float()
        loss_dict = {}

        chroma_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)
        loss_dict["loss_chroma_stft"] = chroma_loss["stft_loss"]
        loss_dict["loss_chroma_spec_mag"] = chroma_loss["spec_mag_loss"]
        loss_dict["loss_chroma_lin_mag"] = chroma_loss["lin_mag_loss"]

        mel_loss = self.mel_loss_fn.float()(recon_mel, mel)
        loss_dict["loss_mel_stft"] = mel_loss["stft_loss"]
        loss_dict["loss_mel_spec_mag"] = mel_loss["spec_mag_loss"]
        loss_dict["loss_mel_lin_mag"] = mel_loss["lin_mag_loss"]

        # CTC
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["loss_ctc"] = ctc_loss

        return loss_dict


class SoundStormLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, masked_acoustic_tokens, rand_qs, mask_id):
        loss_dict = {}
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        targets = targets[batch_indices, :, rand_qs]
        mask = masked_acoustic_tokens[batch_indices, :, rand_qs] == mask_id
        masked_logits = logits[mask].contiguous().float()
        masked_targets = targets[mask].contiguous()

        accu = (masked_logits.argmax(1) == masked_targets).float().mean() * 100
        log_probs = F.log_softmax(
            masked_logits.float().view(-1, masked_logits.size(-1)), dim=-1
        )
        loss = -torch.gather(log_probs, dim=1, index=masked_targets.view(-1, 1)).mean()

        loss_dict["loss"] = loss
        loss_dict["accu"] = accu

        return loss_dict


class CTCPitchMelLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.mel_loss_fn = STFTLoss()
        self.config = config

    def forward(
        self, ctc_logits, text_ids, recon_mel, mel, recon_f0, f0, recon_vuv, vuv
    ):
        loss_dict = {}
        # Mel
        recon_mel = recon_mel.contiguous().float()
        mel = mel.contiguous().float()
        mel_loss = self.mel_loss_fn.float()(recon_mel, mel)
        loss_dict["loss_mel"] = mel_loss["stft_loss"]

        # F0
        recon_f0 = recon_f0.contiguous().float()
        f0 = f0.contiguous().float()
        f0_loss = (torch.abs(recon_f0 - f0) * vuv).sum() / (torch.sum(vuv) + 1)
        loss_dict["f0_loss"] = f0_loss

        # vuv
        recon_vuv = recon_vuv.contiguous().float()
        vuv = vuv.contiguous().float()
        vuv_loss = F.binary_cross_entropy_with_logits(recon_vuv, vuv)
        loss_dict["vuv_loss"] = vuv_loss

        # CTC
        ctc_logits = ctc_logits.contiguous().float()
        input_lengths = torch.full(
            (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
        )
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        loss_dict["loss_ctc"] = ctc_loss

        return loss_dict


class UMMMergeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.mel_loss_fn = STFTLoss()
        self.chroma_loss_fn = STFTLoss()
        self.las_loss_fn = MaskedCrossEntropy()
        self.config = config

    def forward(
        self,
        ctc_logits=None,
        text_ids=None,
        recon_mel=None,
        mel=None,
        recon_chroma=None,
        chroma=None,
        recon_f0=None,
        f0=None,
        recon_vuv=None,
        vuv=None,
        las_logits=None,
        las_targets=None,
    ):
        loss_dict = {}

        # Mel
        if recon_mel is not None:
            recon_mel = recon_mel.contiguous().float()
            mel = mel.contiguous().float()
            mel_loss = self.mel_loss_fn.float()(recon_mel, mel)
            loss_dict["loss_mel"] = mel_loss["stft_loss"]

        # Chroma
        if recon_chroma is not None:
            recon_chroma = recon_chroma.contiguous().float()
            chroma = chroma.contiguous().float()
            chroma_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)
            loss_dict["loss_chroma"] = chroma_loss["stft_loss"]

        # F0
        if recon_f0 is not None:
            recon_f0 = recon_f0.contiguous().float()
            f0 = f0.contiguous().float()
            f0_loss = (torch.abs(recon_f0 - f0) * vuv).sum() / (torch.sum(vuv) + 1)
            loss_dict["f0_loss"] = f0_loss

        # vuv
        if recon_vuv is not None:
            recon_vuv = recon_vuv.contiguous().float()
            vuv = vuv.contiguous().float()
            vuv_loss = F.binary_cross_entropy_with_logits(recon_vuv, vuv)
            loss_dict["vuv_loss"] = vuv_loss

        # CTC
        if ctc_logits is not None:
            ctc_logits = ctc_logits.contiguous().float()
            input_lengths = torch.full(
                (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
            )
            labels_mask = text_ids > 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = text_ids.masked_select(labels_mask)
            # CTCLoss doesn't support fp16
            log_probs = F.log_softmax(
                ctc_logits, dim=-1, dtype=torch.float32
            ).transpose(
                0, 1
            )  # [N, T, C] -> [T, N, C]

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = self.ctc_loss_fn(
                    log_probs, flattened_targets, input_lengths, target_lengths
                )
            loss_dict["loss_ctc"] = ctc_loss

        # LAS
        if las_logits is not None:
            target_mask = (las_targets > 0).float()
            las_loss = self.las_loss_fn(
                las_logits[:, 0:-1], las_targets[:, 1:], mask=target_mask[:, 1:]
            )
            loss_dict["loss_las"] = las_loss["loss"]
            las_acc = (
                las_logits[:, 0:-1].argmax(dim=2) == las_targets[:, 1:]
            ).float() * target_mask[:, 1:]
            las_acc = las_acc.sum() / target_mask[:, 1:].sum()
            loss_dict["las_acc"] = las_acc
        return loss_dict
