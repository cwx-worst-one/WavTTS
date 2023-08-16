from typing import Dict, Optional, Union

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

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
            return {"rq_loss": loss.mean()}

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return {"rq_loss": loss}


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


class UMMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss()
        self.stft_loss_fn = STFTLoss()
        self.multi_stft_loss_fn = MultiScaleSTFTLoss(w_phase=1.0)

    def forward(
        self,
        recon_feature,
        feature,
        logits,
        text_ids,
        recon_wav,
        wav,
        vq_embeds=None,
        mulan_embeds=None,
    ):
        recon_feature = recon_feature.contiguous().float()
        feature = feature.contiguous().float()
        recon_wav = recon_wav.contiguous().float()
        wav = wav.contiguous().float()

        # Recon loss
        loss_dict = self.stft_loss_fn.float()(recon_feature, feature)
        loss_dict["multi_stft_loss"] = self.multi_stft_loss_fn(recon_wav, wav)

        # CTC
        if logits is not None and text_ids is not None:
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long
            )
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

        # MuLan regression loss
        if vq_embeds is not None and mulan_embeds is not None:
            vq_embeds = vq_embeds.contiguous().float()
            mulan_embeds = mulan_embeds.contiguous().float()
            loss_dict["mulan_loss"] = F.cosine_similarity(vq_embeds, mulan_embeds)
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
