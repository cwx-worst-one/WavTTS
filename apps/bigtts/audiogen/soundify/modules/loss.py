import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def sequence_mask(seq_lens, max_len=None, dtype=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    mask = mask.to(dtype=dtype)
    return mask


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask, weight=None):
        # pred, target: [B, C, T]
        # mask: [B, T]
        mae_loss = self.mae(pred, target)
        if weight is not None:
            mae_loss = mae_loss * weight[:, None, None]
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
        masked_mae_loss = torch.sum(mae_loss * mask.unsqueeze(1)) / reduce_sum
        return masked_mae_loss


class MaskedMAELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask):

        mae_loss = self.mae(pred, target)
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
        masked_mae_loss = torch.sum(mae_loss * mask.unsqueeze(1)) / reduce_sum

        return masked_mae_loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None):

        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
        masked_mse_loss = torch.sum(self.mse(pred, target) * mask.unsqueeze(1)) / reduce_sum
        return masked_mse_loss


class MAELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, audio_feat, video_feat):

        seq_len = audio_feat.shape[1]
        seq_dim = audio_feat.shape[2]

        clip_len = int(seq_len // 2)

        start_idx1 = np.random.randint(0, seq_len - clip_len)
        end_idx1 = start_idx1 + clip_len

        start_idx2 = np.random.randint(0, seq_len - clip_len)
        end_idx2 = start_idx2 + clip_len

        audio_feat = audio_feat[:, start_idx1:end_idx1, :]
        video_feat1 = video_feat[:, start_idx1:end_idx1, :]
        video_feat2 = video_feat[:, start_idx2:end_idx2, :]

        mae_loss = (self.mae(audio_feat, video_feat1) - self.mae(audio_feat, video_feat2)) * seq_dim

        return mae_loss


class SIMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        cosine_sim = F.cosine_similarity(pred, target, dim=-1)

        cosine_sim = torch.mean(cosine_sim)

        return 1.0 - cosine_sim


class MaskedCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target, mask):
        # pred, target: [B, C, T]
        # mask: [B, T]
        ce_loss = self.ce(pred, target)
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0)
        masked_ce_loss = torch.sum(ce_loss * mask) / reduce_sum
        return masked_ce_loss


class MaskedBCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.bce(pred, target)
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0)
        masked_loss = torch.sum(loss * mask) / reduce_sum
        return masked_loss


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, audio_embed, video_embed, logit_scale):

        batch_size, device = audio_embed.shape[0], audio_embed.device

        logit_scale = logit_scale.mean()
        logits_per_image = logit_scale * audio_embed @ video_embed.t()
        logits_per_text = logits_per_image.t()

        # batch_size = audio_embed.shape[0]
        # labels = torch.arange(batch_size, device=device).long()
        # extra_contrast_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        # TODO(mashengtao) use real label
        labels_audio = torch.zeros(audio_embed.shape[0], device=device).long()
        labels_video = torch.zeros(video_embed.shape[0], device=device).long()
        extra_contrast_loss = (F.cross_entropy(logits_per_image, labels_audio) +
                               F.cross_entropy(logits_per_text, labels_video)) / 2

        return extra_contrast_loss


class SIMLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse = torch.nn.MSELoss(reduction="none")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def diff(self, audio_feat, video_feat):

        diff = self.mse(audio_feat, video_feat)

        diff = torch.mean(diff, dim=[1, 2], keepdim=False)

        return diff

    def forward(self, audio_feat, video_feat):

        bts, seq_len = audio_feat.shape[0:2]

        clip_len = int(seq_len // 2)

        start_idx1 = np.random.randint(0, seq_len - clip_len)
        end_idx1 = start_idx1 + clip_len

        start_idx2 = np.random.randint(0, seq_len - clip_len)
        end_idx2 = start_idx2 + clip_len

        audio_feat = audio_feat[:, start_idx1:end_idx1, :]
        pos_video = video_feat[:, start_idx1:end_idx1, :]
        neg_video = video_feat[:, start_idx2:end_idx2, :]

        logit_scale = torch.clamp(self.logit_scale.exp(), min=1, max=100)

        diff1 = self.diff(audio_feat=audio_feat, video_feat=pos_video) * logit_scale
        diff2 = self.diff(audio_feat=audio_feat, video_feat=neg_video) * logit_scale

        logits = torch.stack([diff1, diff2], dim=1)

        labels = torch.ones(bts).to(device=logits.device).to(dtype=torch.long)

        loss = F.cross_entropy(logits, labels)

        return loss
