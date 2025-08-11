import torch
from torch import nn

from .ssim import SSIM


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

class MaskedSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, pred, target, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        pred = torch.unsqueeze(pred*mask, dim=1) 
        target = torch.unsqueeze(target*mask, dim=1) 
        masked_ssim_loss = 1 - self.ssim(pred, target)  # 越大越好
        return masked_ssim_loss


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

    def forward(self, pred, target, mask, weight=None):
        # pred, target: [B, C, T]
        # mask: [B, T]
        mae_loss = self.mae(pred, target)
        if weight is not None:
            mae_loss = mae_loss * weight[:, None, None]
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



class MaskedMSEWoReductionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
        masked_mse_wo_reduction_loss = self.mse(pred, target) * mask.unsqueeze(1) / reduce_sum
        return masked_mse_wo_reduction_loss


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
