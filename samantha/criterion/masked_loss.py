import torch
import torch.nn as nn

from samantha.criterion.ssim import SSIM


class PseudoHuberLoss(nn.Module):
    """The Pseudo-Huber loss."""

    reductions = {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}

    def __init__(self, beta=1, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def extra_repr(self):
        return f"beta={self.beta:g}, reduction={self.reduction!r}"

    def forward(self, input, target):
        output = self.beta**2 * input.sub(target).div(self.beta).pow(2).add(
            1
        ).sqrt().sub(1)
        return self.reductions[self.reduction](output)


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None, log_softmax=True):
        logits = logits.contiguous()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        if log_softmax:
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            log_probs = logits
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return loss


class MaskedCrossEntropyV2(nn.Module):
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
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)).mean()
        return loss


class MaskedMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous()
        targets = targets.contiguous()

        b, t, c = targets.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, c)

        loss_reduce_v = torch.clamp(torch.sum(mask), min=1.0)

        masked_mae_loss = torch.sum(self.mae(logits, targets) * mask) / loss_reduce_v
        return masked_mae_loss


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous()
        targets = targets.contiguous()

        b, t, c = targets.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, c)

        loss_reduce_v = torch.clamp(torch.sum(mask), min=1.0)
        masked_mse_loss = torch.sum(self.mse(logits, targets) * mask) / loss_reduce_v
        return masked_mse_loss


class MaskedKLLossPP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m_1, logs_1, m_2, logs_2, z_mask):
        m_1, logs_1, m_2, logs_2 = (
            m_1.contiguous(),
            logs_1.contiguous(),
            m_2.contiguous(),
            logs_2.contiguous(),
        )

        b, t, c = m_1.shape
        z_mask = z_mask.unsqueeze(-1)
        z_mask = z_mask.expand(-1, -1, c)

        m_1, logs_1, m_2, logs_2 = (
            m_1 * z_mask,
            logs_1 * z_mask,
            m_2 * z_mask,
            logs_2 * z_mask,
        )
        kl = (
            logs_2
            - logs_1
            + (torch.exp(logs_1) ** 2 + (m_1 - m_2) ** 2)
            / (2.0 * torch.exp(logs_2) ** 2)
            - 0.5
        )

        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l


class MaskedTimeDeltaL12(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.l2 = nn.MSELoss(reduction="none")
        self.K = K

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous()
        targets = targets.contiguous()

        b, t, c = targets.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, c)

        loss = 0.0
        loss_list = []

        for k in range(self.K):
            loss_reduce_v = torch.clamp(torch.sum(mask), min=1.0)
            masked_l1_loss = torch.sum(self.l1(logits, targets) * mask) / loss_reduce_v
            masked_l2_loss = torch.sum(self.l2(logits, targets) * mask) / loss_reduce_v
            loss += masked_l1_loss + masked_l2_loss
            loss_list.append(masked_l1_loss + masked_l2_loss)

            logits = logits[:, 1:, :] - logits[:, :-1, :]
            targets = targets[:, 1:, :] - targets[:, :-1, :]
            mask = mask[:, 1:, :]

        return loss, loss_list[0], loss_list[1], loss_list[2]


def sequence_mask(seq_lens, max_len=None, device="cpu"):
    # b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


class MaskedSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(size_average=False)

    def forward(self, pred, target, mask, weight=None):
        C = pred.shape[1]
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        pred = torch.unsqueeze(pred * mask, dim=1)
        target = torch.unsqueeze(target * mask, dim=1)
        # masked_ssim_loss = 1 - self.ssim(pred, target)  # 越大越好
        masked_ssim_loss = 1 - self.ssim(pred, target)
        reduce_sum = torch.clamp(torch.sum(mask) * C, min=1.0)
        masked_ssim_loss = (masked_ssim_loss * mask).sum() / reduce_sum
        return masked_ssim_loss


# class MaskedL1Loss(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.mae = nn.L1Loss(reduction="none")
#
#    def forward(self, pred, target, mask, weight=None):
#        # pred, target: [B, C, T]
#        # mask: [B, T]
#        mae_loss = self.mae(pred, target)
#        if weight is not None:
#            mae_loss = mae_loss * weight[:, None, None]
#        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
#        masked_mae_loss = torch.sum(mae_loss * mask.unsqueeze(1)) / reduce_sum
#        return masked_mae_loss


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
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask=None, weight=None):
        if weight is not None:
            reduce_sum = torch.sum(mask, 1) * target.size(1)
            B = pred.shape[0]
            batch_loss = (
                torch.sum(self.mse(pred, target) * mask.unsqueeze(1), (1, 2))
                / reduce_sum
            )
            masked_mse_loss = torch.sum(batch_loss * weight) / B
        else:
            reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
            masked_mse_loss = (
                torch.sum(self.mse(pred, target) * mask.unsqueeze(1)) / reduce_sum
            )
        return masked_mse_loss


class MaskedPseudoHuberLoss(nn.Module):
    def __init__(self, c=0.1):
        super().__init__()
        self.func = PseudoHuberLoss(c, reduction="none")

    def forward(self, pred, target, mask=None, weight=None):
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)

        if weight is not None:  # not supported
            masked_loss = (
                torch.sum(self.func(pred, target) * mask.unsqueeze(1) * weight)
                / reduce_sum
            )
        else:
            masked_loss = (
                torch.sum(self.func(pred, target) * mask.unsqueeze(1)) / reduce_sum
            )
        return masked_loss


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
