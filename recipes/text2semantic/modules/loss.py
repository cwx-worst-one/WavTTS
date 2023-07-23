import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return loss


class MaskedMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')

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
        self.mse = nn.MSELoss(reduction='none')

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
        m_1, logs_1, m_2, logs_2 = m_1.contiguous(), logs_1.contiguous(), m_2.contiguous(), logs_2.contiguous()

        b, t, c = m_1.shape
        z_mask = z_mask.unsqueeze(-1)
        z_mask = z_mask.expand(-1, -1, c)

        m_1, logs_1, m_2, logs_2 = m_1 * z_mask, logs_1 * z_mask, m_2 * z_mask, logs_2 * z_mask
        kl = logs_2 - logs_1 \
            + (torch.exp(logs_1)**2 + (m_1 - m_2)**2) / (2.0 * torch.exp(logs_2)**2) \
            - 0.5

        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l



class MaskedTimeDeltaL12(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')
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
