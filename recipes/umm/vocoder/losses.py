import torch
import math

from torch.nn import functional as F


def kl_loss_pp(m_1, logs_1, m_2, logs_2, z_mask):
    m_1, logs_1, m_2, logs_2 = m_1 * z_mask, logs_1 * z_mask, m_2 * z_mask, logs_2 * z_mask
    kl = logs_2 - logs_1 \
        + (torch.exp(logs_1)**2 + (m_1 - m_2)**2) / (2.0 * torch.exp(logs_2)**2) \
        - 0.5

    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l


def kl_loss_standard(m_1, logs_1, z_mask):
    m_1, logs_1 = m_1 * z_mask, logs_1 * z_mask
    kl = -logs_1 + (torch.exp(logs_1)**2 + m_1**2) / 2.0 - 0.5

    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l


def kl_loss_vits(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l
