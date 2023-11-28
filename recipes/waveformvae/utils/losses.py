# from Basis-MelGAN

import torch
import logging
import numpy as np
import torch.nn.functional as F
import math
from typing import List
from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    if is_pytorch_17plus:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    spectrum = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
    return spectrum


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[8192, 4096, 2048, 1024, 512, 256, 128, 64], # [2048, 1024, 512],  #
        hop_sizes=[4096, 2048, 1024, 512, 256, 128, 64, 32],   # [240, 120, 50],     #
        win_lengths=[512, 512, 512, 256, 128, 64, 32, 16],     # [1200, 600, 240],   #
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """

        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """

        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


@torch.jit.script
def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]):
    loss: torch.Tensor = torch.tensor([0.0], device=fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logdet, logs_q, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    z_mask = z_mask.float()

    e_q = - logs_q - 0.5
    e_q = e_q * z_mask
    e_q = e_q.sum([1, 2])

    e_p = - 0.0 - 0.5 * ((z_p - 0.0)**2) * 1.0 + 0.0
    e_p = e_p * z_mask
    e_p = e_p.sum([1, 2]) + logdet

    kl = e_q - e_p

    # Average
    kl = torch.sum(kl)
    l = kl / torch.sum(z_mask)
    return l


def kl_wf_loss(z_p, logdet, logs_q, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    z_mask = z_mask.float()

    e_q = - logs_q - 0.5
    e_q = e_q * z_mask
    e_q = e_q.sum([1, 2])

    e_p = - 0.0 - 0.5 * math.exp(-2.0 * math.log(1.0)) * ((z_p - 0.0)**2) * 1.0 + 0.0
    e_p = e_p * z_mask
    e_p = e_p.sum([1, 2]) + logdet

    kl = e_q - e_p

    # Average
    kl = torch.sum(kl)
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
    kl = -logs_1 + (torch.exp(logs_1) ** 2 + m_1 ** 2) / 2.0 - 0.5

    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def kl_loss_old(z_p, logs_q, m_p, logs_p, z_mask):
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
