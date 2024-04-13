import typing
from typing import List

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


def feature_loss(fmap_r, fmap_g, dynamic=True):
    loss = 0
    fmap_losses = []
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss_tmp = torch.mean(torch.abs(rl - gl))
            if dynamic:
                loss_tmp = loss_tmp / rl.abs().mean()
            loss += loss_tmp
            fmap_losses.append(loss_tmp.item())
    # divided by number of discriminator and layers
    # loss /= len(fmap_losses)
    if dynamic:
        return loss, fmap_losses
    return loss * 2, fmap_losses


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1) ** 2)
        g_loss = torch.mean((dg - 0) ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    # normalize by the number of discriminator
    # loss /= len(disc_real_outputs)

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((dg - 1) ** 2)
        gen_losses.append(l)
        loss += l
    # normalize by the number of discriminator
    # loss /= len(disc_outputs)

    return loss, gen_losses


class SpectralConvergence(torch.nn.Module):
    def __init__(self):
        """Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p="fro")
        y = torch.norm(targets_mag, p="fro").clamp(1)

        return x / y


class LogSTFTMagnitude(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        bias = 1e-5
        log_predicts_mag = torch.log(predicts_mag + bias)
        log_targets_mag = torch.log(targets_mag + bias)

        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs


def stft_torch(x, fft_size, hop_size, win_length, window):
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
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class STFTLoss(torch.nn.Module):
    def __init__(self, fft_size, hop_size, win_size):
        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        window = torch.hann_window(win_size)
        self.register_buffer("window", window)
        self.sc_loss = SpectralConvergence()
        self.mag_loss = LogSTFTMagnitude()

    def forward(self, predicts, targets):
        """
        Args:
            x: predicted signal (B, T).
            y: truth signal (B, T).

        Returns:
            Tensor: STFT loss values.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            predicts_mag = stft_torch(
                predicts, self.fft_size, self.hop_size, self.win_size, self.window
            )
            targets_mag = stft_torch(
                targets, self.fft_size, self.hop_size, self.win_size, self.window
            )

            sc_loss = self.sc_loss(predicts_mag, targets_mag)
            mag_loss = self.mag_loss(predicts_mag, targets_mag)
        sc_loss, mag_loss = sc_loss.to(predicts.dtype), mag_loss.to(predicts.dtype)

        return sc_loss, mag_loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/
    blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        mel_fmins: List[float] = [0.0, 0.0],
        mel_fmaxes: List[float] = [None, None],
        window_type: str = None,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight

        mel_transforms = nn.ModuleList([])

        for n_mel, window_length, mel_fmin, mel_fmax in zip(
            n_mels, window_lengths, mel_fmins, mel_fmaxes
        ):
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_length,
                hop_length=window_length // 4,
                f_min=mel_fmin,
                f_max=mel_fmax,
                n_mels=n_mel,
                window_fn=torch.hann_window,
                power=pow,
            )
            mel_transforms.append(mel_transform)
        self.mel_transforms = mel_transforms

    def forward(self, pred_signals, ref_signals):
        """Computes mel loss between an estimate and a reference signal.

        Parameters
        ----------
        pred_signals
            Predicted signal
        ref_signals
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for mel_transform in self.mel_transforms:

            ref_mels = mel_transform(ref_signals)
            pred_mels = mel_transform(pred_signals)

            loss += self.log_weight * self.loss_fn(
                pred_mels.clamp(self.clamp_eps).log10(),
                ref_mels.clamp(self.clamp_eps).log10(),
            )
            loss += self.mag_weight * self.loss_fn(pred_mels, ref_mels)
        return loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[8192, 4096, 2048, 512, 128, 64, 32],
        win_sizes=[4096, 2048, 1024, 256, 64, 32, 16],
        hop_sizes=[2048, 1024, 512, 128, 32, 16, 8],
    ):
        super().__init__()

        self.loss_layers = torch.nn.ModuleList()
        for fft_size, win_size, hop_size in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))
        self.num_settings = len(fft_sizes)

    def forward(self, true_signals, fake_signals):
        true_signals = true_signals.squeeze(1)
        fake_signals = fake_signals.squeeze(1)
        sc_losses = 0
        mag_losses = 0
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses += sc_loss
            mag_losses += mag_loss

        sc_loss = sc_losses / self.num_settings
        mag_loss = mag_losses / self.num_settings

        return sc_loss, mag_loss
