import torch
import torch.nn.functional as F
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

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((dg - 1) ** 2)
        gen_losses.append(l)
        loss += l

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

        predicts_mag = stft_torch(
            predicts, self.fft_size, self.hop_size, self.win_size, self.window
        )
        targets_mag = stft_torch(
            targets, self.fft_size, self.hop_size, self.win_size, self.window
        )

        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag_loss(predicts_mag, targets_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[8192, 4096, 2048, 512, 128, 64, 32],
        win_sizes=[4096, 2048, 1024, 256, 64, 32, 16],
        hop_sizes=[2048, 1024, 512, 128, 32, 16, 8],
    ):
        super().__init__()

        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
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
