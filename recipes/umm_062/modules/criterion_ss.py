import torch
import torch.nn.functional as F
from torch import nn
from librosa.filters import mel as librosa_mel_fn

__all__ = [
    'MultiResolutionSTFTLoss', 'MultiResolutionTimeLoss', 'mel_spectrogram',
    'linear_spectrogram', 'discriminator_loss', 'generator_loss', 'feature_loss'
]

mel_basis = {}
hann_window = {}


def feature_loss(fmap_r, fmap_g, dynamic=True, scales=None):
    loss = 0
    if scales is None:
        scales = [1.0] * len(fmap_r)

    fmap_losses = []
    for dr, dg, scale in zip(fmap_r, fmap_g, scales):
        for rl, gl in zip(dr, dg):
            loss_tmp = torch.mean(torch.abs(rl - gl))
            if dynamic:
                loss_tmp = loss_tmp / rl.abs().mean()
            loss += loss_tmp * scale
            fmap_losses.append(loss_tmp.item())
    if dynamic:
        return loss, fmap_losses
    else:
        return loss * 2, fmap_losses


def discriminator_loss(disc_real_outputs, disc_generated_outputs, scales=None):
    loss = 0
    if scales is None:
        scales = [1.0] * len(disc_real_outputs)
    r_losses = []
    g_losses = []
    for dr, dg, scale in zip(disc_real_outputs, disc_generated_outputs, scales):
        r_loss = torch.mean((dr - 1)**2)
        g_loss = torch.mean((dg - 0)**2)
        loss += (r_loss + g_loss) * scale
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs, scales=None):
    loss = 0
    if scales is None:
        scales = [1.0] * len(disc_outputs)
    gen_losses = []
    for dg, scale in zip(disc_outputs, scales):
        l = torch.mean((dg - 1)**2)
        gen_losses.append(l)
        loss += l * scale

    return loss, gen_losses


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y,
                    n_fft,
                    num_mels,
                    sampling_rate,
                    hop_size,
                    win_size,
                    fmin,
                    fmax,
                    center=False):
    global mel_basis, hann_window
    name = "sr_{}_nfft_{}_win_{}_hop_{}_nmels_{}_fmin_{}_fmax_{}_device_{}_ver1".format(
        sampling_rate, n_fft, win_size, hop_size, num_mels, fmin, fmax,
        y.device)
    if name not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[name] = torch.from_numpy(mel).float().to(y.device)
        hann_window[name] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect').squeeze(1)

    spec = torch.stft(y,
                      n_fft,
                      hop_length=hop_size,
                      win_length=win_size,
                      window=hann_window[name],
                      center=center,
                      pad_mode='reflect',
                      normalized=False,
                      onesided=True,
                      return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  # linear spec
    mel = torch.matmul(mel_basis[name], spec)

    return mel


def linear_spectrogram(y,
                       n_fft,
                       num_mels,
                       sampling_rate,
                       hop_size,
                       win_size,
                       fmin,
                       fmax,
                       center=False):
    global mel_basis, hann_window
    name = "sr_{}_nfft_{}_win_{}_hop_{}_nmels_{}_fmin_{}_fmax_{}_device_{}_ver1".format(
        sampling_rate, n_fft, win_size, hop_size, num_mels, fmin, fmax,
        y.device)
    if name not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[name] = torch.from_numpy(mel).float().to(y.device)
        hann_window[name] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int(
        (n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y,
                      n_fft,
                      hop_length=hop_size,
                      win_length=win_size,
                      window=hann_window[name],
                      center=center,
                      pad_mode='reflect',
                      normalized=False,
                      onesided=True,
                      return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = spectral_normalize_torch(spec)
    return spec


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
    x_stft = torch.stft(x,
                        fft_size,
                        hop_size,
                        win_length,
                        window,
                        return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class TimeLoss(torch.nn.Module):

    def __init__(self, hop_length=240, frame_length=480):
        super().__init__()
        self.hop_length = hop_length
        self.frame_length = frame_length

    def forward(self, x, y):
        b = x.size(0)
        length = x.size(-1)
        if self.hop_length != 1:
            e1 = [
                x[:, i:i + self.frame_length].unsqueeze(-1)
                for i in range(0, length, self.hop_length)
                if i + self.frame_length < length
            ]
            e2 = [
                y[:, i:i + self.frame_length].unsqueeze(-1)
                for i in range(0, length, self.hop_length)
                if i + self.frame_length < length
            ]
            e1 = torch.cat(e1, dim=-1)
            e2 = torch.cat(e2, dim=-1)
        else:
            e1 = x.unsqueeze(-1)
            e2 = y.unsqueeze(-1)

        e_loss = F.l1_loss((e1**2).mean(1), (e2**2).mean(1))
        t_loss = F.l1_loss(e1.mean(1), e2.mean(1))
        if self.hop_length > 1:
            de1 = (e1.mean(1)[:, 1:] - e1.mean(1)[:, :-1]) * (
                (e2**2).mean(1)[:, :-1] > 1e-2).float() + 1e-8
            de2 = (e2.mean(1)[:, 1:] - e2.mean(1)[:, :-1]) * (
                (e2**2).mean(1)[:, :-1] > 1e-2).float() + 1e-8
            p_loss = F.l1_loss(de1, de2)
        else:
            p_loss = 0

        return e_loss, t_loss, p_loss


class MultiResolutionTimeLoss(torch.nn.Module):

    def __init__(self,
                 hop_sizes=[1, 120, 240, 480],
                 frame_lengths=[1, 240, 480, 960]):

        super().__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (hop_size, frame_length) in zip(hop_sizes, frame_lengths):
            self.loss_layers.append(TimeLoss(hop_size, frame_length))

    def forward(self, true_signals, fake_signals):
        e_losses = []
        t_losses = []
        p_losses = []
        for layer in self.loss_layers:
            e_loss, t_loss, p_loss = layer(fake_signals, true_signals)
            e_losses.append(e_loss)
            t_losses.append(t_loss)
            p_losses.append(p_loss)

        e_loss = sum(e_losses) / len(e_losses)
        t_loss = sum(t_losses) / len(t_losses)
        p_loss = sum(p_losses) / len(p_losses)

        return e_loss, t_loss, p_loss


class SpectralConvergence(torch.nn.Module):

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p='fro')
        y = torch.norm(targets_mag, p='fro').clamp(1)

        return x / y

        x = F.mse_loss(predicts_mag, targets_mag)
        return x


class LogSTFTMagnitude(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        bias = 1e-5
        log_predicts_mag = torch.log(predicts_mag + bias)
        log_targets_mag = torch.log(targets_mag + bias)

        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs


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

        predicts_mag = stft_torch(predicts, self.fft_size, self.hop_size,
                                  self.win_size, self.window)
        targets_mag = stft_torch(targets, self.fft_size, self.hop_size,
                                 self.win_size, self.window)

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
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes,
                                                  hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))

    def forward(self, true_signals, fake_signals):
        true_signals = true_signals.squeeze(1)
        fake_signals = fake_signals.squeeze(1)
        sc_losses = []
        mag_losses = []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)

        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return sc_loss, mag_loss
