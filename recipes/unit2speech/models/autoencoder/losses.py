import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import nn

__all__ = [
    'InfoNCELoss', 'TimeLoss', 'MultiResolutionSTFTLoss',
    'MultiResolutionTimeLoss', 'feature_loss', 'discriminator_loss',
    'generator_loss', 'mel_loss', 'mel_spectrogram', 'linear_spectrogram'
]

mel_basis = {}
hann_window = {}


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """
    def __init__(self,
                 temperature=0.1,
                 reduction='mean',
                 negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query,
                        positive_key,
                        negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query,
             positive_key,
             negative_keys=None,
             temperature=0.1,
             reduction='mean',
             negative_mode='unpaired'):
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def transpose(x):
        return x.transpose(-2, -1)

    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError(
            '<query> and <positive_key> must must have the same number of samples.'
        )
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            'Vectors of <query> and <positive_key> should have the same number of components.'
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                'Vectors of <query> and <negative_keys> should have the same number of components.'
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key,
                                                   negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits),
                             dtype=torch.long,
                             device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


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
        r_loss = torch.mean((dr - 1)**2)
        g_loss = torch.mean((dg - 0)**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((dg - 1)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def mel_loss(wavs_real, wavs_fake, hp):
    y_mel = mel_spectrogram(wavs_real.squeeze(1), hp.n_fft, hp.num_mels,
                            hp.sample_rate, hp.hop_size, hp.win_size, hp.fmin,
                            hp.fmax)
    y_g_hat_mel = mel_spectrogram(wavs_fake.squeeze(1), hp.n_fft, hp.num_mels,
                                  hp.sample_rate, hp.hop_size, hp.win_size,
                                  hp.fmin, hp.fmax)
    return F.l1_loss(y_mel, y_g_hat_mel)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
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
    #if torch.min(y) < -1.:
    #    print('min value is ', torch.min(y))
    #if torch.max(y) > 1.:
    #    print('max value is ', torch.max(y))

    global mel_basis, hann_window
    name = "sr_{}_nfft_{}_win_{}_hop_{}_fmin_{}_fmax_{}_device_{}".format(sampling_rate, n_fft, win_size, hop_size, fmin, fmax, y.device)
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
                      #window=hann_window[str(fmax) + '_' + str(y.device)],
                      window=hann_window[name],
                      center=center,
                      pad_mode='reflect',
                      normalized=False,
                      onesided=True)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  # linear spec
    mel = torch.matmul(mel_basis[name], spec)
    #spec = spectral_normalize_torch(spec)
    #mel = spectral_normalize_torch(mel)
    return_spec = False
    if return_spec:
        return spec
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
    '''
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    '''
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' +
                  str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(fmax) + '_' +
                    str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int(
        (n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y,
                      n_fft,
                      hop_length=hop_size,
                      win_length=win_size,
                      window=hann_window[str(fmax) + '_' + str(y.device)],
                      center=center,
                      pad_mode='reflect',
                      normalized=False,
                      onesided=True)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  # linear spec
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
        fft_sizes=[8196, 4096, 2048, 512, 128, 64, 32],
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
