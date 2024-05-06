import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from .ssim import ssim

torchaudio.set_audio_backend("sox_io")


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TimeLoss(nn.Module):
    def __init__(self,
                 hop_length,
                 frame_length,
                 reduction):
        super(TimeLoss, self).__init__()
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.reduction = reduction

    def forward(self, x, y):
        b = x.size(0)
        length = x.size(-1)
        if self.hop_length != 1:
            e1 = [x[:, i : i + self.frame_length].unsqueeze(-1) for i in range(0, length, self.hop_length) if i + self.frame_length < length]
            e2 = [y[:, i : i + self.frame_length].unsqueeze(-1) for i in range(0, length, self.hop_length) if i + self.frame_length < length]
            e1 = torch.cat(e1, dim=-1)
            e2 = torch.cat(e2, dim=-1)
        else:
            e1 = x.unsqueeze(-1)
            e2 = y.unsqueeze(-1)

        e_loss = F.l1_loss((e1 ** 2).mean(1), (e2 ** 2).mean(1), reduction=self.reduction)
        t_loss = F.l1_loss(e1.mean(1), e2.mean(1), reduction= self.reduction)
        if self.hop_length > 1:
            de1 = (e1.mean(1)[:, 1:] - e1.mean(1)[:, :-1]) * ((e2 ** 2).mean(1)[:, :-1] > 1e-2).float() + 1e-8
            de2 = (e2.mean(1)[:, 1:] - e2.mean(1)[:, :-1]) * ((e2 ** 2).mean(1)[:, :-1] > 1e-2).float() + 1e-8
            p_loss = F.l1_loss(de1, de2, reduction=self.reduction)
        else:
            if self.reduction == "none":
                p_loss = torch.zeros_like(e_loss)
            else:
                p_loss = 0

        if self.reduction == "none":
            e_loss = torch.mean(e_loss, dim=1)
            t_loss = torch.mean(t_loss, dim=1)
            p_loss = torch.mean(p_loss, dim=1)

        return e_loss, t_loss, p_loss

class MultiResolutionTimeLoss(nn.Module):
    def __init__(self,
                 hop_sizes=[1, 120, 240, 480],
                 frame_lengths=[1, 240, 480, 960],
                 reduction='mean'
                ):

        super(MultiResolutionTimeLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (hop_size, frame_length) in zip(hop_sizes, frame_lengths):
            self.loss_layers.append(TimeLoss(hop_size, frame_length, reduction))

    def forward(self, fake_signals, true_signals):
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

def var_loss(y, x, hop_size=3):
    b = x.size(0)
    length = x.size(-1)
    x = x.view(b, -1, hop_size) #target
    y = y.view(b, -1, hop_size) #predicted
    r = torch.randperm(hop_size)
    y_r = y[:, :, r]
    loss = -1.0 * F.l1_loss(y, y_r)

    return loss

def stft(x, fft_size, hop_size, win_size, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor (B, T).

    Returns:
        Tensor: Magnitude spectrogram (B, T, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    outputs = torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    outputs = torch.sqrt(outputs)
    return outputs

class SpectralConvergence(nn.Module):
    def __init__(self, reduction):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergence, self).__init__()
        self.reduction = reduction

    def forward(self, predicts_mag, targets_mag):
        if self.reduction == "none":
            x = torch.norm(targets_mag - predicts_mag, p='fro', dim=(1,2))
            y = torch.norm(targets_mag, p='fro', dim=(1,2))
        else:
            x = torch.norm(targets_mag - predicts_mag, p='fro')
            y = torch.norm(targets_mag, p='fro')

        return x / y

class LogSTFTMagnitude(nn.Module):
    def __init__(self, reduction):
        super(LogSTFTMagnitude, self).__init__()
        self.reduction = reduction

    def forward(self, predicts_mag, targets_mag):
        #bias = 0.01
        bias = 0.0
        log_predicts_mag = torch.log(predicts_mag + bias)
        log_targets_mag = torch.log(targets_mag + bias)

        outputs = F.l1_loss(log_predicts_mag, log_targets_mag, reduction=self.reduction)
        if self.reduction == "none":
            outputs = torch.mean(outputs, dim=(1,2))

        return outputs

class STFTLoss(nn.Module):
    def __init__(self,
                 fft_size,
                 hop_size,
                 win_size,
                 reduction,
                 device="cuda",
                 ):
        super(STFTLoss, self).__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(win_size).to(device)
        self.sc_loss = SpectralConvergence(reduction)
        self.mag_loss = LogSTFTMagnitude(reduction)

    def forward(self, predicts, targets):
        """
        Args:
            x: predicted signal (B, T).
            y: truth signal (B, T).

        Returns:
            Tensor: STFT loss values.
        """

        predicts_mag = stft(predicts, self.fft_size, self.hop_size, self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size, self.window)

        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag_loss(predicts_mag, targets_mag)

        return sc_loss, mag_loss

# compute L1 loss of mag and phase
class STFTLoss2(nn.Module):
    def __init__(self,
                 fft_size,
                 hop_size,
                 win_size,
                 reduction,
                 device="cuda",
                 ):
        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(win_size).to(device)

    def _compute_mag_phase(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_size, self.window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1))
        phase = torch.atan2(imag, real)
        return mag, phase

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: signal (B, T).
        Returns:
            Tensor: STFT loss values.
        """
        mag1, phase1 = self._compute_mag_phase(x1)
        mag2, phase2 = self._compute_mag_phase(x2)
        mag_loss = F.l1_loss(mag1, mag2)
        phase_loss = F.l1_loss(phase1, phase2)
        return mag_loss, phase_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=[8196, 4096, 2048, 512, 128, 64, 32],
                 win_sizes=[4096, 2048, 1024, 256, 64, 32, 16],
                 hop_sizes=[2048, 1024, 512, 128, 32, 16, 8],
                 reduction="mean",
                 device="cuda"):
                 #fft_sizes=[2048, 1024, 512],
                 #win_sizes=[1024, 512, 256],
                 #hop_sizes=[512, 256, 128],
        super(MultiResolutionSTFTLoss, self).__init__()

        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size, reduction, device))

    def forward(self, fake_signals, true_signals):
        sc_losses = []
        mag_losses = []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)
        
        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)
        return sc_loss, mag_loss


# compute l1 loss of mag and phase
class MultiResolutionSTFTLoss2(nn.Module):
    def __init__(self,
                 fft_sizes=[8196, 4096, 2048, 512, 128, 64, 32],
                 win_sizes=[4096, 2048, 1024, 256, 64, 32, 16],
                 hop_sizes=[2048, 1024, 512, 128, 32, 16, 8],
                 reduction="mean",
                 device="cuda"):
                 #fft_sizes=[2048, 1024, 512],
                 #win_sizes=[1024, 512, 256],
                 #hop_sizes=[512, 256, 128],
        super().__init__()

        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss2(fft_size, hop_size, win_size, reduction, device))

    def forward(self, fake_signals, true_signals):
        mag_losses = []
        phase_losses = []
        for layer in self.loss_layers:
            mag_loss, phase_loss = layer(fake_signals, true_signals)
            mag_losses.append(mag_loss)
            phase_losses.append(phase_loss)
        mag_loss = sum(mag_losses) / len(mag_losses)
        phase_loss = sum(phase_losses) / len(phase_losses)
        return mag_loss + phase_loss


class MelSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=300, win_length=1024,
                 n_mel_channels=80, sampling_rate=24000, mel_fmin=0.0,
                 mel_fmax=None, device='cuda'):
        super(MelSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length).to(device)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float().to(device)
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        #assert(torch.min(y.data) >= -1)
        #assert(torch.max(y.data) <= 1)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.filter_length-self.hop_length)/2), int((self.filter_length-self.hop_length)/2)), mode='reflect')
        y = y.squeeze(1)
        y_stft = torch.stft(y, self.filter_length, self.hop_length, self.win_length, self.window,
                            center=False, pad_mode='reflect', normalized=False, onesided=True)

        magnitudes, phases = y_stft[..., 0], y_stft[..., 1]
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

class MelLoss(nn.Module):
    def __init__(self,
                 hp,
                 device='cuda'):
        super().__init__()
        self.mel = MelSTFT(hp.n_fft,
                hp.hop_length,
                hp.n_fft,
                hp.n_mels,
                hp.sample_rate,
                device=device)

    def forward(self, predicts, targets):
        """
        Args:
            x: predicted signal (B, T).
            y: truth signal (B, T).

        Returns:
            Tensor: Mel loss values.
        """

        pred = self.mel.mel_spectrogram(predicts)
        target = self.mel.mel_spectrogram(targets)
        mel_loss = F.l1_loss(pred, target)

        return mel_loss

class MaskedSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        C = pred.shape[1]
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        pred = torch.unsqueeze(pred*mask, dim=1) 
        target = torch.unsqueeze(target*mask, dim=1) 
        masked_ssim_loss = 1 - ssim(pred, target, size_average=False)
        reduce_sum = torch.clamp(torch.sum(mask) * C, min=1.0)
        masked_ssim_loss = (masked_ssim_loss * mask).sum() / reduce_sum
        return masked_ssim_loss

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

class MaskedMAELossDim2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask, weight=None):
        # pred, target: [B, C, T]
        # mask: [B, T]
        mae_loss = self.mae(pred, target)
        if weight is not None:
            mae_loss = mae_loss * weight[:, None, None]
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(2)
        masked_mae_loss = torch.sum(mae_loss * mask.unsqueeze(2)) / reduce_sum
        return masked_mae_loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask, weight=None):
        # pred, target: [B, C, T]
        # mask: [B, T]
        loss = self.loss_func(pred, target)
        if weight is not None:
            loss = loss * weight[:, None, None]
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0) * target.size(1)
        masked_loss = torch.sum(loss * mask.unsqueeze(1)) / reduce_sum
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

class MaskedCosLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target, mask):
        cos_loss = 1 - self.cos(pred, target)
        reduce_sum = torch.clamp(torch.sum(mask), min=1.0)
        masked_cos_loss = torch.sum(cos_loss * mask) / reduce_sum
        return masked_cos_loss


class MultiScaleMaskedCELoss(nn.Module):
    def __init(self):
        super().__init__()
        self.ce = MaskedCELoss()

    def forward(self, pred, target, mask, down_rates):
        loss = []
        for down_rate, pred_x in zip(down_rates, predicts):
            targ_x = target.unsqueeze(-1).expand(-1, pred_x.size(-1))
            sub_mask = mask[:, 1::down_rate]
            loss.append(self.ce(pred_x, targ_x, mask))
        loss = sum(loss) / len(loss)
        return loss


if __name__ == "__main__":
    pass

