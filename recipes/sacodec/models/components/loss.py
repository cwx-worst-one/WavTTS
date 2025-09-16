import typing
from typing import List, Tuple
import torch
from torch import nn
import torchaudio
import numpy as np
from recipes.umm.transforms.chroma import ChromaSpectrogram


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.
    """
    return torch.log(torch.clip(x, min=clip_val))
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())
def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)

class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    Note (AS) - APCodec ads l2 loss
    """

    def __init__(
        self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
        loss_type="l1" # l1,l2,l1l2
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )
        self.loss_type = loss_type

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = 0
        if "l1" in self.loss_type:
            loss += torch.nn.functional.l1_loss(mel, mel_hat)
        if "l2" in self.loss_type:
            loss += torch.nn.functional.mse_loss(mel, mel_hat)
        return loss


class DescriptMelSpectrogramLoss(nn.Module):
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

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        mel_fmins: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmaxes: List[float] = [None, None, None, None, None, None, None, None],
        window_type: str = None,
    ):
        super().__init__()
    
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight

        mel_transforms =  nn.ModuleList([])

        for n_mel, window_length, mel_fmin, mel_fmax in zip(
                n_mels, 
                window_lengths,
                mel_fmins,
                mel_fmaxes
            ):
            mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate, 
                    n_fft=window_length, 
                    hop_length=window_length//4, 
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

class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

class ChromaLoss(nn.Module):
    def __init__(
        self, sample_rate: int = 24000, n_fft: int = 2048, hop_length: int = 240, n_chroma: int = 12,
        win_length: int = None
    ):
        super().__init__()
        if win_length is None: win_length = n_fft
        self.chroma_transform = ChromaSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_chroma=n_chroma,
            normalized=False,
        )


    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: mse loss between chroma.
        """
        chroma_pred = self.chroma_transform(y_hat)
        chroma_gt = self.chroma_transform(y)

        def normalize_chroma(chroma: torch.Tensor):
            B, A_C, CH, T = chroma.shape
            chroma = chroma.reshape(B * A_C, CH, T) #  bs, a_c, ch, l -> bs*ac, ch x l
            chroma = chroma[:, :, :-1].transpose(1, 2) # bs x ch x l -> bs x l x ch
            return torch.nn.functional.normalize(chroma, p=2, dim=-1, eps=1e-7)
        loss = torch.nn.functional.mse_loss(normalize_chroma(chroma_pred), normalize_chroma(chroma_gt))

        return loss



# APCodec
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

    return loss, gen_losses


def phase_loss_channel(phase_r, phase_g, n_fft, frames):
    assert len(phase_r.shape) == 4, "Invalid dim. Needs to be B, C, N, T"
    # N = n_fft//2+1, T = frames
    B, C, N, T = phase_r.shape
    phase_r = phase_r.reshape(B * C, N, T)
    phase_g = phase_g.reshape(B * C, N, T)
    IP_loss, GD_loss, PTD_loss = phase_loss(phase_r, phase_g, n_fft, frames)
    return IP_loss, GD_loss, PTD_loss

def phase_loss(phase_r, phase_g, n_fft, frames):

    GD_matrix = torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=1)-torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=2)-torch.eye(n_fft//2+1)
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0,2,1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0,2,1), GD_matrix)

    PTD_matrix = torch.triu(torch.ones(frames,frames),diagonal=1)-torch.triu(torch.ones(frames,frames),diagonal=2)-torch.eye(frames)
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r-GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r-PTD_g))


    return IP_loss, GD_loss, PTD_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def amplitude_loss(log_amplitude_r, log_amplitude_g):

    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss

def STFT_consistency_loss(rea_r, rea_g, imag_r, imag_g):
    C_loss=torch.mean(torch.mean((rea_r-rea_g)**2+(imag_r-imag_g)**2,(1,2)))
    return C_loss



## Custom wrapper losses
def melspec_stereo_loss(audio_hat, audio_input, melspec_loss_fn):
    # Taken from Soundstream recipes
    wavs_g = audio_hat
    gt_audio = audio_input

    # diff and sum representation
    sum_wavs_g = wavs_g[:, 0, :] + wavs_g[:, 1, :]
    sum_gt_audio = gt_audio[:, 0, :] + gt_audio[:, 1, :]

    diff_wavs_g = wavs_g[:, 0, :] - wavs_g[:, 1, :]
    diff_gt_audio = gt_audio[:, 0, :] - gt_audio[:, 1, :]

    sum_mel_loss = melspec_loss_fn(sum_wavs_g, sum_gt_audio)
    dif_mel_loss = melspec_loss_fn(diff_wavs_g, diff_gt_audio)
    
    mel_loss = melspec_loss_fn(audio_hat, audio_input)


    mel_loss = 0.5 * sum_mel_loss + 0.5 * dif_mel_loss + mel_loss
    return mel_loss

## CTC Loss
class CTCLoss(nn.Module):
    def __init__(self, blank, reduction, zero_infinity):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(self, ctc_logits, text_ids):
        # CTC
        ctc_logits = ctc_logits.contiguous().float()
        input_lengths = torch.full(
            (ctc_logits.size(0),), ctc_logits.size(1), dtype=torch.long
        )
        labels_mask = text_ids > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = text_ids.masked_select(labels_mask)

        # CTCLoss doesn't support fp16
        log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(
            0, 1
        )  # [N, T, C] -> [T, N, C]

        with torch.backends.cudnn.flags(enabled=False):
            loss = self.ctc_loss_fn(
                log_probs, flattened_targets, input_lengths, target_lengths
            )
        return loss
    

# SDR losses

def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses

class SNRLoss(torch.nn.Module):
    """Signal-to-noise ratio loss module.

    Note that this does NOT implement the SDR from
    [Vincent et al., 2006](https://ieeexplore.ieee.org/document/1643671),
    which includes the application of a 512-tap FIR filter.
    
    https://github.com/csteinmetz1/auraloss/blob/main/auraloss/time.py
    """

    def __init__(self, zero_mean=True, eps=1e-7, reduction="mean"):
        super(SNRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        res = input - target
        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses

class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Note that this returns the negative of the SI-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-7 for mixed precision
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.

    https://github.com/csteinmetz1/auraloss/blob/main/auraloss/time.py
    """

    def __init__(self, zero_mean=True, eps=1e-7, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses