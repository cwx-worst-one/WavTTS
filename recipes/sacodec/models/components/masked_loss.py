
import typing
from typing import List, Tuple

import torch
import torchaudio
from torch import nn
import numpy as np

from einops import rearrange, repeat
from recipes.umm.transforms.chroma import ChromaSpectrogram

class MaskedDescriptMelSpectrogramLoss(nn.Module):
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
        loss_fn: typing.Callable = None,
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

    def forward(self, pred_signals, ref_signals, keep_mask=None):
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
            pred_mels = mel_transform(pred_signals) # bs, a_c, n_mels, t

            # Masked loss takes bs x t x ch as input
            B, A_C, CH, T = pred_mels.shape
            ref_mels = ref_mels.reshape(B * A_C, CH, T).transpose(1, 2) #  bs, a_c, n_mels, t -> bs*ac, t, ch
            pred_mels = pred_mels.reshape(B * A_C, CH, T).transpose(1, 2) #  bs, a_c, n_mels, t -> bs*ac, t, ch


            if keep_mask is not None:
                # print('Keep mask before', keep_mask.shape)
                keep_mask_reshape = keep_mask.repeat(A_C, 1, 1) # bs x 1 x L -> bs*ac x 1 x L
                # print('Keep mask shape', A_C, keep_mask_reshape.shape, ref_mels.shape)
                loss_mask = torch.nn.functional.interpolate(keep_mask_reshape.float(), size=T)
                loss_mask = loss_mask.squeeze(1) # bs*ac x 1 x L -> bs*ac x L. Required dimensions masked loss
                # print('loss mask shape', loss_mask.shape)
            else:
                loss_mask = None

            loss += self.log_weight * self.loss_fn(
                pred_mels.clamp(self.clamp_eps).log10(),
                ref_mels.clamp(self.clamp_eps).log10(),
                loss_mask
            )
            loss += self.mag_weight * self.loss_fn(pred_mels, ref_mels, loss_mask)
        return loss

class MaskedChromaLoss(nn.Module):
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
        self.loss_func = MaskedMSE()


    def forward(self, y_hat, y, keep_mask=None) -> torch.Tensor:
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
            chroma = chroma[:, :, :-1].transpose(1, 2) # bs x a_ch, ch x l -> bs x l x ch
            return torch.nn.functional.normalize(chroma, p=2, dim=-1)
        # print('Chroma_pred_before_norm', chroma_pred.shape)

        
        chroma_pred = normalize_chroma(chroma_pred)
        chroma_gt = normalize_chroma(chroma_gt)

        # print('Chroma_pred', chroma_pred.shape)
        
        if keep_mask is not None:
            A_C = y_hat.shape[1]
            T = chroma_pred.shape[-1]
            keep_mask_reshape = keep_mask.repeat(A_C, 1, 1) # bs x 1 x L -> bs x ac x L
            loss_mask = torch.nn.functional.interpolate(keep_mask_reshape.float(), size=chroma_pred.shape[1])
            loss_mask = loss_mask.squeeze(1) # bs x ch=1 x L -> bs x L
        else:
            loss_mask = None
        loss = self.loss_func(chroma_pred, chroma_gt, loss_mask)

        return loss

## Masked losses
class MaskedMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, logits, targets, mask=None):
        if mask is None:
            return torch.nn.functional.l1_loss(logits, targets, reduction='mean')

        logits = logits.contiguous()
        targets = targets.contiguous()

        b, t, c = targets.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, c)

        loss_reduce_v = torch.clamp(torch.sum(mask), min=1.0)

        # print('MAE:', logits.shape, targets.shape, mask.shape)

        masked_mae_loss = torch.sum(self.mae(logits, targets) * mask) / loss_reduce_v
        return masked_mae_loss


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits, targets, mask=None):
        if mask is None:
            return torch.nn.functional.mse_loss(logits, targets)
        logits = logits.contiguous()
        targets = targets.contiguous()

        b, t, c = targets.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, c)

        loss_reduce_v = torch.clamp(torch.sum(mask), min=1.0)
        masked_mse_loss = torch.sum(self.mse(logits, targets) * mask) / loss_reduce_v
        return masked_mse_loss



def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.ones(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) > prob

class MaskedEmb(nn.Module):

    def __init__(self, n_embd: int, length=None, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.null_emb = nn.Parameter(torch.randn(n_embd), requires_grad=True)
        self.length = length

    def forward(
        self, x: torch.Tensor, mask_prob: float) -> torch.Tensor:
        B, D, T = x.shape # bs x emb x seq

        if self.length != None:
            ratio = T // self.length
            keep_mask = prob_mask_like(
                (B, 1, ratio), mask_prob, device=x.device
            )
            keep_mask = torch.nn.functional.interpolate(keep_mask.float(), size=T).bool()
        else:
            keep_mask = prob_mask_like(
                (B, 1, T), mask_prob, device=x.device
            )
        if self.learnable:
            x_null = repeat(self.null_emb, "d -> b d t", b=B, t=T)
            return torch.where(keep_mask, x, x_null), keep_mask
        else:
            x_null = torch.randn((B, D, T), device=x.device) * 0.1
            return torch.where(keep_mask, x, x_null), keep_mask
