from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from samantha.transforms.audio import MelSpectrogram, Spectrogram, safe_log


def apply_reduction(losses: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses


class SpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss.
    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class ItakuraSaitoDivergenceLoss(nn.Module):
    def __init__(self):
        """Calculate the Itakura-Saito Divergence"""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return y_mag / x_mag - safe_log(y_mag / x_mag) - 1


class MagnitudeSTFTLoss(nn.Module):
    def __init__(self, distance: str = "L1", reduction: str = "mean"):
        super().__init__()
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Takes in two magnitude spectrograms and returns the
        distance between them.

        Args:
            x_mag (torch.Tensor): _description_
            y_mag (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.distance(x_mag, y_mag)


class LogMagnitudeSTFTLoss(MagnitudeSTFTLoss):
    def __init__(self, distance: str = "L1", reduction: str = "mean"):
        super().__init__(distance=distance, reduction=reduction)

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Takes in two magnitude spectrograms, applies a log and
        returns the l1 loss between them.

        Args:
            x_mag (torch.Tensor): _description_
            y_mag (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.distance(safe_log(x_mag), safe_log(y_mag))


class STFTLoss(nn.Module):
    """STFT loss module.
    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).
    """

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        scale: Optional[str] = None,
        sample_rate: Optional[int] = None,
        n_mels: Optional[int] = None,
        w_spectral_convergence: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phase: float = 0.0,
        reduction: str = "mean",
        mag_distance: Optional[str] = "L1",
    ):
        super().__init__()
        self.spec_conv_loss = SpectralConvergengeLoss()
        self.log_mag_stft_loss = LogMagnitudeSTFTLoss(
            distance=mag_distance, reduction=reduction
        )
        self.mag_stft_loss = MagnitudeSTFTLoss(
            distance=mag_distance, reduction=reduction
        )
        self.phase_loss = nn.MSELoss()
        self.w_spectral_convergence = w_spectral_convergence
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phase = w_phase
        self.reduction = reduction

        if scale is None:
            self.transform = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=1.0,  # magnitude spectrogram
                return_phase=True,
            )
        elif scale == "mel":
            assert n_mels is not None and sample_rate is not None
            self.transform = MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=1.0,  # magnitude spectrogram
                return_phase=True,
            )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_all: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        mag_x, phase_x = self.transform(x)
        mag_y, phase_y = self.transform(y)

        spec_mag_loss = (
            self.spec_conv_loss(mag_x, mag_y) if self.w_spectral_convergence else 0.0
        )
        log_mag_loss = self.log_mag_stft_loss(mag_x, mag_y) if self.w_log_mag else 0.0
        lin_mag_loss = self.mag_stft_loss(mag_x, mag_y) if self.w_lin_mag else 0.0
        phase_loss = self.phase_loss(phase_x, phase_y) if self.w_phase else 0.0

        loss = (
            (self.w_spectral_convergence * spec_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phase * phase_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)
        if return_all:
            return loss, dict(
                spec_mag_loss=spec_mag_loss,
                lin_mag_loss=lin_mag_loss,
                log_mag_loss=log_mag_loss,
                phase_loss=phase_loss,
            )
        return loss


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        n_ffts: List[int] = [8192, 4096, 2048, 1024, 512, 256, 128, 64],
        win_lengths: List[int] = [512, 512, 512, 256, 128, 64, 32, 16],
        hop_lengths: List[int] = [4096, 2048, 1024, 512, 256, 128, 64, 32],
        scale: Optional[str] = None,
        sample_rate: Optional[int] = None,
        n_mels: Optional[List[int]] = None,
        w_spectral_convergence: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phase: float = 0.0,
        reduction: str = "mean",
        mag_distance: Optional[str] = "L1",
    ):
        super().__init__()
        self.losses = nn.ModuleList([])
        for idx in range(len(n_ffts)):
            self.losses.append(
                STFTLoss(
                    n_fft=n_ffts[idx],
                    win_length=win_lengths[idx],
                    hop_length=hop_lengths[idx],
                    scale=scale,
                    sample_rate=sample_rate,
                    n_mels=n_mels[idx] if n_mels is not None else None,
                    w_spectral_convergence=w_spectral_convergence,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_phase=w_phase,
                    reduction=reduction,
                    mag_distance=mag_distance,
                )
            )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        losses = []
        total_loss = 0.0
        for loss_fn in self.losses:
            loss = loss_fn(x, y)
            total_loss += loss
            losses.append(loss)

        if return_all:
            return total_loss, losses
        return total_loss
