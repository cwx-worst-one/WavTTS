from abc import ABC
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from samantha.criterion.spectral_loss import (
    MagnitudeSTFTLoss,
    MultiScaleSTFTLoss,
    SpectralConvergengeLoss,
    apply_reduction,
)


class CriterionJoint(ABC):
    def __init__(
        self,
        loss_criteria: List = None,
        loss_weights: List = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_criteria = loss_criteria
        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(loss_criteria)

    def get_loss(self, x):
        output_dict = {}
        loss = 0
        if self.loss_criteria is not None:
            for criterion, weight in zip(self.loss_criteria, self.loss_weights):
                loss += weight * criterion(x)
        output_dict['loss'] = loss
        return output_dict


class STFTLoss(nn.Module):
    """STFT loss module.
    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).
    """

    def __init__(
        self,
        w_spectral_convergence: float = 1.0,
        w_lin_mag: float = 1.0,
        reduction: str = "mean",
        mag_distance: Optional[str] = "L1",
    ):
        super().__init__()
        self.spec_conv_loss = SpectralConvergengeLoss()
        self.mag_stft_loss = MagnitudeSTFTLoss(
            distance=mag_distance, reduction=reduction
        )
        self.w_spectral_convergence = w_spectral_convergence
        self.w_lin_mag = w_lin_mag
        self.reduction = reduction

    def forward(
        self, mag_x: torch.Tensor, mag_y: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        spec_mag_loss = (
            self.spec_conv_loss(mag_x, mag_y) if self.w_spectral_convergence else 0.0
        )
        lin_mag_loss = self.mag_stft_loss(mag_x, mag_y) if self.w_lin_mag else 0.0

        loss = (self.w_spectral_convergence * spec_mag_loss) + (
            self.w_lin_mag * lin_mag_loss
        )

        if self.reduction == "none":
            loss = loss
        else:
            loss = apply_reduction(loss, reduction=self.reduction)
        return {
            "stft_loss": loss,
            "spec_mag_loss": spec_mag_loss,
            "lin_mag_loss": lin_mag_loss,
        }