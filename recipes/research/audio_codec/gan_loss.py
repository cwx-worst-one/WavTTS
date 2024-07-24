from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature

    def loss(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen_loss, feature_distance = self.generator_loss(fake, real)
        dis_loss = self.discriminator_loss(fake, real)

        return dis_loss, gen_loss, feature_distance
