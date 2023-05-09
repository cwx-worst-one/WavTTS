# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from math import log as ln


class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList(
            [nn.Linear(64, 128), nn.Linear(128, 256), nn.Linear(256, 512)]
        )

    def forward(self, std_step):
        """
        Arguments:
          std_step:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(std_step)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, std_step):
        """
        Arguments:
          std_step:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        freqs = freqs.to(device=torch.device("cuda"))
        table = 2 * np.pi * std_step * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class GammaBeta(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(512, 2 * output_dim)

    def forward(self, noise_level_encoding):
        noise_level_encoding = self.output_layer(noise_level_encoding)
        noise_level_encoding = noise_level_encoding.unsqueeze(-1)
        gamma, beta = torch.chunk(noise_level_encoding, 2, dim=1)
        return gamma, beta


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = nn.Conv1d(input_size, hidden_size, 1, stride=factor)
        self.layer_1 = nn.Conv1d(
            input_size, hidden_size, 3, dilation=1, padding=1, stride=factor
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
                nn.Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
                nn.Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8),
            ]
        )

    def forward(self, x, gamma, beta):
        residual = self.residual_dense(x)

        x = F.leaky_relu(x, 0.2)
        x = self.layer_1(x)
        x = gamma * x + beta
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 32, 5, padding="same")
        self.embedding = RFF_MLP_Block()
        self.downsample = nn.ModuleList(
            [
                DBlock(32, 128, 4),
                DBlock(128, 256, 6),
                DBlock(256, 256, 5),
                DBlock(256, 256, 5),
                DBlock(256, 512, 5),
                DBlock(512, 512, 7),
            ]
        )
        self.gamma_beta = nn.ModuleList(
            [
                GammaBeta(128),
                GammaBeta(256),
                GammaBeta(256),
                GammaBeta(256),
                GammaBeta(512),
                GammaBeta(512),
            ]
        )
        self.last_conv = nn.Linear(512, 3)

    def forward(self, audio, noise_scale):
        x = audio.mean(dim=1, keepdim=True)
        x = self.conv_1(x)
        noise_scale = self.embedding(noise_scale.squeeze(1))

        for film, layer in zip(self.gamma_beta, self.downsample):
            gamma, beta = film(noise_scale)
            x = layer(x, gamma, beta)

        x = F.softmax(self.last_conv(x.permute([0, 2, 1])), dim=2)
        return x.permute([0, 2, 1])
