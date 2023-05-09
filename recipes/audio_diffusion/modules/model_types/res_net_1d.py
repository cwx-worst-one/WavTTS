import torch.nn as nn

from .blocks import DepthWiseConvBlock1d
from .layers import SigmaEmbedding


class ResNet1d(nn.Module):
    def __init__(self, width=256, depth=16, kernel_size=49, num_signal_channels=2):
        super().__init__()
        self.sigma_embeddings = nn.ModuleList([SigmaEmbedding(64, width)] * depth)
        self.input_layer = nn.Conv1d(num_signal_channels, width, 3, padding="same")
        self.output_layer = nn.Conv1d(width, num_signal_channels, 3, padding="same")
        self.layers = nn.ModuleList(
            [DepthWiseConvBlock1d(width, width, width, kernel_size=kernel_size)] * depth
        )

    def forward(self, inp, sigma):
        inputs = self.input_layer(inp)
        for layer, embedding in zip(self.layers, self.sigma_embeddings):
            sigma_embedding = embedding(sigma)
            inputs = layer(inputs + sigma_embedding)
        return self.output_layer(inputs)
