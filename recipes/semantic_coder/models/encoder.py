from torch import nn

from recipes.semantic_coder.models.blocks import ResidualUnit, Snake1d, WNConv1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride + 1,
                stride=stride,
                padding_mode="zeros",
                padding=stride,
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self, feature_dim: int = 1024, model_dim: int = 64, strides: list = [2, 4, 8, 8]
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(feature_dim, model_dim, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            model_dim *= 2
            self.block += [EncoderBlock(model_dim, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(model_dim),
            WNConv1d(model_dim, model_dim, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = model_dim

    def forward(self, x):
        return self.block(x)
