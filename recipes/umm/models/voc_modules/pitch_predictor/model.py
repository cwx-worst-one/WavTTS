import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        n=2,
        norm_type="bn",
        dropout=0.0,
        c_multiple=2,
        ln_eps=1e-12,
        left_pad=False,
    ):
        super(ResidualBlock, self).__init__()

        if norm_type == "bn":
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == "in":
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == "gn":
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == "ln":
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        else:
            norm_builder = lambda: nn.Identity()

        if left_pad:
            self.blocks = [
                nn.Sequential(
                    norm_builder(),
                    nn.ConstantPad1d(((dilation * (kernel_size - 1)) // 2 * 2, 0), 0),
                    nn.Conv1d(
                        channels,
                        c_multiple * channels,
                        kernel_size,
                        dilation=dilation,
                        padding=0,
                    ),
                    LambdaLayer(lambda x: x * kernel_size**-0.5),
                    nn.GELU(),
                    nn.Conv1d(
                        c_multiple * channels,
                        channels,
                        1,
                        dilation=dilation,
                        padding_mode="reflect",
                    ),
                )
                for i in range(n)
            ]
        else:
            self.blocks = [
                nn.Sequential(
                    norm_builder(),
                    nn.Conv1d(
                        channels,
                        c_multiple * channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(dilation * (kernel_size - 1)) // 2,
                        padding_mode="reflect",
                    ),
                    LambdaLayer(lambda x: x * kernel_size**-0.5),
                    nn.GELU(),
                    nn.Conv1d(
                        c_multiple * channels,
                        channels,
                        1,
                        dilation=dilation,
                        padding_mode="reflect",
                    ),
                )
                for i in range(n)
            ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """
    Main building block for training perceptual pitch model for V2 Dual Arch UMM.

    Arguments:
        left_pad: False (default) for making the model causal
    """

    def __init__(
        self,
        hidden_size,
        out_dims,
        dilations,
        kernel_size,
        norm_type="ln",
        layers_in_block=2,
        c_multiple=2,
        dropout=0.0,
        ln_eps=1e-5,
        init_weights=True,
        is_BTC=True,
        num_layers=None,
        post_net_kernel=3,
        left_pad=False,
        c_in=None,
    ):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        if num_layers is not None:
            dilations = [1] * num_layers
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_size,
                    kernel_size,
                    d,
                    n=layers_in_block,
                    norm_type=norm_type,
                    c_multiple=c_multiple,
                    dropout=dropout,
                    ln_eps=ln_eps,
                    left_pad=left_pad,
                )
                for d in dilations
            ]
        )
        if norm_type == "bn":
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type == "in":
            norm = nn.InstanceNorm1d(hidden_size, affine=True)
        elif norm_type == "gn":
            norm = nn.GroupNorm(8, hidden_size)
        elif norm_type == "ln":
            norm = LayerNorm(hidden_size, dim=1, eps=ln_eps)
        self.last_norm = norm
        if left_pad:
            self.post_net1 = nn.Sequential(
                nn.ConstantPad1d((post_net_kernel // 2 * 2, 0), 0),
                nn.Conv1d(
                    hidden_size, out_dims, kernel_size=post_net_kernel, padding=0
                ),
            )
        else:
            self.post_net1 = nn.Conv1d(
                hidden_size,
                out_dims,
                kernel_size=post_net_kernel,
                padding=post_net_kernel // 2,
                padding_mode="reflect",
            )
        self.c_in = c_in
        if c_in is not None:
            self.in_conv = nn.Conv1d(
                c_in, hidden_size, kernel_size=1, padding_mode="reflect"
            )
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x, nonpadding=None):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
        if self.c_in is not None:
            x = self.in_conv(x)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class PitchPredictor(nn.Module):
    """
    Pitch Predictor for use as a perceptual pitch loss in V2 UMM-Dual Arch training.

    Arguments:
        c_in: number of mel channels (int), normally 160
        c_out: number of outputs (int), normally 2 corresponding to f0_hz prediction and voice/unvoiced state
        hidden_size: hidden size (int) of ResNet style blocks
        num_layers: number of layers of (int) of ResNet style block
    """

    def __init__(self, c_in=160, c_out=2, hidden_size=128, num_layers=5):
        super().__init__()
        self.linear_in = nn.Linear(c_in, hidden_size)
        self.blocks = ConvBlocks(
            hidden_size, hidden_size, None, 5, num_layers=num_layers
        )
        self.linear_out = nn.Linear(hidden_size, c_out)
        self.h = None  # explicitly save this for use as perceptual loss

    def forward(self, x):
        x = self.linear_in(x)
        x = self.blocks(x)
        self.h = x
        x = self.linear_out(x)
        return x

    def get_hidden_state(self):
        return self.h
