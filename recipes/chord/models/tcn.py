import torch.nn.functional as F
from torch import nn

from recipes.beat.models.networks import Preprocess
from samantha.core import BaseStage

BN_MOMENTUM = 0.01
epsilon = 1e-10


def init_layer(layer):
    r"""Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    r"""Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        r"""
        Args:
          input: (batch_size, in_channels, freq_bins, time_steps)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class tcn_block(nn.Module):
    def __init__(self, in_channels, oup_channels, kernel_size, dilate):
        super(tcn_block, self).__init__()

        pad = dilate * (kernel_size - 1) // 2

        self.conv1x1 = nn.Conv1d(in_channels, oup_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(oup_channels)
        self.conv_d = nn.Conv1d(
            oup_channels,
            oup_channels,
            kernel_size,
            padding=pad,
            dilation=dilate,
            groups=oup_channels,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(oup_channels)
        self.conv_out = nn.Conv1d(oup_channels, in_channels, 1)
        self.conv_skip = nn.Conv1d(oup_channels, in_channels, 1)

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_timesteps)
        """

        hid = self.conv1x1(x)
        hid = self.prelu1(hid)
        hid = self.norm1(hid)
        hid = self.conv_d(hid)
        hid = self.prelu2(hid)
        hid = self.norm2(hid)
        output = self.conv_out(hid)
        skip = self.conv_skip(hid)

        return output, skip


class TCNStage(BaseStage):
    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        input_feature,
        n_channel=128,
        n_chan_conv=512,
        front_conv=3,
        resnet_pools=[[2, 2]],
        n_harmonic=6,
        n_repeats=1,
        n_layers=11,
        kernel_size=3,
        semitone_scale=1,
        dropout_rate=0.1,
        learn_bw="only_Q",
        takes=["audio", "aug_hop_size"],
        provides=["skip_sum"],
        serialize_opts=None,
    ):
        super().__init__(takes, provides, serialize_opts)

        padding = front_conv // 2

        # input feature
        self.preprocess = Preprocess(
            sample_rate,
            n_fft,
            n_harmonic,
            semitone_scale,
            learn_bw,
            hop_length,
            input_feature,
        )
        self.dropout = nn.Dropout2d(p=0.3)

        # input dilation
        conv_blocks = []
        input_channel = 6 * 64
        for pool in resnet_pools:
            conv_blocks.append(
                nn.Conv1d(input_channel, n_channel, front_conv, 1, padding)
            )
            conv_blocks.append(nn.BatchNorm1d(n_channel))
            conv_blocks.append(nn.MaxPool1d(pool[-1]))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.Dropout(p=dropout_rate))
            input_channel = n_channel
        self.input_layer = nn.Sequential(*conv_blocks)
        self.input_norm = nn.BatchNorm1d(6 * 64)

        # TCN dileted convolution
        di_conv_layers = []
        for _ in range(n_repeats):
            for i_block in range(n_layers):
                di_conv_layers.append(
                    tcn_block(n_channel, n_chan_conv, kernel_size, 2**i_block)
                )
        self.layers = nn.ModuleList(di_conv_layers)

        self.prelu_out = nn.PReLU()

    def forward(self, data):
        oup = {}
        inp, aug_hop_size = data["audio"], data["aug_hop_size"]

        # audio to spectrogram
        inp = self.preprocess(inp, aug_hop_size)

        # input conv
        b, c, f, t = inp.shape
        inp = inp.reshape(b, -1, t)
        output = self.input_layer(self.input_norm(inp))

        output = self.dropout(output)

        # TCN
        skip_sum = 0.0
        for i, di_conv_layer in enumerate(self.layers):
            residual, skip = di_conv_layer(output)
            output = output + residual
            skip_sum = skip_sum + skip
            oup[f"emb_{str(i)}"] = self.prelu_out(skip_sum).permute(0, 2, 1)
        skip_sum = self.prelu_out(skip_sum).permute(0, 2, 1)

        oup["emb"] = skip_sum

        return oup
