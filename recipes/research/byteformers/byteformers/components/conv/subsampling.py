import math

import torch
import torch.nn as nn

from byteformers.components.conv.causal import CausalConv2D
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ConvSubsampling(nn.Module):
    """Convolutional subsampling
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling_factor (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling: str,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'dw_striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == 1 else conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 2 else conv_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:

            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(1, 2)
        
        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == 'dw_striding':
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)  # try anyway
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)
        return x, lengths


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
