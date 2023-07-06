import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from recipes.soundstream.models.modules.resblocks import ModulatedResBlock1, WNResBlock1
from recipes.soundstream.utils.utils import get_padding, init_weights

LRELU_SLOPE = 0.1
padding_mode = "zeros"


class Downsample(nn.Module):
    def __init__(self, down_scale, in_channels, out_channels):
        super().__init__()
        self.down_scale = down_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        self.conv = norm_f(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * down_scale + 1,
                stride=down_scale,
                padding_mode=padding_mode,
                padding=down_scale,
            )
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    def remove_weight_norm(self):
        try:
            remove_weight_norm(self.conv)
        except:
            pass


class Encoder(nn.Module):
    def __init__(
        self,
        down_rates,
        encoder_initial_channel,
        trunc_noise,
        smaller_encoder,
        model_type="bytewave",
    ):
        super().__init__()

        if model_type == "bytewave":
            resblock = ModulatedResBlock1
        elif model_type == "bytewave_wn":
            resblock = WNResBlock1
        elif model_type == "bytewave_wn_causal":
            resblock = WNResBlock1
        else:
            raise Exception

        norm_f = weight_norm
        self.num_kernels = 2
        self.num_downsamples = len(down_rates)

        ch0 = encoder_initial_channel
        ch1 = ch0 * 2
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        ch4 = ch3 * 2

        if not "causal" in model_type:
            self.conv_pre = norm_f(
                Conv1d(1, ch0, 59, padding=29, padding_mode=padding_mode)
            )
            self.downs = nn.ModuleList()
            for i, d in enumerate(down_rates):
                self.downs.append(Downsample(d, ch0 * (2**i), ch0 * (2 ** (i + 1))))
            self.resblocks = nn.ModuleList(
                [
                    # --------------------------------------
                    resblock(ch0, 25, [1, 2, 4], trunc_noise),
                    resblock(ch0, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------------
                    resblock(ch1, 25, [1, 1, 1], trunc_noise),
                    resblock(ch1, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------------
                    resblock(ch2, 25, [1, 1, 1], trunc_noise),
                    resblock(ch2, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------------
                    resblock(ch3, 25, [1, 1, 1], trunc_noise),
                    resblock(ch3, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------------
                    resblock(ch4, 25, [1, 1, 1], trunc_noise),
                    resblock(ch4, 49, [1, 1, 1], trunc_noise),
                ]
            )
            self.conv_post = norm_f(
                Conv1d(ch4, ch4, 17, 1, padding=8, padding_mode=padding_mode)
            )
        else:
            self.conv_pre = norm_f(
                Conv1d(1, ch0, 59, padding=29, padding_mode=padding_mode)
            )
            self.downs = nn.ModuleList()
            for i, d in enumerate(down_rates):
                self.downs.append(Downsample(d, ch0 * (2**i), ch0 * (2 ** (i + 1))))
            if smaller_encoder:
                self.resblocks = nn.ModuleList(
                    [
                        # --------------------------------------
                        resblock(ch0, 13, [1, 2, 4], trunc_noise),
                        resblock(ch0, 25, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch1, 13, [1, 1, 1], trunc_noise),
                        resblock(ch1, 25, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch2, 13, [1, 1, 1], trunc_noise),
                        resblock(ch2, 25, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch3, 5, [1, 1, 1], trunc_noise),
                        resblock(ch3, 7, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch4, 1, [1, 1, 1], trunc_noise),
                        resblock(ch4, 1, [1, 1, 1], trunc_noise),
                    ]
                )
                self.conv_post = norm_f(
                    Conv1d(ch4, ch4, 1, 1, padding=0, padding_mode=padding_mode)
                )
            else:
                self.resblocks = nn.ModuleList(
                    [
                        # --------------------------------------
                        resblock(ch0, 25, [1, 2, 4], trunc_noise),
                        resblock(ch0, 49, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch1, 25, [1, 1, 1], trunc_noise),
                        resblock(ch1, 49, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch2, 25, [1, 1, 1], trunc_noise),
                        resblock(ch2, 49, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch3, 7, [1, 1, 1], trunc_noise),
                        resblock(ch3, 11, [1, 1, 1], trunc_noise),
                        # --------------------------------------
                        resblock(ch4, 1, [1, 1, 1], trunc_noise),
                        resblock(ch4, 1, [1, 1, 1], trunc_noise),
                    ]
                )
                self.conv_post = norm_f(
                    Conv1d(ch4, ch4, 3, 1, padding=1, padding_mode=padding_mode)
                )

        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)
        # self.register_buffer("cnt", torch.FloatTensor([0]))
        self.encoder_bn = nn.Identity() #nn.BatchNorm1d(ch4, affine=False, momentum=0.05)
        
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_downsamples + 1):
            x = F.leaky_relu(x, LRELU_SLOPE)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # x = xs / self.num_kernels
            x = xs
            if i < self.num_downsamples:
                x = self.downs[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = self.encoder_bn(x)
        # if self.training:
        #     x = x + torch.randn_like(x) * ((1e4 - self.cnt).clamp(0) / 1e4)
            # self.cnt.add_(1)
        return x

    def freeze_bn(self):
        self.encoder_bn.eval()

    def remove_weight_norm(self):
        print("Removing weight norm...")
        pass


if __name__ == "__main__":

    encoder = Encoder(
        down_rates=[2, 2, 10, 12],
        encoder_initial_channel=16,
        trunc_noise=False,
        smaller_encoder=True,
        model_type="bytewave_wn_causal",
    )
    x = torch.randn(size=[2, 1, 32 * 400])
    y = encoder(x)
    print("Input shape: {}\nOutput shape: {}".format(x.shape, y.shape))

    err = (y - 1).mean()
    err.backward()
    for name, params in encoder.named_parameters():
        if params.grad is None:
            print(name, params.shape)
