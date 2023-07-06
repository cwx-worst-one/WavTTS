import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from recipes.soundstream.models.modules.resblocks import (
    CausalWNResBlock1,
    ModulatedResBlock1,
    WNResBlock1,
)
from recipes.soundstream.utils.utils import export_tensor, init_weights

LRELU_SLOPE = 0.1
padding_mode = "zeros"


class NearestUpsample(nn.Module):
    def __init__(self, up_scale, in_channels, out_channels):
        super().__init__()
        self.up_scale = up_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        self.conv = norm_f(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * up_scale - 1,
                padding_mode=padding_mode,
                padding=up_scale - 1,
            )
        )

    def forward(self, x):
        x_size = x.size(2)
        x = F.interpolate(x, scale_factor=self.up_scale, mode="nearest")
        x = self.conv(x)
        return x

    def remove_weight_norm(self):
        try:
            remove_weight_norm(self.conv)
        except:
            pass


class CausalUpsample(nn.Module):
    def __init__(self, up_scale, in_channels, out_channels):
        super().__init__()
        self.up_scale = up_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        self.pad_layer = nn.ConstantPad1d([2 * up_scale - 2, 0], 0.0)
        self.conv = norm_f(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=2 * up_scale - 1, padding=0
            )
        )

    def forward(self, x):
        b, d, t = x.size()
        res = torch.zeros([b, d, t * int(self.up_scale)]).to(x.device)
        res[:, :, 0 :: int(self.up_scale)] = x
        return self.conv(self.pad_layer(res))

    def remove_weight_norm(self):
        try:
            remove_weight_norm(self.conv)
        except:
            pass


class Generator(nn.Module):
    def __init__(
        self,
        upsample_rates,
        decoder_initial_channel,
        encoder_initial_channel,
        model_type="bytewave",
        trunc_noise=False,
    ):
        super().__init__()

        if model_type == "bytewave":
            resblock = ModulatedResBlock1
        elif model_type == "bytewave_wn":
            resblock = WNResBlock1
        elif model_type == "bytewave_wn_causal":
            resblock = CausalWNResBlock1
        else:
            raise Exception

        self.num_kernels = 2
        self.num_upsamples = len(upsample_rates)
        norm_f = weight_norm

        self.ups = nn.ModuleList()
        ch1 = decoder_initial_channel // 2
        ch2 = ch1 // 2
        ch3 = ch2 // 2
        ch4 = ch3 // 2
        chs = [ch1 * 2, ch1, ch2, ch3, ch4]

        if "causal" not in model_type:
            self.conv_pre = nn.Sequential(
                Conv1d(
                    16, #encoder_initial_channel * (2**4),
                    decoder_initial_channel,
                    7,
                    padding=3,
                    padding_mode=padding_mode,
                ),
                torch.nn.BatchNorm1d(decoder_initial_channel),
            )
            for i, u in enumerate(upsample_rates):
                layer = ConvTranspose1d(
                    chs[i], chs[i + 1], kernel_size=3 * u, padding=u, stride=u
                )
                layer = weight_norm(layer)
                self.ups.append(layer)
            self.ups.apply(init_weights)
            self.resblocks = nn.ModuleList(
                [
                    # --------------------------------- 11 --
                    resblock(ch1, 13, [1, 1, 1], trunc_noise),
                    resblock(ch1, 25, [1, 1, 1], trunc_noise),
                    # --------------------------------- 3 --
                    resblock(ch2, 25, [1, 1, 1], trunc_noise),
                    resblock(ch2, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------- 2 --
                    resblock(ch3, 25, [1, 1, 1], trunc_noise),
                    resblock(ch3, 49, [1, 1, 1], trunc_noise),
                    # --------------------------------- 1 --
                    resblock(ch4, 25, [1, 2, 4], trunc_noise),
                    resblock(ch4, 49, [1, 1, 1], trunc_noise),
                ]
            )
            self.conv_post = norm_f(
                Conv1d(ch4, 1, 63, 1, padding=31, padding_mode=padding_mode)
            )
            self.conv_post.apply(init_weights)
        else:
            self.conv_pre = nn.Sequential(
                nn.ConstantPad1d([6, 0], 0.0),
                Conv1d(
                    16, #encoder_initial_channel * (2**4),
                    decoder_initial_channel,
                    7,
                    padding=0,
                ),
                torch.nn.BatchNorm1d(decoder_initial_channel),
            )
            for i, u in enumerate(upsample_rates):
                layer = CausalUpsample(u, chs[i], chs[i + 1])
                self.ups.append(layer)
            self.resblocks = nn.ModuleList(
                [
                    # --------------------------------- 11 --
                    resblock(ch1, 7, [1, 1, 1], trunc_noise),
                    resblock(ch1, 13, [1, 1, 1], trunc_noise),
                    # --------------------------------- 3 --
                    resblock(ch2, 13, [1, 1, 1], trunc_noise),
                    resblock(ch2, 25, [1, 1, 1], trunc_noise),
                    # --------------------------------- 2 --
                    resblock(ch3, 13, [1, 1, 1], trunc_noise),
                    resblock(ch3, 25, [1, 1, 1], trunc_noise),
                    # --------------------------------- 1 --
                    resblock(ch4, 13, [1, 2, 4], trunc_noise),
                    resblock(ch4, 25, [1, 1, 1], trunc_noise),
                ]
            )
            self.conv_post = nn.Sequential(
                nn.ConstantPad1d([30, 0], 0.0), norm_f(Conv1d(ch4, 1, 31, 1, padding=0))
            )
            self.conv_post[1].apply(init_weights)

    def forward(self, x):
        # x = torch.tanh(x)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        pass


if __name__ == "__main__":
    from utils.utils import get_config_from_file

    hp = get_config_from_file("./configs/config_libri_causal.yaml").hparams
    generator = Generator(hp, model_type=hp.model_type)
    generator.eval()  # prevent bn influence
    x = torch.randn(size=[2, hp.encoder_initial_channel * (2**4), 100])
    x_ = x.clone()
    x_[:, :, 80:100] = 0

    y = generator(x)
    print("Input shape: {}\nOutput shape: {}".format(x.shape, y.shape))
    err = y - generator(x_)
    err = err[0, 0, 79 * hp.hop_size : 81 * hp.hop_size]
    print(err)
