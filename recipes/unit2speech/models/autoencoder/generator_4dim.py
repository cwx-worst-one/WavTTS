import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .resblocks import ModulatedResBlock1, WNResBlock1
from .utils import init_weights

LRELU_SLOPE = 0.1
padding_mode = 'zeros'


class NearestUpsample(nn.Module):

    def __init__(self, up_scale, in_channels, out_channels):
        super().__init__()
        self.up_scale = up_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        self.conv = norm_f(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=2 * up_scale - 1,
                      padding_mode=padding_mode,
                      padding=up_scale - 1))

    def forward(self, x):
        x_size = x.size(2)
        x = F.interpolate(x, scale_factor=self.up_scale, mode='nearest')
        x = self.conv(x)
        return x

    def remove_weight_norm(self):
        try:
            remove_weight_norm(self.conv)
        except:
            pass


class Generator(nn.Module):

    def __init__(self, hp, model_type='bytewave'):
        super().__init__()
        self.hp = hp
        upsample_rates = hp.upsample_rates
        decoder_initial_channel = hp.decoder_initial_channel

        if hasattr(hp, 'model_type'):
            model_type = hp.model_type
        if model_type == 'bytewave':
            resblock = ModulatedResBlock1
        elif model_type == 'bytewave_wn':
            resblock = WNResBlock1
        else:
            raise Exception

        self.num_kernels = 2
        self.num_upsamples = len(upsample_rates)
        norm_f = weight_norm

        self.conv_pre = nn.Sequential(
            Conv1d(4, decoder_initial_channel, 7, padding=3, padding_mode=padding_mode),
            torch.nn.BatchNorm1d(decoder_initial_channel)
        )
        self.ups = nn.ModuleList()
        '''
        for i, u in enumerate(upsample_rates):
            self.ups.append(
                NearestUpsample(u, decoder_initial_channel // (2**i),
                                decoder_initial_channel // (2**(i + 1))))
        '''
        ch1 = decoder_initial_channel // 2
        ch2 = ch1 // 2
        ch3 = ch2 // 2
        ch4 = ch3 // 2
        chs = [ch1*2, ch1, ch2, ch3, ch4]
        for i, u in enumerate(upsample_rates):
            layer = ConvTranspose1d(chs[i], chs[i+1], kernel_size=3*u, padding=u, stride=u)
            layer = weight_norm(layer)
            self.ups.append(layer)

        self.resblocks = nn.ModuleList([
            # --------------------------------- 11 --
            resblock(self.hp, ch1, 13, [1, 1, 1]),
            resblock(self.hp, ch1, 25, [1, 1, 1]),
            #resblock(self.hp, ch1, 11, [1, 3, 5]),
            # --------------------------------- 3 --
            resblock(self.hp, ch2, 25, [1, 1, 1]),
            resblock(self.hp, ch2, 49, [1, 1, 1]),
            #resblock(self.hp, ch2, 11, [1, 3, 5]),
            # --------------------------------- 2 --
            resblock(self.hp, ch3, 25, [1, 1, 1]),
            resblock(self.hp, ch3, 49, [1, 1, 1]),
            #resblock(self.hp, ch3, 11, [1, 3, 5]),
            # --------------------------------- 1 --
            resblock(self.hp, ch4, 25, [1, 2, 4]),
            resblock(self.hp, ch4, 49, [1, 1, 1]),
            #resblock(self.hp, ch4, 11, [1, 3, 5]),
        ])

        self.conv_post = norm_f(Conv1d(ch4, 1, 63, 1, padding=31, padding_mode=padding_mode))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)


    def forward(self, x):
        #x = torch.tanh(x)
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
        print('Removing weight norm...')
        pass
