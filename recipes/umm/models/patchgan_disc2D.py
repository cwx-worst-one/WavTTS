import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscriminatorBlock2D(nn.Module):
    def __init__(self, input_channels, filters, strides=(2, 2), kernel_size=3, use_norm=True, norm_type='in'):
        super().__init__()
        self.strides = strides
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=strides)
        self.net = [
            nn.Conv2d(input_channels, filters, kernel_size, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2)
        ]
        if use_norm:
            if norm_type == 'in':
                self.net.append(nn.InstanceNorm2d(filters, affine=True))
            elif norm_type == 'gn':
                self.net.append(nn.GroupNorm(4, filters))
            elif norm_type == 'sn':
                self.net[0] = spectral_norm(self.net[0])
            else:
                assert False
        self.net = nn.Sequential(*self.net)
        self.downsample_layer = nn.Conv2d(filters, filters, 3, padding=1, stride=strides)

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = self.downsample_layer(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class PatchGANLayers(nn.Module):
    def __init__(self, freq_length=80, num_layers=4, pooling_size=(2, 2), hidden_size=32,
                 c_cond=0, max_hidden_size=256, kernel_size=3, norm_type='in'):
        super().__init__()
        filters = [1] + [min(hidden_size * (2 ** i), max_hidden_size) for i in range(num_layers)]
        chan_in_out = list(zip(filters[:-1], filters[1:]))
        freq_length = freq_length // pooling_size[0]
        self.pooling = nn.AvgPool2d(pooling_size) if pooling_size[0] != 1 else nn.Identity()
        self.blocks = nn.ModuleList()
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            block = DiscriminatorBlock2D(in_chan, out_chan, (2, 2), kernel_size, norm_type=norm_type)
            self.blocks.append(block)
        self.out = nn.Conv1d(filters[-1] * math.ceil(freq_length / (2 ** 4)), 1, kernel_size=1)

        self.c_cond = c_cond
        if c_cond > 0:
            self.cond_pooling = nn.AvgPool2d((pooling_size[0], 1)) if pooling_size is not None else nn.Identity()
            self.c_blocks = nn.ModuleList()
            self.cond_ds_blocks = nn.ModuleList()
            for ind, (in_chan, out_chan) in enumerate(chan_in_out):
                if ind > len(chan_in_out) // 2:
                    block = DiscriminatorBlock2D(in_chan, out_chan, (2, 2,), kernel_size)
                    self.c_blocks.append(block)
                self.cond_ds_blocks.append(nn.Conv2d(c_cond, out_chan, (3, 1), (2, 1), padding=(1, 0)))
                c_cond = out_chan
            self.cond_out = nn.Conv1d(filters[-1] * math.ceil(freq_length / (2 ** 4)), 1, kernel_size=1)

    def forward(self, x_inp, c=None):
        h = []
        x = self.pooling(x_inp)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i == len(self.blocks) // 2:
                mid_hidden = x
            h.append(x)
        B, C, T, W = x.shape
        x = self.out(x.transpose(2, 3).reshape(B, -1, T))
        if c is not None:
            c = self.cond_pooling(c)
            x_c = mid_hidden
            for i, b_c in enumerate(self.cond_ds_blocks):
                c = b_c(c)
                if i > len(self.cond_ds_blocks) // 2:
                    b = self.c_blocks[i - len(self.cond_ds_blocks) // 2 - 1]
                    x_c = b(x_c)
                    x_c = x_c + c
                h.append(x_c)
            x_c = self.cond_out(x_c.transpose(2, 3).reshape(B, -1, T))
            x = torch.cat([x, x_c], 0)
        return x[:, 0], h


class PatchGANDisc2D(nn.Module):
    def __init__(self, freq_length=80, time_length=512, cond_size=0, hidden_size=32, max_hidden_size=256,
                 kernel_size=3, num_layers=4, norm_type='in', num_disc=3, same_clip_batch=True):
        super().__init__()
        self.win_length = time_length
        if self.win_length <= 0:
            self.win_length = 1e9
        layer_cls = partial(PatchGANLayers, hidden_size=hidden_size, c_cond=cond_size,
                            max_hidden_size=max_hidden_size, kernel_size=kernel_size,
                            num_layers=num_layers, norm_type=norm_type, freq_length=freq_length)
        self.disc_layers = nn.ModuleList([
            layer_cls(pooling_size=(1, 1)),
            layer_cls(pooling_size=(2, 2)),
            layer_cls(pooling_size=(4, 4)),
        ])[:num_disc]
        self.same_clip_batch = same_clip_batch

    def forward(self, x, cond=None, start_frames=None):
        '''
        Args:
            x (tensor): input mel, (B, T, n_bins).
            cond (tensor): input condition, (B, T, c_cond).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        x_len = (x.abs().sum(-1) > 0).sum(-1)
        hiddens = []
        x = x[:, None]
        if cond is not None:
            cond = cond.transpose(1, 2)[..., None]
        if start_frames is None:
            start_frames = [None]
        # (B, win_length, C)
        x_clip, cond_clip, start_frame = self.clip(x, cond, x_len, self.win_length, start_frames[0])
        start_frames[0] = start_frame
        v = []
        for disc in self.disc_layers:
            validity, h = disc(x_clip, cond_clip)
            v.append(validity)
            hiddens += h
        return {"y": v, "start_frames": start_frames, "h": hiddens}

    def clip(self, x, cond, x_len, win_length, start_frame=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, c_cond, T, n_bins).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        T_start = 0
        if self.same_clip_batch:
            T_end = x_len.max() - win_length
            if T_end <= 0:
                return x, cond, 0
            T_end = T_end.item()
            if start_frame is None:
                start_frame = np.random.randint(low=T_start, high=T_end + 1)
            if cond is not None:
                cond = cond[:, :, start_frame: start_frame + win_length]
            x_batch = x[:, :, start_frame: start_frame + win_length]
        else:
            x_batch_ = []
            cond_ = []
            for b in range(x.shape[0]):
                T_end = x_len[b].item() - win_length
                if T_end <= 0:
                    if cond is not None:
                        cond_.append(cond[b])
                    x_batch_.append(x[b])
                else:
                    start_frame = np.random.randint(low=T_start, high=T_end + 1)
                    if cond is not None:
                        cond_.append(cond[b, :, start_frame: start_frame + win_length])
                    x_batch_.append(x[b, :, start_frame: start_frame + win_length])
            x_batch = torch.stack(x_batch_, 0)
            if cond is not None:
                cond = torch.stack(cond_, 0)
        return x_batch, cond, start_frame
