# Author: liguangzheng

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from core.models.se.model_denoise_base import BaseModel
from core.models.layers.complex_layer import PhaseEncoderLogCh
from core.models.layers.normalization import ChannelwiseLayerNorm
from core.models.layers.tcn_layer import TcnAttBlock
from core.models.utils import unfold


class DSConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1):
        super(DSConv2d, self).__init__()
        self.dconv2d = nn.Conv2d(
            in_channel, in_channel, kernel, stride, padding, dilation, groups=in_channel
        )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.pconv2d = nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), (0, 0))
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

    def forward(self, input):
        out1 = self.nonlinearity1(self.bn1(self.dconv2d(input)))
        out2 = self.nonlinearity2(self.bn2(self.pconv2d(out1)))
        return out2


class DenoiseMcmodelFullsubnet3CIrmCplv2(nn.Module):
    def __init__(
        self,
        args,
    ):
        """
        Frame shift 10ms model, 3-channels model; delay=20ms; macs is 38.82 MMac params is 258.33 k
        """
        super().__init__()

        self.fb_layers = args.fb_layers
        self.sb_layers = args.sb_layers
        self.num_freqs = args.num_freqs
        self.fb_model_hidden_size = args.fb_model_hidden_size
        self.sb_model_hidden_size = args.sb_model_hidden_size
        self.faam = args.faam
        self.sb_kernel = args.sb_kernel
        self.sb_stride = args.sb_stride
        self.fb_num_neighbors = args.fb_num_neighbors

        self.phase_encoder = PhaseEncoderLogCh(3, 3)

        self.mc_model1 = DSConv2d(3, 2, (3, 3), (2, 1), (1, 2))
        self.mc_model2 = DSConv2d(2, 2, (3, 3), (1, 1), (1, 2))
        self.mc_model3 = DSConv2d(2, 1, (3, 3), (1, 1), (1, 2))

        self.fc_out = nn.Sequential(nn.Conv1d(80, 160, 1), nn.Conv1d(160, 160, 1))

        self.gru1 = nn.GRU(80, 80, 1, bias=True, batch_first=True)
        self.gru2 = nn.GRU(80, 80, 1, bias=True, batch_first=True)

        self.fb_model = torch.nn.ModuleList([])
        for i in range(self.fb_layers // 2):
            self.fb_model.append(
                TcnAttBlock(
                    self.fb_model_hidden_size // 2,
                    self.fb_model_hidden_size,
                    self.fb_model_hidden_size // 2,
                    3,
                    1,
                    1,
                    0.1,
                    True,
                    True,
                    self.faam,
                )
            )
            if i == 0:
                self.fb_model.append(
                    TcnAttBlock(
                        self.fb_model_hidden_size // 2,
                        self.fb_model_hidden_size,
                        self.fb_model_hidden_size // 2,
                        3,
                        1,
                        2,
                        0.1,
                        True,
                        True,
                        self.faam,
                    )
                )
            else:
                self.fb_model.append(
                    TcnAttBlock(
                        self.fb_model_hidden_size // 2,
                        self.fb_model_hidden_size,
                        self.fb_model_hidden_size // 2,
                        3,
                        1,
                        2,
                        0.1,
                        True,
                        False,
                        self.faam,
                    )
                )

        self.sb_model = torch.nn.ModuleList([])
        sb_model_in_size = self.sb_kernel + self.sb_stride
        for _ in range(self.sb_layers // 2):
            self.sb_model.append(
                TcnAttBlock(
                    sb_model_in_size,
                    self.sb_model_hidden_size,
                    sb_model_in_size,
                    3,
                    1,
                    1,
                    0.1,
                    True,
                    True,
                    self.faam,
                )
            )
            self.sb_model.append(
                TcnAttBlock(
                    sb_model_in_size,
                    self.sb_model_hidden_size,
                    sb_model_in_size,
                    3,
                    1,
                    2,
                    0.1,
                    True,
                    True,
                    self.faam,
                )
            )
        self.sb_model.append(torch.nn.Conv1d(sb_model_in_size, self.sb_stride, 1))

    def forward(self, input):
        noisy_r_1 = input["noisy_r"][:, 0:1, :, :160]
        noisy_r_2 = input["noisy_r"][:, 2:3, :, :160]
        noisy_r_3 = input["noisy_r"][:, 4:5, :, :160]
        noisy_i_1 = input["noisy_i"][:, 0:1, :, :160]
        noisy_i_2 = input["noisy_i"][:, 2:3, :, :160]
        noisy_i_3 = input["noisy_i"][:, 4:5, :, :160]
        noisy_r = torch.cat([noisy_r_1, noisy_r_2, noisy_r_3], dim=1)
        noisy_i = torch.cat([noisy_i_1, noisy_i_2, noisy_i_3], dim=1)
        noisy = torch.cat([noisy_r.unsqueeze(-1), noisy_i.unsqueeze(-1)], dim=-1)

        noisy_mag = self.phase_encoder(noisy)
        # print(f"after complex conv noisy_mag shape is {noisy_mag.shape}")
        noisy_mag = noisy_mag[:, :, :, 1:]
        # print(f"noisy shape {noisy.shape} noisy_mag shape {noisy_mag.shape}")

        # mcnet
        noisy_ri = noisy_mag.transpose(2, 3)
        noisy_ri = self.mc_model1(noisy_ri)
        noisy_ri = noisy_ri[:, :, :, :-2]
        noisy_ri = self.mc_model2(noisy_ri)
        noisy_ri = noisy_ri[:, :, :, :-2]
        noisy_ri = self.mc_model3(noisy_ri)
        noisy_ri = noisy_ri[:, :, :, :-2]
        noisy_mag = noisy_ri

        assert noisy_mag.dim() == 4
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # fullnet
        fb_input = noisy_mag.reshape(batch_size, -1, num_frames)
        for i in range(self.fb_layers):
            fb_input = self.fb_model[i](fb_input)

        fb_input = fb_input.transpose(1, 2)
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        fb_input = self.gru1(fb_input)[0]
        fb_input = fb_input.transpose(1, 2)
        fb_output = fb_input.reshape(batch_size, 1, num_freqs, num_frames)
        fb_output_unfolded = unfold(fb_output, num_neighbors=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(
            batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames
        )

        # subnet
        noisy_mag_ = noisy_mag.squeeze(1).transpose(1, 2).unsqueeze(-1)
        pad_num = int((self.sb_kernel - self.sb_stride) / 2)
        noisy_mag_ = F.pad(noisy_mag_, [0, 0, pad_num, pad_num])
        sb_input1 = F.unfold(noisy_mag_, (self.sb_kernel, 1), stride=(self.sb_stride, 1))
        sb_input1 = sb_input1.reshape([batch_size, num_frames, self.sb_kernel, -1])
        fb_output_unfolded_ = fb_output_unfolded.squeeze(2).transpose(1, 2).unsqueeze(-1)
        sb_input2 = F.unfold(fb_output_unfolded_, (self.sb_stride, 1), stride=(self.sb_stride, 1))
        sb_input2 = sb_input2.reshape([batch_size, num_frames, self.sb_stride, -1])
        sb_input = torch.cat([sb_input1, sb_input2], dim=2)
        sb_input = sb_input.permute(0, 3, 2, 1).reshape(
            -1, self.sb_kernel + self.sb_stride, num_frames
        )
        for j in range(self.sb_layers + 1):
            sb_input = self.sb_model[j](sb_input)

        sb_input = sb_input.reshape([batch_size, -1, self.sb_stride, num_frames])

        mask = sb_input.reshape([batch_size, -1, num_frames])
        mask = mask.transpose(1, 2)
        mask = self.gru2(mask)[0]
        mask = mask.transpose(1, 2)

        mask = torch.sigmoid(self.fc_out(mask)).transpose(1, 2)

        mask = F.pad(mask, [0, 1], 'constant', 0)

        enh_r = mask * input["noisy_r"][:, 0, :, :]
        enh_i = mask * input["noisy_i"][:, 0, :, :]

        out = {"enh_r": enh_r, "enh_i": enh_i}

        return out


class CplCnnPoolFgruTgruRealmaskV10(nn.Module):
    def __init__(
        self,
        args,
    ):
        """
        macs is 182.16 MMac params is 506.88 k, delay is 20ms
        """
        super().__init__()
        self.gru_dim = args.gru_dim

        self.phase_encoder = PhaseEncoderLogCh(1, 3, kernel_size=[3, 3], padding=[1, 1])

        self.conv_2d1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_2d2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1)
        )
        self.conv_2d3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1)
        )
        self.conv_2d4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1)
        )
        self.conv_2d5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1)
        )

        # normal bn
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        self.act5 = nn.PReLU()

        self.maxpool2 = nn.MaxPool2d((3, 1), stride=(2, 1), ceil_mode=True)  # 向上取整
        self.maxpool3 = nn.MaxPool2d((3, 1), stride=(2, 1), ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d((3, 1), stride=(2, 1), ceil_mode=True)
        self.maxpool5 = nn.MaxPool2d((3, 1), stride=(2, 1), ceil_mode=True)

        # # frequency gru
        self.gru_frequency = nn.GRU(32, 32, 1, bias=True, batch_first=True, bidirectional=True)

        # # time axis grp
        self.cln = ChannelwiseLayerNorm(640)
        self.BN = nn.Conv1d(640, self.gru_dim, 1)
        self.gru_time = nn.GRU(self.gru_dim, self.gru_dim, 1, bias=True, batch_first=True)

        self.linear = nn.Linear(in_features=self.gru_dim, out_features=160)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        noisy_r = input["noisy_r"][:, 0:1, :, :]
        noisy_i = input["noisy_i"][:, 0:1, :, :]
        noisy = torch.cat([noisy_r.unsqueeze(-1), noisy_i.unsqueeze(-1)], dim=-1)

        noisy_mag = self.phase_encoder(noisy)
        noisy_mag = noisy_mag[:, :, :, 1:]

        e1 = self.act1(self.bn1(self.conv_2d1(noisy_mag)))
        e2 = self.act2(self.bn2(self.conv_2d2(e1)))[:, :, :-2, :]
        e2 = self.maxpool2(e2.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        e3 = self.act3(self.bn3(self.conv_2d3(e2)))[:, :, :-2, :]
        e3 = self.maxpool3(e3.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        e4 = self.act4(self.bn4(self.conv_2d4(e3)))[:, :, :-2, :]
        e4 = self.maxpool4(e4.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        e5 = self.act5(self.bn5(self.conv_2d5(e4)))[:, :, :-2, :]
        e5 = self.maxpool5(e5.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()

        # frequency gru processing
        B, C, T, F = e5.shape
        gru_fre_input = e5.permute(0, 2, 1, 3).contiguous()
        gru_fre_input = gru_fre_input.view(B * T, F, C).contiguous()
        gru_fre_output = self.gru_frequency(gru_fre_input)[0]

        # time axis gru processing
        gru_time_input = gru_fre_output.view(B, T, F, C * 2).contiguous()
        gru_time_input = gru_time_input.view(B, T, C * F * 2).contiguous()
        gru_time_input = self.cln(gru_time_input.permute(0, 2, 1).contiguous())
        gru_time_input = self.BN(gru_time_input).permute(0, 2, 1).contiguous()
        gru_time_output = self.gru_time(gru_time_input)[0]

        mask = self.Sigmoid(self.linear(gru_time_output))

        # real mask
        mask = torch.nn.functional.pad(mask, [1, 0], 'constant', 0)
        enh_r = mask * input["noisy_r"][:, 0, :, :]
        enh_i = mask * input["noisy_i"][:, 0, :, :]

        out = {
            "enh_r": enh_r,
            "enh_i": enh_i,
        }

        return out
