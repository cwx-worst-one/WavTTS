'''
Description: Bytedance AILab SSP
Author: gongyuzhou@bytedance.com
Date: 2021-11-06 05:22:13
LastEditors: gongyuzhou@bytedance.com
LastEditTime: 2021-12-29 13:24:53
'''
import torch
from torch import nn
from core.models.layers.cnn import Conv1DBlockLD, Conv1DBlockLDP


class TcnLowDelaySub(nn.Module):
    """
    Temporal convolution network of aec.
    Input: w [B, n_feats, T]
    Output: mask [B, T, n_feats]
    """

    def __init__(self, args):
        super().__init__()
        self.n_b = args.n_b
        n_feats = args.n_feats
        dim_c = args.C
        dim_b = args.B
        dim_h = args.H
        dim_p = args.P
        self.proj1 = Conv1DBlockLDP(
            in_channels=n_feats, out_channels=dim_c, kernel_size=3, dilation=1, causal=False
        )
        self.proj2 = Conv1DBlockLDP(
            in_channels=dim_c, out_channels=dim_b, kernel_size=3, dilation=1, causal=False
        )
        self.blk = nn.ModuleList([])
        i = 0
        while i < self.n_b // 2:
            self.blk.append(
                Conv1DBlockLD(
                    in_channels=dim_b,
                    conv_channels=dim_h,
                    kernel_size=dim_p,
                    dilation=1,
                    causal=True,
                )
            )
            self.blk.append(
                Conv1DBlockLD(
                    in_channels=dim_b,
                    conv_channels=dim_h,
                    kernel_size=dim_p,
                    dilation=2,
                    causal=True,
                )
            )
            i += 1
        self.cproj1 = nn.Conv1d(dim_b, 129, 1)

    def forward(self, w):
        y = self.proj1(w.detach())
        y = self.proj2(y)

        for i in range(0, self.n_b):
            y = self.blk[i](y)

        e = self.cproj1(y)
        m = torch.sigmoid(e)
        mask = m.permute(0, 2, 1).contiguous()
        return mask


class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 1],
        stride=[2, 1],
        padding=[0, 0, 0, 0],
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.padding = padding

    def forward(self, x):
        x = torch.nn.functional.pad(
            x, [self.padding[2], self.padding[3], self.padding[0], self.padding[1]], 'constant', 0.0
        )
        x = self.conv1(x) * torch.sigmoid(self.conv2(x))
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Conv1dGRUMerge_opt_gated(nn.Module):
    def __init__(self, in_channels=256, mid_channels=256, kernel_size=3, stride=1, padding=[2, 0]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, stride=stride, padding=0)
        self.conv3 = nn.Conv1d(mid_channels, in_channels, 1, stride=1, padding=0)
        self.rnn = nn.GRU(mid_channels, mid_channels, batch_first=True)
        self.prelu1 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.bn = nn.BatchNorm1d(in_channels)
        self.padding = padding

    def forward(self, x):
        x_in = self.conv1(x)
        x_in = self.prelu1(x_in)
        x_rnn = x_in.permute(0, 2, 1)
        x_rnn = self.rnn(x_rnn)[0]
        x_rnn = x_rnn.permute(0, 2, 1)
        x_cnn = torch.nn.functional.pad(x_in, self.padding, 'constant', 0.0)
        x_cnn = self.conv2(x_cnn)

        x = x_rnn * x_cnn.sigmoid()
        x = self.conv3(x)
        x = self.bn(x)
        x = self.prelu3(x)
        return x


class AECSmallV42Align(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_in = nn.Linear(161, 64, bias=False)

        self.encoder = nn.ModuleList(
            [
                GatedConv2d(2, 4, kernel_size=[3, 3], padding=[1, 1, 1, 1], stride=[1, 1]),
                GatedConv2d(4, 4, kernel_size=[1, 3], padding=[0, 0, 1, 1], stride=[1, 2]),
                GatedConv2d(4, 8, kernel_size=[1, 3], padding=[0, 0, 1, 1], stride=[1, 1]),
                GatedConv2d(8, 8, kernel_size=[1, 3], padding=[0, 0, 1, 1], stride=[1, 2]),
            ]
        )

        self.rnn1 = Conv1dGRUMerge_opt_gated(128, 64)
        self.rnn2 = Conv1dGRUMerge_opt_gated(128, 64)

        self.fc = nn.Linear(128, 161)

    def forward(self, batch):
        fir_stft = batch["fir_stft"]
        ref_stft = batch["ref_stft"]

        fir_mag = fir_stft.pow(2).sum(-1).clamp(min=1e-12).log()
        ref_mag = ref_stft.pow(2).sum(-1).clamp(min=1e-12).log()
        x = torch.stack([fir_mag, ref_mag], dim=1)  # [b,c,t,f]
        x = self.fc_in(x)

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        B, C, T, F = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, -1, T)

        x = self.rnn1(x)
        x = self.rnn2(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x).sigmoid()

        ehn_stft = fir_stft * x.unsqueeze(-1)

        out = {
            "ehn_stft": ehn_stft,
            "fir_stft": fir_stft,
            "ref_stft": ref_stft,
            "mic_stft": batch["mic_stft"],
            "speech_stft": batch["speech_stft"],
        }
        return out
