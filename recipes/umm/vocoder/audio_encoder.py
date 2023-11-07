import torch

from torch import nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import MelSpectrogram, Spectrogram

from recipes.umm.vocoder.mel_processing import spectrogram_torch, mel_spectrogram_torch
from recipes.umm.vocoder.bigvgan import CausalConv1d
from recipes.umm.vocoder.flow import WN


class Downsample(nn.Module):

    def __init__(self, down_scale, in_channels, out_channels):
        super().__init__()
        self.down_scale = down_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * down_scale - 1,
                stride=down_scale,
                padding=1 * down_scale - 1,
            )
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, bias=False):
        super().__init__()
        kernel_size1 = kernel_size
        kernel_size2 = max(1, kernel_size - 2)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=bias))
        self.conv2 = weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=bias))
        self.act = torch.nn.GELU()

    def forward(self, x):
        xt = self.act(x)
        xt = self.conv1(xt)
        xt = self.act(xt)
        xt = self.conv2(xt)
        x = x + xt
        return x


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


class AudioEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        ch = hidden_channels
        self.audio_downs = nn.Sequential(
            weight_norm(nn.Conv1d(1, ch, kernel_size=55, padding=27, bias=False)),
            ResBlock(in_channels=ch, kernel_size=55),
            ResBlock(in_channels=ch, kernel_size=55),
            # down
            Downsample(down_scale=8, in_channels=ch, out_channels=ch),
            ResBlock(in_channels=ch, kernel_size=7),
            # down
            Downsample(down_scale=6, in_channels=ch, out_channels=ch),
            ResBlock(in_channels=ch, kernel_size=5),
            # down
            Downsample(down_scale=5, in_channels=ch, out_channels=ch),
            ResBlock(in_channels=ch, kernel_size=3),
            # down
            Downsample(down_scale=4, in_channels=ch, out_channels=ch),
            ResBlock(in_channels=ch, kernel_size=1),
        )

    def forward(self, wav):
        # wav: [b, t]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            feature = self.audio_downs(wav)
        return feature.float()


class SpecEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_pre = weight_norm(CausalConv1d(1025, hidden_channels, kernel_size=3, padding=1))
        self.conv1 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        self.conv2 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0))
        self.conv3 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        self.conv4 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0))
        self.act = torch.nn.GELU()

    def forward(self, x):
        spec = spectrogram_torch(x, n_fft=2048, sampling_rate=24000, hop_size=960, win_size=1920)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            x = self.conv_pre(spec)
            # resnet 1
            xt = self.act(x)
            xt = self.conv1(xt)
            xt = self.act(xt)
            xt = self.conv2(xt)
            x = x + xt
            # resnet 2
            xt = self.act(x)
            xt = self.conv3(xt)
            xt = self.act(xt)
            xt = self.conv4(xt)
            x = x + xt
        return x.float(), spec


class MelEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_pre = weight_norm(CausalConv1d(120, hidden_channels, kernel_size=3, padding=1))
        self.conv1 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        self.conv2 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0))
        self.conv3 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        self.conv4 = weight_norm(CausalConv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0))
        self.act = torch.nn.GELU()

    def forward(self, x):
        mel = mel_spectrogram_torch(x, n_fft=2048, num_mels=120, sampling_rate=24000, hop_size=960, win_size=1920, fmin=0, fmax=12000)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            x = self.conv_pre(mel)
            # resnet 1
            xt = self.act(x)
            xt = self.conv1(xt)
            xt = self.act(xt)
            xt = self.conv2(xt)
            x = x + xt
            # resnet 2
            xt = self.act(x)
            xt = self.conv3(xt)
            xt = self.act(xt)
            xt = self.conv4(xt)
            x = x + xt
        return x.float(), mel


class MelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=4, p_dropout=0):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(in_channels, hidden_channels, 1))
        self.net = WN(hidden_channels=hidden_channels,
                      kernel_size=5,
                      dilation_rate=1,
                      n_layers=n_layers,
                      p_dropout=p_dropout)
        self.out = weight_norm(nn.Conv1d(hidden_channels, out_channels, 1, bias=False))
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.pre(x)
        x = self.act(x)
        x = self.net(x)
        x = self.out(x)
        return x

