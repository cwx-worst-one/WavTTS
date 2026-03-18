from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
import torchaudio


class AFF(nn.Module):
    def __init__(self, channels: int = 64, r: int = 4):
        super().__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, ds_y: torch.Tensor) -> torch.Tensor:
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        return x * x_att + ds_y * (2.0 - x_att)


class TSTP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooling_mean = x.mean(dim=-1).flatten(start_dim=1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8).flatten(start_dim=1)
        return torch.cat((pooling_mean, pooling_std), dim=1)


class ReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super().__init__(0, 20, inplace)


class BasicBlockERes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, planes: int, stride: int = 1, baseWidth: int = 32, scale: int = 2):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, dim=1)

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)

        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.shortcut(residual))
        return out


class BasicBlockERes2NetDiffAFF(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, planes: int, stride: int = 1, baseWidth: int = 32, scale: int = 2):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        fuse_models = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        for _ in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, dim=1)

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.relu(self.bns[i](self.convs[i](sp)))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)

        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.shortcut(residual))
        return out


class ERes2Net(nn.Module):
    def __init__(
        self,
        block=BasicBlockERes2Net,
        block_fuse=BasicBlockERes2NetDiffAFF,
        num_blocks: list[int] = [3, 4, 6, 3],
        m_channels: int = 32,
        feat_dim: int = 80,
        embedding_size: int = 192,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.stats_dim = int(feat_dim / 8) * m_channels * 8

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2)

        self.layer1_downsample = nn.Conv2d(m_channels * 2, m_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4, m_channels * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.fuse_mode12 = AFF(channels=m_channels * 4)
        self.fuse_mode123 = AFF(channels=m_channels * 8)
        self.fuse_mode1234 = AFF(channels=m_channels * 16)

        self.pool = TSTP()
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * 2, embedding_size)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_value in strides:
            layers.append(block(self.in_planes, planes, stride_value))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.permute(0, 2, 1).unsqueeze(1)  # [B,1,F,T]
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        fuse_out12 = self.fuse_mode12(out2, self.layer1_downsample(out1))
        out3 = self.layer3(out2)
        fuse_out123 = self.fuse_mode123(out3, self.layer2_downsample(fuse_out12))
        out4 = self.layer4(out3)
        fuse_out1234 = self.fuse_mode1234(out4, self.layer3_downsample(fuse_out123))
        stats = self.pool(fuse_out1234)
        return self.seg_1(stats)


class FBank:
    def __init__(self, n_mels: int, sample_rate: int, mean_nor: bool = True):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav: torch.Tensor, dither: float = 0.0) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        feat = Kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate,
            dither=dither,
        )
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class ERes2NetFeatureLoss(nn.Module):
    def __init__(
        self,
        model_path: str,
        source_sample_rate: int = 24000,
        target_sample_rate: int = 16000,
        feat_dim: int = 80,
        embedding_size: int = 192,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.feat_extractor = FBank(n_mels=feat_dim, sample_rate=target_sample_rate, mean_nor=True)

        ckpt_path = Path(model_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "pretrained_eres2net.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ERes2Net checkpoint not found: {ckpt_path}")

        self.speaker_model = ERes2Net(feat_dim=feat_dim, embedding_size=embedding_size)
        try:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location="cpu")
        self.speaker_model.load_state_dict(state, strict=True)
        self.speaker_model.eval()
        for p in self.speaker_model.parameters():
            p.requires_grad = False

        self.resampler = None
        if self.source_sample_rate != self.target_sample_rate:
            self.resampler = nn.Sequential(
                torchaudio.transforms.Resample(
                    orig_freq=self.source_sample_rate,
                    new_freq=self.target_sample_rate,
                )
            )

    def _to_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        feats = [self.feat_extractor(sample) for sample in wav]  # [T, F] per sample
        feats = torch.stack(feats, dim=0)  # [B, T, F]
        feats = feats.to(wav.device)
        return self.speaker_model(feats)

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        # x_pred, x_true: [B, T] or [B, 1, T]
        if x_pred.ndim == 3:
            x_pred = x_pred.squeeze(1)
        if x_true.ndim == 3:
            x_true = x_true.squeeze(1)

        if self.resampler is not None:
            x_pred = self.resampler(x_pred)
            x_true = self.resampler(x_true)

        with torch.no_grad():
            emb_true = self._to_embedding(x_true.float())
        emb_pred = self._to_embedding(x_pred.float())

        cos_sim = F.cosine_similarity(emb_pred, emb_true, dim=-1)
        return (1.0 - cos_sim.mean()) * self.weight
