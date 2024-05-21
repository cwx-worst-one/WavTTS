import logging
import math
from functools import lru_cache

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from samantha.utils.infer_utils import mel_spectrogram_torch, spectrogram_torch

logger = logging.getLogger(__name__)


# Keep track of 10 different messages and then warn again
@lru_cache(10)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


def preprocess_audio(audio_bin, sample_rate, resampler, device, freq=40, *_, **__):
    wav, sr = librosa.load(audio_bin, sr=None)
    if wav.size == 0:
        return None, None
    audio_dur = wav.shape[0] / float(sr)
    wav = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    if sr != sample_rate:
        if sr not in resampler:
            resampler[sr] = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        warn_once(logger, f"resample audio from {sr=} to {sample_rate=}")
        wav = resampler[sr](wav)
    scale = max(0.001, torch.max(torch.abs(wav)).item())
    wav = wav / scale * 0.95
    wav = wav.unsqueeze(0).float()
    pad_mod = sample_rate // freq
    wav = F.pad(
        wav,
        (0, (wav.size(-1) // pad_mod + 1) * pad_mod - wav.size(-1), 0, 0, 0, 0),
        value=0.0,
    )
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, device, *_, **__):
    wvae_encoder = model["wvae_encoder"]
    mel_norm = model["mel_norm"]
    encoder = model["encoder"]
    fail_cnt = 0
    for wav in batch:
        try:
            yield get_speechcaption_spk_embedding(wav, wvae_encoder, mel_norm, encoder).cpu().float().view(-1).numpy()
        except Exception as e:
            logger.warning("process sample failed, skip this one", exc_info=e)
            fail_cnt += 1
            if fail_cnt == len(batch):
                raise e
            yield None


def load_model(device, model_path, *_, **__):
    # diffusion_v3_streaming_path='./yyh_resource/diffusion_v3_streaming/prompt_encoder.pt'
    # wvae_encoder_path='./yyh_resource/wvae_3.1/wavevae_encoder_z1_%d.pt'
    if isinstance(device, str):
        rank = int(device[-1])
    else:
        rank = device.index
    wvae_encoder_path = f"{model_path}/wvae_3.1/wavevae_encoder_%d.pt"
    encoder_path = f"{model_path}/diffusion_v3_streaming/prompt_encoder.pt"
    wvae_encoder = torch.jit.load(wvae_encoder_path % rank).to(device).eval()
    mel_norm = MelNorm(mean=0, std=2)
    diffusion_v3_streaming_spk_encoder = nn.Sequential(
        ECAPA_TDNN_GN(64, 1024, 512), nn.Softsign()
    )
    state_dict = torch.load(encoder_path, map_location="cpu")
    diffusion_v3_streaming_spk_encoder.load_state_dict(state_dict, strict=True)
    return {
        "wvae_encoder": wvae_encoder,
        "mel_norm": mel_norm,
        "encoder": diffusion_v3_streaming_spk_encoder.to(device).eval(),
    }


def get_speechcaption_spk_embedding(
    wav,
    wvae_encoder,
    mel_norm_cls,
    diffusion_v3_streaming_spk_encoder,
    debug=False,
    *_,
    **__,
):
    spec = spectrogram_torch(wav.squeeze(1), 2048, 24000, 300, 1200)
    mel_spec = mel_spectrogram_torch(
        y=wav.squeeze(1),
        n_fft=2048,
        num_mels=80,
        sampling_rate=24000,
        hop_size=300,
        win_size=1200,
        fmin=0.0,
        fmax=None,
    )

    _, m, logs = wvae_encoder(wav, spec, mel_spec)  # for wvae v3.1
    m = m.transpose(2, 1).squeeze(0)
    logs = logs.transpose(2, 1).squeeze(0)
    if debug:
        crop_bn = m + torch.exp(logs)
    else:
        crop_bn = m + torch.randn_like(m) * torch.exp(logs)
    speech_caption_spk_bn_gt = mel_norm_cls.norm_mel(crop_bn.unsqueeze(0)).transpose(
        1, 2
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        speech_caption_spk_embedding_gt = diffusion_v3_streaming_spk_encoder(
            speech_caption_spk_bn_gt
        )
    return speech_caption_spk_embedding_gt


class MelNorm(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def denorm_mel(self, mel):
        return (mel * self.std) + self.mean

    def norm_mel(self, mel):
        return (mel - self.mean) / self.std


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.GroupNorm(4, width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.GroupNorm(4, width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.GroupNorm(4, planes)
        self.relu = nn.GELU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

    # Replace BN with GN


class ECAPA_TDNN_GN(nn.Module):
    def __init__(self, in_dim, C, out_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.GELU()
        self.bn1 = nn.GroupNorm(4, C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(4, 256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.GroupNorm(8, 3072)
        self.fc6 = nn.Linear(3072, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.std(x, dim=2, keepdim=True).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        return x
