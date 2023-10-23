import librosa
import numpy as np
import torch
import torchaudio
from einops import rearrange
from torch import einsum, nn


# Pre-processing Harmonic representation
def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale, low_note="C1"):
    # MIDI
    # lowest note
    base_note = librosa.core.note_to_midi("C1")
    low_midi = librosa.core.note_to_midi(low_note)

    # highest note
    high_note = librosa.core.hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = librosa.core.note_to_midi(high_note) + (low_midi - base_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level


class HarmonicSTFT(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=513,
        win_length=None,
        hop_length=None,
        power=2,
        normalized=False,
        n_harmonic=2,
        semitone_scale=2,
        bw_Q=1.0,
        learn_bw=None,
    ):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            window_fn=torch.hann_window,
            power=power,
            normalized=normalized,
            wkwargs=None,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale
        )

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_hz.astype("float32"))

        # Bandwidth parameters
        if learn_bw == "only_Q":
            self.bw_Q = nn.Parameter(torch.tensor(np.array([bw_Q]).astype("float32")))
        elif learn_bw == "fix":
            self.bw_Q = torch.tensor(np.array([bw_Q]).astype("float32"))

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def to_device(self, device, n_bins):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)
        # fft bins
        self.fft_bins = torch.linspace(0, self.sample_rate // 2, n_bins).to(device)
        self.zero = torch.zeros(1).to(device)

    def forward(self, waveform):
        """
        :param waveform: array (n_batch, n_sample)
        :return: array  (n_batch, n_harmonics, n_frame)
        """

        # stft
        spectrogram = self.spec(waveform)

        # to device
        self.to_device(waveform.device, spectrogram.size(1))

        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = torch.matmul(
            spectrogram.transpose(1, 2), harmonic_fb
        ).transpose(1, 2)

        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, _, length = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(b, self.n_harmonic, self.level, length)

        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec


# Output classifier
class DensNet(nn.Module):
    def __init__(self, channels=20, oup_dim=1):
        super(DensNet, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, oup_dim)

    def forward(self, x):
        output = self.norm(x)
        output = self.linear(output)
        return output


class Classifier(nn.Module):
    def __init__(self, channels=20, oup_dim=1, pools=[1]):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        for i, pool in enumerate(pools):
            layer = nn.ModuleList()
            layer.append(nn.MaxPool1d(pool))
            if i == (len(pools) - 1):
                layer.append(DensNet(channels, oup_dim))
            else:
                layer.append(DensNet(channels, channels))
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            if len(x.shape) == 3:
                x = layer[0](x.permute(0, 2, 1)).permute(0, 2, 1)
                x = layer[1](x)
            else:
                x = layer[1](x)

        return x


class Attention(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, manual_att=None):
        _, _, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if manual_att is not None:
            attn = torch.zeros(sim.size())
            attn[..., manual_att] = 1
            attn = attn.to(q.device)
        else:
            attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out), attn


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Pre-processing for transformer
class Preprocess(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        n_harmonic=6,
        semitone_scale=1,
        learn_bw="only_Q",
        hop_length=512,
        input_feature="hcqt",
    ):
        super(Preprocess, self).__init__()
        self.input_feature = input_feature
        if input_feature == "mel":
            self.audio_module = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=None,
                hop_length=hop_length,
                pad=0,
                window_fn=torch.hann_window,
                power=2,
                normalized=False,
                wkwargs=None,
            )
            self.mel_scale = torchaudio.transforms.MelScale(
                n_mels=81,
                sample_rate=sample_rate,
                f_min=30,
                f_max=17000,
                n_stft=n_fft // 2 + 1,
            )
            self.input_dim = 1
            self.input_freq = 128
        elif input_feature == "hcqt":
            self.hstft = HarmonicSTFT(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_harmonic=n_harmonic,
                semitone_scale=semitone_scale,
                learn_bw=learn_bw,
                hop_length=hop_length,
            )
            self.hstft_bn = nn.BatchNorm2d(n_harmonic)
            self.input_dim = 6
            self.input_freq = 64
        else:
            raise Exception(f"input feature {input_feature} not suqqport")

    def forward(self, x, aug_hop_size):
        """
        :param x: array (n_batch, n_sample)
        :param aug_hop_size: int
        :return: array (n_batch, n_channel, n_frame)
        """

        if self.input_feature == "mel":
            self.audio_module.hop_length = aug_hop_size
            x = 10 * torch.log10(1e-10 + self.mel_scale(self.audio_module(x)))[
                ..., :-1
            ].unsqueeze(1)
        elif self.input_feature == "hcqt":
            self.hstft.spec.hop_length = aug_hop_size
            x = self.hstft_bn(self.hstft(x))[..., :-1]
        return x
