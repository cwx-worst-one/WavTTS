import warnings

import numpy as np
import torch
import torchaudio
from einops import rearrange
from librosa import hz_to_note, midi_to_hz, note_to_midi
from torch import nn

from samantha.core import BaseStage


def initialize_filterbank(sample_rate, n_harmonic=1):
    # MIDI
    # lowest note
    low_midi = note_to_midi("C1")

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = high_midi - low_midi
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level


class ChromaticSTFT(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240, bw_Q=1.0):
        super(ChromaticSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7
        self.fft_bins = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        center_freq, self.level = initialize_filterbank(sample_rate)

        # Center frequncies to tensor
        self.f0 = torch.tensor(center_freq.astype("float32"))

        # Bandwidth parameters
        self.bw_Q = torch.tensor(np.array([bw_Q]).astype("float32"))

        self.zero = torch.zeros(1)

        # Chromatic FB
        self.register_buffer("chromatic_fb", self.get_chromatic_fb())

    def get_chromatic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)

        # triangle filter
        chromatic_spec = torch.matmul(
            spectrogram.transpose(1, 2), self.chromatic_fb
        ).transpose(1, 2)

        # amplitude to db
        chromatic_spec = self.amplitude_to_db(chromatic_spec)
        return chromatic_spec


class HarmonicSTFT(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240, n_harmonic=6):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7
        self.n_harmonic = n_harmonic
        self.fft_bins = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_freq, self.level = initialize_filterbank(
            sample_rate, n_harmonic=n_harmonic
        )

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_freq.astype("float32"))

        self.zero = torch.zeros(1)

        # Chromatic FB
        self.register_buffer("harmonic_fb", self.get_harmonic_fb())

    def get_harmonic_fb(self):
        # bandwidth
        bw = self.bw_alpha * self.f0 + self.bw_beta
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)

        # triangle filter
        harmonic_spec = torch.matmul(
            spectrogram.transpose(1, 2), self.harmonic_fb
        ).transpose(1, 2)

        # stack harmonics
        harmonic_spec = rearrange(
            harmonic_spec, "b (h f) l -> b h f l", h=self.n_harmonic
        )

        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(
        self, idim, odim, dropout_rate, conv_layers, kernel_size=5, input_channel=1
    ):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        assert len(conv_layers) == 2

        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channel,
                conv_layers[0],
                self.kernel_size,
                2,
                self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_layers[0],
                conv_layers[1],
                self.kernel_size,
                2,
                self.kernel_size // 2,
            ),
            nn.ReLU(),
        )
        self.conv_out_size = conv_layers[1] * (idim // 2 // 2)
        self.linear = nn.Linear(self.conv_out_size, odim)

    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, idim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.linear(x)
        return x


class ChromaticMRQ(nn.Module):
    """ChromaticMRQ"""

    def __init__(
        self,
        codebook_dim=16,
        codebook_size=8192,
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=6,
        transfer_depth=0,
        mask_hop=0.4,
        mask_prob=0.01,
        is_flash=True,
        global_mean=9.6,
        global_std=14.6,
        is_causal=False,
        is_torchscript=False,
        num_rq=8,
    ):
        super(ChromaticMRQ, self).__init__()

        # global variables
        self.global_mean = global_mean
        self.global_std = global_std
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_rq = num_rq
        self.codebook_size = codebook_size
        self.transfer_depth = transfer_depth  # only applies when it's for finetuning
        n_mels = 64

        # preprocessing
        self.melspec = HarmonicSTFT(
            sample_rate=24000, n_fft=2048, hop_length=hop_length
        )

        # two convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            n_mels, encoder_dim, 0.1, (conv_dim, conv_dim), input_channel=6
        )

        # T5
        from transformers.models.t5.modeling_t5 import T5Config, T5EncoderModel

        config = T5Config(
            d_model=encoder_dim,
            num_heads=16,
            num_layers=encoder_depth,
            torchscript=is_torchscript,
        )
        self.t5_encoder = T5EncoderModel(config)

    @torch.no_grad()
    def preprocessing(self, x, shift_ix):
        """log-mel spectrogram"""
        return (self.melspec.float()(x)).half()[:, :, shift_ix + 3 : shift_ix + 67, :]

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        out = self.t5_encoder(
            inputs_embeds=x, output_hidden_states=True, return_dict=True
        )
        last_emb = out["last_hidden_state"]
        return last_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        return (x - self.global_mean) / self.global_std

    def forward(self, x, pitch_shift=0):
        # forward
        x = self.preprocessing(x, pitch_shift)[:, :, :-1]
        x = self.normalize(x)
        return self.encoder(x)


class Frontend(BaseStage):
    def __init__(
        self,
        takes=["audio"],
        provides=["latent"],
        serialize_opts=None,
        layer_ix=12,
        is_update=False,
    ):
        super().__init__(takes, provides, serialize_opts)

        self.model = ChromaticMRQ()
        self.layer_ix = layer_ix
        self.is_update = is_update
        if not is_update:
            self.model.eval()

    def extract(self, audio, pitch_shift):
        """
        Input:
            audio (torch.FloatTensor): a batch of input audio (batch, length)
        Output:
            emb (torch.FloatTensor): a batch of output embeddings (batch, length, 1024)
        """
        if self.is_update:
            emb = self.model(audio, pitch_shift=pitch_shift)
        else:
            self.model.eval()
            with torch.no_grad():
                emb = self.model(audio, pitch_shift=pitch_shift)

        return emb

    def forward(self, x):
        """
        Input:
            x (dict): input dictionary with a key "audio". Input includes torch.FloatTensor(batch, length)
        Output:
            oup (dict): output dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, 1024, length)
        """
        # init dict
        oup = {}

        # BEST-RQ
        try:
            x = self.extract(x["audio"], x["pitch_shift"])
        except RuntimeError as e:
            if x["audio"].shape[0] == 1:
                warnings.warn("Torchscript only supports a batch size larger than 1.")
                x = self.extract(x["audio"].repeat(2, 1), x["pitch_shift"])
                x = x[:1]
        x = rearrange(x, "b t c -> b c t")

        # return dict
        oup["latent"] = x

        return oup
