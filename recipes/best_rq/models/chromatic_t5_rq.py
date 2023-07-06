import numpy as np
import torch
import torchaudio
from einops import rearrange
from librosa import hz_to_note, midi_to_hz, note_to_midi
from torch import einsum, nn


class RandomProjectionQuantizer(nn.Module):
    """Random projection and codebook lookup module"""

    def __init__(self, input_dim, codebook_dim, codebook_size, seed=142):
        super().__init__()

        # random seed
        torch.manual_seed(seed)

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

        # input norm
        self.input_norm = nn.LayerNorm(input_dim)

    def codebook_lookup(self, x):
        # reshape
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)

        # reshape
        xq = rearrange(nearest_indices, "(b n) -> b n", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x):
        # always eval
        self.eval()

        # input norm
        x = self.input_norm(x)

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq


def initialize_filterbank(sample_rate):
    # MIDI
    # lowest note
    low_midi = note_to_midi("C1")

    # highest note
    high_note = hz_to_note(sample_rate / 2)
    high_midi = note_to_midi(high_note)

    # number of scales
    level = high_midi - low_midi
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    return hz, level


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
        # self.chromatic_fb = self.get_chromatic_fb()
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


class BEST_RQ(nn.Module):
    """BEST-RQ"""

    def __init__(
        self,
        codebook_dim=16,
        codebook_size=8192,
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=24,
        transfer_depth=0,
        mask_hop=0.4,
        mask_prob=0.01,
        is_flash=True,
        global_mean=9.6,
        global_std=14.6,
        is_causal=False,
        is_torchscript=False,
    ):
        super().__init__()

        # global variables
        self.global_mean = global_mean
        self.global_std = global_std
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.transfer_depth = transfer_depth  # only applies when it's for finetuning
        n_mels = 100

        # random quantizer
        self.quantizer = RandomProjectionQuantizer(
            n_mels * 4, codebook_dim, codebook_size
        )

        # preprocessing
        self.melspec = ChromaticSTFT(
            sample_rate=24000, n_fft=2048, hop_length=hop_length
        )

        # input norm
        self.input_norm = nn.LayerNorm(n_mels)

        # two convolution layers + one projection layer
        self.conv = Conv2dSubsampling(n_mels, encoder_dim, 0.1, (conv_dim, conv_dim))

        # T5
        if is_flash:
            from recipes.best_rq.models.flash_t5 import T5Config, T5EncoderModel
        else:
            from transformers.models.t5.modeling_t5 import T5Config, T5EncoderModel
        config = T5Config.from_pretrained("t5-large", torchscript=is_torchscript)
        self.t5_encoder = T5EncoderModel(config)

        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

    def masking(self, x):
        """random masking of 400ms with 0.01 probability"""
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop)

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x):
        """log-mel spectrogram"""
        return (self.melspec.float()(x)).half()[:, :100, :]

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        out = self.t5_encoder(
            inputs_embeds=x, output_hidden_states=True, return_dict=True
        )
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        return logits, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        return (x - self.global_mean) / self.global_std

    def get_latent(self, x, layer_ix=12):
        x = self.preprocessing(x)[:, :, :-1]
        x = self.normalize(x)
        x = self.conv(x)
        hidden_states = self.t5_encoder(
            inputs_embeds=x, output_hidden_states=True, return_dict=True
        )["hidden_states"]
        emb = hidden_states[layer_ix]
        return emb

    def forward(self, x, is_pretrain=False):
        if is_pretrain:
            # get target tokens
            raw_x = self.preprocessing(x)[:, :, :-1]
            raw_x = self.normalize(raw_x)
            raw_x = rearrange(raw_x, "b f (t s) -> b t (s f)", s=4)
            target_tokens = self.quantizer(raw_x)

            # masking
            x, masked_indices = self.masking(x)

        # forward
        x = self.preprocessing(x)[:, :, :-1]
        x = self.normalize(x)
        logits, hidden_emb = self.encoder(x)

        # return logits and loss
        if is_pretrain:
            masked_logits = logits[tuple(masked_indices.t())]
            masked_tokens = target_tokens[tuple(masked_indices.t())]
            loss = self.loss(masked_logits, masked_tokens)
            print(loss)
            acc = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
            return logits, loss, acc
        else:
            return logits, hidden_emb
