import torch
import torchaudio
from einops import rearrange
from torch import nn
from transformers.models.t5.modeling_t5 import T5Config, T5EncoderModel


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, conv_layers, kernel_size=5, input_channel=1):
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


class ExampleConvT5(nn.Module):
    """CNN + T5 Encoder"""

    def __init__(self, hop_length=240, conv_dim=128, transformer_dim=512, n_mels=128):
        super().__init__()

        # preprocessing
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=2048, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # two convolution layers + one projection layer
        self.conv = Conv2dSubsampling(n_mels, transformer_dim, (conv_dim, conv_dim))

        # transformer
        config = T5Config()
        self.t5_encoder = T5EncoderModel(config)

    @torch.no_grad()
    def preprocessing(self, x):
        """log-mel spectrogram"""
        return self.amplitude_to_db.float()(self.melspec.float()(x.float())).half()

    def get_latent(self, x, layer_ix):
        x = self.preprocessing(x)
        x = self.conv(x)
        hidden_states = self.t5_encoder(inputs_embeds=x, output_hidden_states=True)[
            "hidden_states"
        ]
        return hidden_states[layer_ix]

    def forward(self, x):
        x = self.preprocessing(x)
        x = self.conv(x)
        return self.t5_encoder(inputs_embeds=x)["last_hidden_state"]
