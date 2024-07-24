import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from recipes.research.audio_codec.models.quantize import ResidualVectorQuantize
from samantha.nn.layers import SnakeBeta, WNConv1d, WNConvTranspose1d
from samantha.utils.logger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


def init_weights(m):
    pass
    # if isinstance(m, nn.Conv1d):
    #     nn.init.trunc_normal_(m.weight, std=0.02)

    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """This modulate adds a residual of x to the normalisation of x = (scale * x + shift) + x"""
    return x * (1 + scale) + shift


class FourierFeatureMapping(nn.Module):
    def __init__(self, n_channels: int, mapping_size: int, scale=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.mapping_size = mapping_size
        self.scale = scale

        # Create a matrix B of random frequencies drawn from a Gaussian distribution for each channel
        self.B = nn.Parameter(
            torch.normal(mean=0, std=scale, size=(n_channels, mapping_size // 2)),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier feature mapping across the 'time' dimension for each 'channel' and 'batch' instance
        # Using einsum to handle multi-dimensional input: batch, channels, time -> batch, channels, features
        projected = 2 * math.pi * torch.einsum("bct,cj->bjt", x, self.B)
        return torch.cat([projected.cos(), projected.sin()], dim=1)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.layers = nn.Sequential(
            SnakeBeta(in_channels),
            WNConv1d(
                in_channels, out_channels, kernel_size=7, dilation=dilation, padding=pad
            ),
            SnakeBeta(out_channels),
            WNConv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride

        if stride % 2 == 0:
            kernel_size = 2 * stride
        else:
            kernel_size = 2 * stride + 1

        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1),
            ResidualUnit(in_channels, in_channels, dilation=3),
            ResidualUnit(in_channels, in_channels, dilation=9),
            SnakeBeta(in_channels),
            WNConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        # x_len = x.shape[2] / self.stride
        x = self.layers(x)
        # print(x.shape[2], x_len)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: Tuple[int],
        strides: Tuple[int],
        output_dim: int,
    ):
        super().__init__()
        assert len(d_model) == len(strides)

        d_model = (d_model[0], *d_model)

        # fourier_dim = 32
        # self.fourier = FourierFeatureMapping(in_channels, fourier_dim, scale=1.0)

        self.layers = [WNConv1d(in_channels, d_model[0], kernel_size=7, padding=3)]

        for i in range(len(strides)):
            self.layers += [EncoderBlock(d_model[i], d_model[i + 1], stride=strides[i])]

        self.layers += [
            SnakeBeta(d_model[-1]),
            WNConv1d(d_model[-1], output_dim, kernel_size=3, padding=1),
        ]

        self.layers = nn.Sequential(*self.layers)
        self.enc_dim = d_model

    def _init_weights(self):
        logger.warning("Initializing Encoder weights...")
        self.apply(init_weights)

    def forward(self, x):
        # x = self.fourier(x) # TODO: residual?
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.stride = stride

        # if stride is uneven, add +1 to kernel_size to keep input/output the same
        if stride % 2 == 0:
            kernel_size = 2 * stride
        else:
            kernel_size = 2 * stride + 1

        self.layers = nn.Sequential(
            SnakeBeta(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, output_dim, dilation=1),
            ResidualUnit(output_dim, output_dim, dilation=3),
            ResidualUnit(output_dim, output_dim, dilation=9),
        )

    def forward(self, x):
        # x_len = x.shape[2] * self.stride
        x = self.layers(x)
        # print(x.shape[2], x_len)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        d_model: Tuple[int],
        strides: Tuple[int],
        act_out: nn.Module,
    ):
        super().__init__()
        assert len(strides) == len(d_model)
        d_model = (*d_model, d_model[-1])

        # Add first conv layer
        layers = [WNConv1d(input_channel, d_model[0], kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i in range(len(strides)):
            layers += [DecoderBlock(d_model[i], d_model[i + 1], strides[i])]

        # Add final conv layer
        layers += [
            SnakeBeta(d_model[-1]),
            WNConv1d(d_model[-1], output_channel, kernel_size=7, padding=3, bias=False),
        ]

        self.layers = nn.Sequential(*layers)
        self.act_out = act_out

    def _init_weights(self):
        logger.warning("Initializing Decoder weights...")
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.act_out(x)


class MoVQDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, z_q_channel: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            SnakeBeta(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )
        self.adaLN_mod = nn.Sequential(
            SnakeBeta(z_q_channel),
            WNConv1d(z_q_channel, 2 * output_dim, kernel_size=1, bias=True),
        )

    def forward(self, x, z_q: torch.Tensor):
        # TODO: should z_q.requires_grad be False?
        x = self.block(x)

        # post-adaLN
        z_q = F.interpolate(
            z_q, size=x.shape[2], mode="nearest"
        )  # TODO: linear # [B, output_dim, T_x]
        shift, scale = self.adaLN_mod(z_q).chunk(2, dim=1)  # [B, 2 * output_dim, T_x]
        x = modulate(x, shift, scale)  # [B, output_dim, T_x]
        return x


class MoVQDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        z_q_channel: int,
        channels,
        rates,
        act_out: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.z_q_channel = z_q_channel

        self.quant_conv = nn.Sequential(
            WNConv1d(input_channel, channels, kernel_size=7, padding=3)
        )

        blocks = []
        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            blocks.append(MoVQDecoderBlock(input_dim, z_q_channel, output_dim, stride))
        self.blocks: nn.ModuleList[MoVQDecoderBlock] = nn.ModuleList(blocks)

        # Add final conv layer
        self.final_adaLN_mod = nn.Sequential(
            SnakeBeta(z_q_channel),
            WNConv1d(z_q_channel, 2 * output_dim, kernel_size=1, bias=True),
        )
        self.conv_out = nn.Sequential(
            SnakeBeta(output_dim),
            WNConv1d(output_dim, output_channel, kernel_size=7, padding=3),
        )
        self.act_out = act_out

    def _init_weights(self):
        logger.warning("Initializing Decoder weights...")
        self.apply(init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_mod[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod[-1].bias, 0)
        nn.init.constant_(self.final_adaLN_mod[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_mod[-1].bias, 0)

    def forward(self, x: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        # if z_q.shape[1] < self.z_q_channel:
        #     # NOTE: this fills missing codebook latents with 0's
        #     fill_dim = self.z_q_channel - z_q.shape[1]
        #     z_q = F.pad(z_q, (0, 0, 0, fill_dim))

        x = self.quant_conv(x)

        for block in self.blocks:
            x = block.forward(x, z_q)

        # final adaLN
        z_q = F.interpolate(z_q, size=x.shape[2], mode="nearest")  # TODO: linear?
        shift, scale = self.final_adaLN_mod(z_q).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.conv_out(x)
        return self.act_out(x)


@dataclass
class UACConfig:
    sample_rate: int = 44100
    n_channels: int = 2
    encoder_dims: Tuple[int] = (128, 256, 512, 1024, 2048)
    encoder_strides: Tuple[int] = (2, 2, 7, 7, 9)
    vae_latent_dim: int = 64
    decoder_dims: int = (1024, 512, 256, 128, 64)
    decoder_strides: Tuple[int] = (9, 7, 7, 2, 2)
    n_codebooks: int = 16
    codebook_size: int = 8192
    codebook_dim: Union[int, list] = 8
    quantizer_dropout: float = 0.5
    vae_beta: float = 1e-4
    decoder_act_out: nn.Module = nn.Identity()  # nn.Tanh() # nn.Identity


class UACResult(NamedTuple):
    audio: torch.Tensor
    latents: torch.Tensor
    commitment_loss: Union[torch.Tensor, float]
    codebook_loss: Union[torch.Tensor, float]
    codes: Optional[torch.Tensor] = None
    pre_latents: Optional[torch.Tensor] = None
    z_q: Optional[torch.Tensor] = None
    z_q_masked: Optional[torch.Tensor] = None


class UACBase(nn.Module):

    def __init__(self, config: UACConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.hop_length = math.prod(config.encoder_strides)
        self.frame_rate = config.sample_rate / self.hop_length
        self.bit_rate = (
            self.frame_rate
            * (config.n_codebooks * math.log2(config.codebook_size))
            / 1000
        )

    def preprocess(
        self, audio: torch.Tensor, sample_rate: int, force_hop_length: bool = True
    ) -> Tuple[torch.Tensor, int]:
        assert sample_rate == self.sample_rate

        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length

        if force_hop_length:
            assert (
                right_pad == 0
            ), f"Make sure that the audio has a length multiple of {self.hop_length}"

        audio = nn.functional.pad(audio, (0, right_pad))
        return audio, right_pad

    @staticmethod
    def align_codes_after_padding(
        codes: torch.Tensor, pad_len: int, sample_rate: int, frame_rate: float
    ) -> torch.Tensor:
        pad_len_sec = pad_len / sample_rate
        pad_len_frames = math.ceil(pad_len_sec * frame_rate)
        if pad_len_frames > 0:
            codes = codes[..., :-pad_len_frames]
        return codes

    def to_torchscript(self):
        self = self.apply(
            lambda m: remove_parametrizations(m, "weight") if is_parametrized(m) else m
        )
        self = self.eval()
        return torch.jit.script(self.cpu())


class UACVQ(UACBase):
    def __init__(self, config: UACConfig):
        super().__init__(config)
        self.encoder = Encoder(
            config.n_channels,
            config.encoder_dims,
            config.encoder_strides,
            config.encoder_latent_dim,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=config.encoder_latent_dim,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
        )

        # self.decoder = MoVQDecoder(
        #     config.latent_dim,
        #     config.n_channels,
        #     self.quantizer.total_codebook_dim,
        #     config.decoder_dim,
        #     config.decoder_rates,
        #     config.decoder_act_out,
        # )
        self.decoder = Decoder(
            config.encoder_latent_dim,
            config.n_channels,
            config.decoder_dims,
            config.decoder_strides,
            config.decoder_act_out,
        )
        self.init_weights()

    def init_weights(self):
        logger.warning("Initializing codec weights...")
        self.encoder._init_weights()
        self.quantizer.apply(init_weights)
        self.decoder._init_weights()

    @property
    @torch.jit.export
    def latent_dim(self) -> int:
        return self.quantizer.total_codebook_dim

    def encode(self, audio_data: torch.Tensor, n_quantizers: Optional[int] = None):
        z = self.encoder(audio_data)

        latents, codes, pre_latents, z_q, z_q_masked, commitment_loss, codebook_loss = (
            self.quantizer.forward(z, n_quantizers)
        )
        return (
            latents,
            codes,
            pre_latents,
            z_q,
            z_q_masked,
            commitment_loss,
            codebook_loss,
        )

    def decode(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        # return self.decoder(z, z_q)
        return self.decoder(z)

    def forward(
        self, audio: torch.Tensor, sample_rate: int, n_quantizers: Optional[int] = None
    ) -> UACResult:
        length = audio.shape[-1]
        audio, pad_len = self.preprocess(audio, sample_rate)
        latents, codes, pre_latents, z_q, z_q_masked, commitment_loss, codebook_loss = (
            self.encode(audio, n_quantizers)
        )

        x = self.decode(latents, z_q_masked)

        if pad_len:
            codes = self.align_codes_after_padding(
                codes, pad_len, sample_rate, self.frame_rate
            )
            x = x[..., :-pad_len]

        assert x.shape[2] == length
        return UACResult(
            audio=x,
            latents=latents,
            codes=codes,
            pre_latents=pre_latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            z_q=z_q,
            z_q_masked=z_q_masked,
        )

    @torch.jit.export
    def get_z(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio, pad_len = self.preprocess(audio, sample_rate)
        h = self.encoder(audio)
        z_e = self.quantizer.get_z(h)
        z_e = self.align_codes_after_padding(z_e, pad_len, sample_rate, self.frame_rate)
        return z_e

    @torch.jit.export
    def decode_z(self, z_e: torch.Tensor):
        latents, z_q = self.quantizer.latents_from_pre_z(z_e)
        return self.decode(latents, z_q)

    def get_codes(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ) -> torch.Tensor:
        audio, pad_len = self.preprocess(audio, sample_rate)
        latents, codes, pre_latents, z_q, z_q_masked, commitment_loss, codebook_loss = (
            self.encode(audio, n_quantizers)
        )
        codes = self.align_codes_after_padding(
            codes, pad_len, sample_rate, self.frame_rate
        )
        return codes

    def z_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.z_from_codes(codes)

    def decode_from_codes(self, codes: torch.Tensor):
        latents, pre_latents, _ = self.quantizer.from_codes(codes)
        x = self.decode(latents)
        return {
            "audio": x,
            "latents": latents,
            "pre_latents": pre_latents,
            "codes": codes,
        }


def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4  # NOTE: tie to beta?
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    # NOTE: the batch-wise sum is done with dim=[1, 2], which is mathetmatically correct
    kl = (mean * mean + var - logvar - 1).sum(dim=[1, 2]).mean()

    # TODO review:
    # kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class UACVAE(UACBase):
    def __init__(self, config: UACConfig):
        super().__init__(config)
        self.beta = config.vae_beta
        self.vae_latent_dim = config.vae_latent_dim

        self.encoder = Encoder(
            config.n_channels,
            config.encoder_dims,
            config.encoder_strides,
            self.vae_latent_dim * 2,
        )

        self.decoder = Decoder(
            self.vae_latent_dim,
            config.n_channels,
            config.decoder_dims,
            config.decoder_strides,
            config.decoder_act_out,
        )
        self.init_weights()

    @property
    @torch.jit.export
    def latent_dim(self) -> int:
        return self.vae_latent_dim

    def init_weights(self):
        logger.warning("Initializing codec weights...")
        self.encoder._init_weights()
        self.decoder._init_weights()

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Encode given audio data with a VAE and return latents"""
        mu_scale = self.encoder(audio)

        mean, scale = mu_scale.chunk(2, dim=1)
        z, codebook_loss = vae_sample(mean, scale)

        codebook_loss = self.beta * codebook_loss
        return z, codebook_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, audio: torch.Tensor, sample_rate: int) -> UACResult:
        length = audio.shape[-1]
        audio, pad_len = self.preprocess(audio, sample_rate)

        z, codebook_loss = self.encode(audio)
        x = self.decode(z)

        if pad_len:
            x = x[..., :-pad_len]

        assert x.shape[2] == length
        return UACResult(
            audio=x,
            latents=z,
            codes=None,
            pre_latents=None,
            commitment_loss=0.0,
            codebook_loss=codebook_loss,
        )

    @torch.jit.export
    def get_z(self, audio: torch.Tensor, sample_rate: int):
        audio, pad_len = self.preprocess(audio, sample_rate, force_hop_length=False)
        z, codebook_loss = self.encode(audio)

        if pad_len:
            z = self.align_codes_after_padding(z, pad_len, sample_rate, self.frame_rate)
        return z

    @torch.jit.export
    def decode_z(self, z: torch.Tensor):
        return self.decode(z)


if __name__ == "__main__":
    from functools import partial

    import numpy as np

    duration = 0.76
    config = UACConfig()
    model = UACVAE(config).to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = math.floor(duration * config.sample_rate)
    x = torch.randn(1, config.n_channels, length).to("cpu")
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    result = model.forward(x, config.sample_rate)
    out = result.audio
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print(f"z shape: {result.latents.shape}")

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    # x = torch.randn(1, 1, 44100 * 60)
    # result = model.forward(x, config.sample_rate)
    # model.decompress(model.compress(x, verbose=True), verbose=True)
