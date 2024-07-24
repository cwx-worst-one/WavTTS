import gc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from typing_extensions import Self

# from recipes.research.audio_codec.models.encodec_discriminator import (
#     EncodecDiscriminator,
# )
from recipes.research.audio_codec.gan_loss import GANLoss
from recipes.research.audio_codec.models.discriminator import (
    Discriminator,
    DiscriminatorConfig,
)
from recipes.research.audio_codec.models.spectral_loss import (
    MultiResolutionSTFTLoss,
    SumAndDifferenceSTFTLoss,
)
from recipes.research.audio_codec.models.uac import UACVAE, UACVQ, UACConfig, UACResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.optim.lr_scheduler.inverse_lr import InverseLR
from samantha.utils.logger import RankedLogger

logger = RankedLogger()


@dataclass
class AudioCodecConfig:
    sample_rate: int = 44100

    # generator
    bottleneck: str = "vae"  # rvq
    n_channels: int = 2
    encoder_dims: Tuple[int] = (128, 256, 512, 1024, 2048)
    encoder_strides: Tuple[int] = (2, 4, 4, 8, 8)
    vae_latent_dim: int = 64
    decoder_dims: int = (2048, 1024, 512, 256, 128)
    decoder_strides: Tuple[int] = (8, 8, 4, 4, 2)
    n_codebooks: int = 24
    codebook_size: int = 8192
    codebook_dim: Union[int, list] = 4
    quantizer_dropout: float = 0.5
    vae_beta: float = 1e-5
    decoder_act_out: nn.Module = nn.Identity()
    generator_warmup_steps: int = 0

    # discriminator
    disc_n_filters: int = 32
    disc_n_ffts: Tuple[int] = (2048, 1024, 512, 256, 128)
    disc_hop_lengths: Tuple[int] = (512, 256, 128, 64, 32)
    disc_window_lengths: Tuple[int] = (2048, 1024, 512, 256, 128)

    # loss weights:
    stft_loss: float = 1.0
    waveform_loss: float = 0.0
    feat_loss: float = 5.0
    disc_loss: float = 1.0
    adverserial_loss: float = 0.1
    codebook_loss: float = 1.0
    commitment_loss: float = 0.25

    # optimization
    gen_learning_rate: float = 5.0e-5
    disc_learning_rate: float = 5.0e-5
    betas: Tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 1e-3


@dataclass
class AudioCodecResult:
    audio: torch.Tensor
    latents: torch.Tensor
    codes: torch.Tensor
    vq_commitment_loss: torch.Tensor
    vq_codebook_loss: torch.Tensor
    loss: Optional[LossDict] = None
    audio_y: Optional[torch.Tensor] = None


class AudioCodec(DefaultTrainingBaseModule):
    def __init__(self, config: AudioCodecConfig):  # vae
        super().__init__()
        self.config = config
        self.automatic_optimization = False

        codec_config = UACConfig(
            n_channels=config.n_channels,
            encoder_dims=config.encoder_dims,
            encoder_strides=config.encoder_strides,
            vae_latent_dim=config.vae_latent_dim,
            decoder_dims=config.decoder_dims,
            decoder_strides=config.decoder_strides,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
            decoder_act_out=config.decoder_act_out,
            vae_beta=config.vae_beta,
        )

        logger.info(config)
        logger.info(codec_config)

        if config.bottleneck == "vae":
            self.generator = UACVAE(codec_config)
        elif config.bottleneck == "rvq":
            self.generator = UACVQ(codec_config)
        else:
            raise NotImplementedError(f"{config.bottleneck} is not implemented")

        logger.info(f"Bottleneck: {config.bottleneck}")

        # self.discriminator = EncodecDiscriminator(
        #     filters=config.disc_n_filters,
        #     in_channels=config.n_channels,
        #     out_channels=1, # TODO
        #     n_ffts=config.disc_n_ffts,
        #     hop_lengths=config.disc_hop_lengths,
        #     win_lengths=config.disc_window_lengths,
        # )

        disc_config = DiscriminatorConfig(
            sample_rate=config.sample_rate,
            n_channels=config.n_channels,
            n_filters=config.disc_n_filters,
            window_lengths=config.disc_window_lengths,
        )
        self.discriminator = Discriminator(disc_config)

        self.sd_stft = SumAndDifferenceSTFTLoss(
            fft_sizes=[2048, 1024, 512, 256, 128, 64, 32],
            hop_sizes=[512, 256, 128, 64, 32, 16, 8],
            win_lengths=[2048, 1024, 512, 256, 128, 64, 32],
            perceptual_weighting=True,
            sample_rate=config.sample_rate,
        )
        self.lr_stft = MultiResolutionSTFTLoss(
            fft_sizes=[2048, 1024, 512, 256, 128, 64, 32],
            hop_sizes=[512, 256, 128, 64, 32, 16, 8],
            win_lengths=[2048, 1024, 512, 256, 128, 64, 32],
            perceptual_weighting=True,
            sample_rate=config.sample_rate,
        )

        self.waveform_loss = L1Loss()

        self.gan_loss = GANLoss(self.discriminator)

        self.sample_rate = config.sample_rate
        self.frame_rate = self.generator.frame_rate
        self.hop_length = self.generator.hop_length
        self.latent_dim = self.generator.latent_dim
        self.n_channels = config.n_channels
        self.vae_beta = config.vae_beta
        logger.info(self)

    def __repr__(self):
        return f"\n{self.__class__.__name__}\nsample rate: {self.sample_rate}\nframe rate: {self.frame_rate}\nlatent dim: {self.latent_dim}\nn_channels: {self.n_channels}"

    def clean(self):
        del self.discriminator
        del self.sd_stft
        del self.lr_stft
        del self.waveform_loss
        gc.collect()
        torch.cuda.empty_cache()

    def to_torchscript(self):
        self._jit_is_scripting = True
        self.clean()
        self.generator = self.generator.apply(
            lambda m: remove_parametrizations(m, "weight") if is_parametrized(m) else m
        )
        self = self.eval()
        return torch.jit.script(self.cpu())

    def slice_quantizers(self, n_quantizers: int) -> Self:
        self.generator.quantizer.quantizers = self.generator.quantizer.quantizers[
            :n_quantizers
        ]
        self.latent_dim = self.generator.latent_dim
        return self

    def forward(self, audio: torch.Tensor) -> UACResult:
        return self.generator.forward(audio, self.sample_rate)

    def check_audio_dims(self, audio: torch.Tensor):
        if audio.ndim != 3:
            raise Exception("Make sure the tensor has shape: [batch, channel, time]")

    def get_codes(
        self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int = 480
    ):
        ## TODO: There's a small difference in the transition at `chunk_seconds`
        ## Audible when substracting chunked - non-chunked audio.

        self.check_audio_dims(audio)

        b, c, t = audio.shape

        # chunked inference
        max_samples = sample_rate * chunk_seconds
        audio = audio.split(max_samples, dim=-1)
        codes = []
        for a in audio:
            c = self.generator.get_codes(a, sample_rate)
            codes.append(c)
        codes = torch.cat(codes, dim=-1)

        # turn into int32, possibly even int16 if vocab < 32767
        codes = codes.to(torch.int32)
        return codes

    @torch.jit.export
    def get_z(
        self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int = 480
    ) -> torch.Tensor:
        ## TODO: There's a small difference in the transition at `chunk_seconds`
        ## Audible when substracting chunked - non-chunked audio.

        self.check_audio_dims(audio)

        # chunked inference
        max_samples = sample_rate * chunk_seconds
        audio = audio.split(max_samples, dim=-1)
        zs = []
        for a in audio:
            z = self.generator.get_z(a, sample_rate)
            zs.append(z)
        zs = torch.cat(zs, dim=-1)
        return zs

    @torch.jit.export
    def decode_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator.decode_z(z)

    def step(self, batch, batch_idx: int, return_loss: bool):
        targets = batch.audio
        result = self.forward(batch.audio.clone())
        recons = result.audio

        commitment_loss = result.commitment_loss
        codebook_loss = result.codebook_loss

        d_optimizer, g_optimizer = self.optimizers()
        d_scheduler, g_scheduler = self.lr_schedulers()

        loss_dict = dict()

        # discriminator losses:
        loss_disc = self.gan_loss.discriminator_loss(recons, targets)
        loss_dict["disc_loss"] = self.config.disc_loss * loss_disc

        ##########################
        # Optimize Discriminator #
        ##########################
        self.toggle_optimizer(d_optimizer)
        if (
            batch_idx % 2
            and batch_idx > self.config.generator_warmup_steps
            and self.training
        ):
            d_optimizer.zero_grad()
            self.manual_backward(loss_dict["disc_loss"])
            self.clip_gradients(
                d_optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm"
            )
            d_optimizer.step()
            d_scheduler.step()
        self.untoggle_optimizer(d_optimizer)

        # generator losses:
        loss_adv, feature_matching = self.gan_loss.generator_loss(recons, targets)

        loss_dict["adv_loss"] = loss_adv
        loss_dict["feature_matching"] = feature_matching

        # stft losses:
        loss_dict["sd_stft_loss"] = self.sd_stft(recons, targets)

        left_stft_loss = self.lr_stft(recons[:, 0:1], targets[:, 0:1])
        right_stft_loss = self.lr_stft(recons[:, 1:2], targets[:, 1:2])
        loss_dict["lr_stft_loss"] = (left_stft_loss * 0.5) + (right_stft_loss * 0.5)

        loss_dict["waveform_loss"] = self.waveform_loss(recons, targets)

        loss_dict["vq_commitment_loss"] = commitment_loss
        loss_dict["vq_codebook_loss"] = codebook_loss

        # scale losses
        if batch_idx > self.config.generator_warmup_steps:
            w_commitment_loss = self.config.commitment_loss
            w_codebook_loss = self.config.codebook_loss
        else:
            w_commitment_loss = 0
            w_codebook_loss = 0

        # aggregate losses
        loss_dict["loss"] = (
            self.config.stft_loss
            * (loss_dict["sd_stft_loss"] + loss_dict["lr_stft_loss"])
            + self.config.waveform_loss * loss_dict["waveform_loss"]
            + self.config.adverserial_loss * loss_dict["adv_loss"]
            + self.config.feat_loss * loss_dict["feature_matching"]
            + w_commitment_loss * loss_dict["vq_commitment_loss"]
            + w_codebook_loss * loss_dict["vq_codebook_loss"]
        )

        ######################
        # Optimize Generator #
        ######################
        self.toggle_optimizer(g_optimizer)
        if self.training:
            g_optimizer.zero_grad()
            self.manual_backward(loss_dict["loss"])
            self.clip_gradients(
                g_optimizer, gradient_clip_val=1e3, gradient_clip_algorithm="norm"
            )
            g_optimizer.step()
            g_scheduler.step()
        self.untoggle_optimizer(g_optimizer)

        if self.trainer.global_step % 100 == 0:
            latents = self.all_gather(result.latents)
            loss_dict["latents/mean"] = latents.mean()
            loss_dict["latents/std"] = latents.std()

        loss_dict["learning_rate_gen"] = g_optimizer.param_groups[0]["lr"]
        loss_dict["learning_rate_disc"] = d_optimizer.param_groups[0]["lr"]
        loss_dict["batch_size"] = recons.shape[0]
        return AudioCodecResult(
            audio=recons,
            audio_y=targets,
            latents=result.latents,
            codes=result.codes,
            vq_commitment_loss=result.commitment_loss,
            vq_codebook_loss=result.codebook_loss,
            loss=loss_dict,
        )

    def configure_optimizers(self):

        ## discriminator
        d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.disc_learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            fused=False,
        )
        d_scheduler = InverseLR(d_optimizer, inv_gamma=200000, power=0.5, warmup=0.999)
        d_scheduler = {"scheduler": d_scheduler, "name": "d_scheduler"}

        ## generator
        g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.gen_learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            fused=False,
        )
        g_scheduler = InverseLR(g_optimizer, inv_gamma=200000, power=0.5, warmup=0.999)
        g_scheduler = {"scheduler": g_scheduler, "name": "g_scheduler"}
        return [d_optimizer, g_optimizer], [d_scheduler, g_scheduler]


if __name__ == "__main__":
    # export CUDA_VISIBLE_DEVICES=0
    from pytorch_lightning import Trainer

    device = "cuda"

    config = AudioCodecConfig(bottleneck="vae")
    # config = SmallCodecConfig()
    audio_codec = AudioCodec(config)
    audio_codec = audio_codec.to(device)
    print(audio_codec.summarize(max_depth=3))

    batch_size = 8
    duration = 0.88236
    duration_samples = int(config.sample_rate * duration)

    x = torch.randn(batch_size, config.n_channels, duration_samples, device=device)
    result = audio_codec.forward(x)

    print(f"frame rate: {audio_codec.frame_rate}")
    print(f"z: {result.latents.shape}")
    print(result.latents.mean(), result.latents.std())
    exit(0)
