import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from byteformers.models import LlamaConfig, LlamaModel
from samantha.data.audio.types import AudioDataResult, SegmentInfo
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

from recipes.research.audio_codec.zoo import AudioCodec
from recipes.research.diff.diff import (
    DiffConfig,
    DiffResult,
    NullEmb,
    SemanticPreNet,
    TimeEmbedding,
    UniformDistribution,
    WeightedMSELoss,
    get_alpha_beta,
    get_steps_from_schedule,
    linear_schedule,
    xavier_init,
)
from recipes.research.mel_codec.zoo import (
    MelCodec_6432147_64l_251k,
    MelCodec_d100dfd_64l,
    MelCodec_d100dfd_64l_642k,
)


class DiffCodec(DefaultTrainingBaseModule):
    def __init__(self, config: DiffConfig):
        super().__init__()
        self.config = config

        ###
        ### Load Models ###
        ###
        logger.info("Loading AudioCodec model checkpoint...")
        self.audio_codec_autocast = False  # TODO codec is trained in fp32

        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            self.audio_codec = AudioCodec()
            self.x_latent_dim = self.audio_codec.latent_dim
            assert (
                config.sample_rate == self.audio_codec.sample_rate
            ), "Sampling rates do not match"
            self.audio_codec.eval()
            self.audio_codec.freeze()

        ###
        ### Conditioning Models
        ###
        logger.info("Loading MelCodec checkpoint...")

        ## ConvCodec
        # self.mel_codec = MelCodec_d100dfd_64l()
        # self.mel_codec = MelCodec_d100dfd_64l_642k()
        # self.mel_codec = MelCodec_6432147_64l_251k()

        ## TCodec
        # from recipes.research.mel_codec.scripts.compile import get_mel_codec
        # self.mel_codec = get_mel_codec("af358c2", ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/af358c2/2024-05-26/02-31-25/checkpoints/step=288000.ckpt") # 60s
        # self.mel_codec = get_mel_codec("9a660a8", ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/9a660a8/2024-05-26/02-33-05/checkpoints/step=999000.ckpt") # 10s
        # self.mel_codec = get_mel_codec("01769e1", ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/01769e1/2024-05-31/14-27-06/checkpoints/step=1086000.ckpt") # 10s, 10hz
        # self.cond_latent_dim = self.mel_codec.n_freq_bins

        # from recipes.research.mel_codec.mel_rq import get_mel_codec, get_mel_vq
        # self.mel_codec = get_mel_codec(
        #     "657b0ed",
        #     ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/657b0ed/2024-06-19/06-20-26/checkpoints/step=278000.ckpt",
        # )
        # self.mel_codec_layer_idx = 12
        # self.mel_codec.model.layers = self.mel_codec.model.layers[: self.mel_codec_layer_idx]

        ## nrq
        # self.mel_codec = get_mel_codec(
        #     "f7db9ef",
        #     ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/f7db9ef/2024-06-27/14-03-25/checkpoints/step=387000.ckpt",
        # )
        # self.mel_codec_layer_idx = 24
        # self.mel_codec.model.h = self.mel_codec.model.h[:self.mel_codec_layer_idx]
        # self.mel_frame_rate = self.mel_codec.frame_rate
        # self.cond_latent_dim = self.mel_codec.encoder_n_embd

        ## vq:
        # self.cond_latent_dim = 1024
        # self.vocab_size = self.mel_codec.codebook_size * self.mel_codec.n_codebooks
        # self.input_embs = nn.ModuleList(
        #     [nn.Embedding(self.mel_codec.codebook_size, self.cond_latent_dim) for _ in range(self.mel_codec.n_codebooks)]
        # )
        # self.input_emb = nn.Embedding(self.vocab_size, self.cond_latent_dim)
        # self.mel_frame_rate = self.mel_codec.frame_rate * self.mel_codec.n_codebooks

        # assert (
        #     config.sample_rate == self.mel_codec.sample_rate
        # ), "Sampling rates do not match"

        from recipes.research.diff.mulan import Mulan

        self.mel_codec = Mulan(output_type="seq", text_only=False)
        self.mel_frame_rate = self.mel_codec.frame_rate
        self.cond_latent_dim = self.mel_codec.n_embd

        if config.sample_rate != self.mel_codec.sample_rate:
            logger.warning("!!! SAMPLE RATES DO NOT MATCH, WILL RESAMPLE ONLINE !!!!")

        self.mel_codec.eval()
        self.mel_codec.freeze()

        self.cond_n_embd = 1024
        self.prenet = SemanticPreNet(
            self.cond_latent_dim, self.cond_n_embd, upscale=[7, 7], downscale=[5, 10]
        )

        ###
        ### Diffusion
        ###
        self.sigma_distribution = UniformDistribution(
            vmin=config.min_t, vmax=config.max_t
        )
        self.proj_global_cond = nn.Linear(self.cond_n_embd, config.n_embd, bias=False)
        self.null_emb = NullEmb(config.n_embd, config.cfg_dropout)
        self.time_embedding = TimeEmbedding(config.time_n_embd, bias=False)
        self.proj_t = nn.Linear(config.time_n_embd, config.n_embd, bias=False)
        self.proj_x = nn.Linear(self.x_latent_dim, config.n_embd, bias=False)

        model_config = LlamaConfig(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            use_rotary_embeddings=True,
            is_causal=False,
        )

        self.model = LlamaModel(model_config)
        # model_config = ConformerConfig(
        #     n_layer=config.n_layer,
        #     n_head=config.n_head,
        #     n_embd=config.n_embd,
        #     n_inner=config.n_embd * 4,
        #     max_seq_len=self.max_seq_len,
        #     mlp_dropout=0.0,
        #     attn_dropout=0.0,
        #     conv_dropout=0.0,
        #     conv_expansion_factor=2,  # TODO
        #     use_rotary_embeddings=True,
        #     is_causal=False,
        #     attention_kwargs={
        #         "enable_flash": True,
        #         "enable_mem_efficient": False,
        #         "enable_math": False,
        #     },
        # )

        # if config.cross_attn:
        #     self.model = ConformerContextModel(model_config)
        # else:
        #     self.model = Conformer(model_config)

        self.proj_out = nn.Linear(config.n_embd, self.x_latent_dim, bias=False)
        self.criterion = WeightedMSELoss()
        self.initialize_weights()

        self.sample_rate = config.sample_rate

    def setup(self, stage: Optional[str] = None):
        pass

    def initialize_weights(self):
        logger.info("Initializing weights...")
        self.time_embedding.init_weights()
        self.proj_global_cond.apply(xavier_init)
        self.proj_t.apply(xavier_init)
        self.proj_x.apply(xavier_init)
        self.proj_out.apply(xavier_init)
        self.model.apply(self.model._init_weights)

    def get_audio_latents(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            with torch.no_grad():
                if self.audio_codec.training:
                    self.audio_codec = self.audio_codec.eval()

                z = self.audio_codec.get_z(audio, sample_rate)
                z = z.detach()
                z = z.squeeze(dim=1)  # 1 channel

                # convention is: [B, T, D]
                z = z.permute(0, 2, 1)  # [B, T_codec, D_codec]
                return z

    def get_mel_latents(
        self, audio: torch.Tensor, sample_rate: int, lengths: torch.Tensor
    ):
        # with torch.cuda.amp.autocast(enabled=False):
        #     with torch.no_grad():
        #         if self.mel_codec.training:
        #             self.mel_codec = self.mel_codec.eval()
        #         z = self.mel_codec.get_z(audio, sample_rate, chunk_seconds=10)  # [B, D, T]
        #         z = self.mel_codec.get_rec_mel(audio, sample_rate, chunk_seconds=10)  # [B, D, T]
        #         return z

        ## MelRQ:
        # with torch.no_grad():
        #     if self.mel_codec.training:
        #         self.mel_codec = self.mel_codec.eval()
        #     z = self.mel_codec.get_hidden_states(audio, sample_rate, lengths, chunk_seconds=30)  # [B, T, D]
        #     z = z.detach()
        #     return z.transpose(1, 2)

        ## MelVQ:
        # codes = self.mel_codec.get_codes(audio, sample_rate, chunk_seconds=30)  # [B, C, T]
        # codes = codes.detach()

        # embs = []
        # for c_idx in range(len(self.input_embs)):
        #     codes_emb = self.input_embs[c_idx](codes[:, c_idx])
        #     embs.append(codes_emb)

        # flattened_embs = torch.cat(embs, dim=1)  # [B, T, D]
        # torch.testing.assert_close(flattened_embs[:, 0:750], embs[0])
        # torch.testing.assert_close(flattened_embs[:, 750:1500], embs[1])
        # torch.testing.assert_close(flattened_embs[:, 1500:2250], embs[2])
        # torch.testing.assert_close(flattened_embs[:, 2250:3000], embs[3])

        # flattened_codes = self.mel_codec.flatten_codes(codes)  # [B, T]
        # flattened_embs = self.input_emb(flattened_codes)  # [B, T, D]
        # return flattened_embs.transpose(1, 2)  # [B, D, T]

        ## Mulan
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if self.mel_codec.training:
                    self.mel_codec = self.mel_codec.eval()

                result = self.mel_codec.forward(audio=audio, sample_rate=sample_rate)
                z = result.mulan_audio_embs.transpose(1, 2)  # [B, D, T]
                z = z.detach()
                return z

    def setup(self, stage: Optional[str] = None):
        self.mel_codec.cast_to_rank(self.local_rank)

    def decode_audio(self, z: torch.Tensor):
        if self.audio_codec.training:
            self.audio_codec.eval()

        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            with torch.no_grad():
                z = z.permute(0, 2, 1)  # [B, D, T]
                return self.audio_codec.decode_z(z)

    def get_all_embs(
        self, audio: torch.Tensor, sample_rate: int, lengths: torch.Tensor
    ):
        audio_latents = self.get_audio_latents(audio, sample_rate)

        ## Mel Codec:
        mel_latents = self.get_mel_latents(audio, sample_rate, lengths)
        return {"global_cond": mel_latents, "audio_latents": audio_latents}

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sigmas: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = noisy_latent
        time_emb = self.time_embedding(sigmas)  # [B, n_embd]
        time_emb = repeat(
            time_emb, "b time_n_embd -> b t time_n_embd", t=1
        )  # [B, T_codec, n_embd]

        time_emb = self.proj_t(time_emb)  # [B, T_codec, n_embd]
        noisy_latent = self.proj_x(noisy_latent)  # [1, T_codec, n_embd]

        # additive
        noisy_latent = noisy_latent + time_emb + global_cond

        pred_velocity = self.model.forward(noisy_latent)  # [B, T_codec, n_embd]

        pred_velocity = self.proj_out(pred_velocity)
        pred_velocity = residual + pred_velocity
        return pred_velocity

    def forward_with_cfg(
        self,
        noisy_latent: torch.Tensor,
        sigmas: torch.Tensor,
        global_cond: torch.Tensor,
    ):
        noisy_latent = torch.cat((noisy_latent, noisy_latent), dim=0)
        sigmas = torch.cat((sigmas, sigmas), dim=0)

        ### uncond emb
        global_cond = self.proj_global_cond(global_cond)
        uncond_emb = self.null_emb.forward_cfg(global_cond)
        global_cond = torch.cat((global_cond, uncond_emb), dim=0)

        pred_velocity = self.forward(noisy_latent, sigmas, global_cond)
        return pred_velocity

    def loss(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> LossDict:
        loss = self.criterion.forward(pred_velocity, target_velocity)  # [B]

        # Timestep ranges we want to report (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0)
        levels = np.linspace(0, 1.0, 5)
        ranges = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

        # Initialize a list to store mean loss for each range
        mean_losses = {}
        for lower, upper in ranges:
            # Create a mask for noise levels within the current range
            mask = (sigmas > lower) & (sigmas <= upper)

            # Use the mask to index loss values and compute mean loss
            mean_loss = loss[mask].mean()
            if not mean_loss.isnan():
                mean_losses[f"loss_{lower}-{upper}"] = mean_loss

        loss = loss.mean()
        return {"loss": loss, "batch_size": pred_velocity.shape[0], **mean_losses}

    def step(self, batch: AudioDataResult, batch_idx: int, return_loss: bool):
        audio = batch.audio

        device = batch.audio.device

        lengths = torch.tensor(
            [s.n_frames for s in batch.segment_info], device=device
        ).floor()

        embs = self.get_all_embs(audio, self.config.sample_rate, lengths)
        x = embs["audio_latents"]  # [B, T_codec, x_latent_dim]
        global_cond = embs["global_cond"]  # [B, T_cond, cond_latent_dim]

        global_cond = self.prenet.forward(global_cond).permute(
            0, 2, 1
        )  # [B, T_melcodec, D]

        # # TODO: why is this trim necessary?
        # if global_cond.shape[1] > x.shape[1]:
        #     global_cond = global_cond[:, :x.shape[1]]
        # elif x.shape[1] > global_cond.shape[1]:
        #     x = x[:, :global_cond.shape[1]]

        ## CFG
        global_cond = self.proj_global_cond(global_cond)  # [B, T_cond, n_embd]
        global_cond = self.null_emb(global_cond)  # [B, T_cond, cond_latent_dim]

        batch_size = x.shape[0]

        ## v-diffusion:
        sigmas = self.sigma_distribution(num_samples=batch_size, device=x.device)
        alphas, betas = get_alpha_beta(sigmas[:, None, None])

        # T. Chen et. al (2023)
        if self.config.noise_scale_factor != 1.0:
            x = x * self.config.noise_scale_factor

        noise = torch.randn_like(x)  # [B, T_codec, x_latent_dim]
        noisy_latent = alphas * x + betas * noise
        target_velocity = alphas * noise - betas * x
        pred_velocity = self.forward(noisy_latent, sigmas, global_cond)

        result = DiffResult(
            global_cond=embs["global_cond"],
            prefix_cond=None,
            cross_attn_cond=None,
            audio_latents=embs["audio_latents"],
        )

        with torch.cuda.amp.autocast(enabled=False):
            ## v-diffusion:
            result.loss = self.loss(pred_velocity, target_velocity, sigmas)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        scheduler = WarmupCosine(
            optimizer,
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.cycle_steps,
            self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def ddim_sample(
        self, global_cond: torch.Tensor, t: int, cfg_weight: float, schedule_tau: float
    ):

        device = "cuda"
        n_frames = self.config.max_duration * self.audio_codec.frame_rate

        x = torch.randn((1, n_frames, self.x_latent_dim), device=device)

        sigmas = get_steps_from_schedule(
            t + 1,
            schedule="cosine",
            start=self.config.min_t,
            end=self.config.max_t,
            tau=schedule_tau,
            device=device,
        )
        sigmas = repeat(sigmas, "t -> t b", b=1)

        ## velocity:
        alphas, betas = get_alpha_beta(sigmas[:, None, None])

        for i in range(t):
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # TODO: CFG only done when sigmas[i] < 1?
                v_pred_cond, v_pred_uncond = self.forward_with_cfg(
                    x, sigmas[i], global_cond
                ).chunk(2, dim=0)
                v_pred = cfg_weight * v_pred_cond + (1 - cfg_weight) * v_pred_uncond

            x_pred = alphas[i] * x - betas[i] * v_pred
            noise_pred = betas[i] * x + alphas[i] * v_pred

            # TODO evaluate https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/diffusion/sampling.py#L32
            x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        if self.config.noise_scale_factor != 1.0:
            x_pred = x_pred / self.config.noise_scale_factor
        return x_pred

    def sample_z_audio_with_z_mel(
        self, z_mel: torch.Tensor, t: int, cfg_weight: float, schedule_tau: float
    ) -> torch.Tensor:

        SILENCE_VAL = -80

        assert z_mel.shape[0] == 1  # [B, T, D]
        x_preds = []
        for m in z_mel.split(
            math.floor(self.mel_frame_rate * self.config.max_duration), dim=1
        ):
            m = m.permute(0, 2, 1)  # [B, D, T]

            pad_len_frames = (
                math.ceil(self.config.max_duration * self.mel_frame_rate) - m.shape[2]
            )
            if pad_len_frames > 0:
                raise Exception("Not supported (SILENCE_VAL not valid for z_mel)")
                m = F.pad(m, (0, pad_len_frames), value=SILENCE_VAL)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    global_cond = self.prenet.forward(m).permute(
                        0, 2, 1
                    )  # [B, T_melcodec, D]

            ## v-diffusion:
            x_pred = self.ddim_sample(global_cond, t, cfg_weight, schedule_tau)

            if pad_len_frames > 0:
                pad_len_audio_frames = math.ceil(
                    pad_len_frames / (self.mel_frame_rate / self.audio_codec.frame_rate)
                )
                x_pred = x_pred[:, :-pad_len_audio_frames]

            x_preds.append(x_pred)
        x_preds = torch.cat(x_preds, dim=1)  # [B, T, D]
        return x_preds

    def sample_from_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        t: int,
        cfg_weight: float,
        schedule_tau: float,
    ):
        assert audio.shape[0] == 1
        x_preds = []
        for a in audio.split(sample_rate * self.config.max_duration, dim=2):
            pad_len = (
                math.ceil(self.config.max_duration * self.config.sample_rate)
                - a.shape[2]
            )
            if pad_len > 0:
                a = F.pad(a, (0, pad_len))

            lengths = torch.full(
                (a.shape[0],), a.shape[2], dtype=torch.long, device=a.device
            )
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    z_mel = self.get_mel_latents(a, sample_rate, lengths)  # [B, D, T]

            z_mel = z_mel.permute(0, 2, 1)  # [B, T, D]
            x_pred = self.sample_z_audio_with_z_mel(z_mel, t, cfg_weight, schedule_tau)

            pad_len_frames = math.ceil(
                pad_len / self.config.sample_rate * self.audio_codec.frame_rate
            )
            if pad_len_frames > 0:
                x_pred = x_pred[:, :-pad_len_frames]

            x_preds.append(x_pred)

        x_preds = torch.cat(x_preds, dim=1)  # [B, T, D]
        return x_preds


if __name__ == "__main__":
    config = DiffConfig(max_duration=30, n_layer=4, n_embd=1024, n_head=16)
    model = DiffCodec(config).to("cuda")
    model.setup()

    batch_size = 2
    duration_samples = config.max_duration * config.sample_rate
    audio = torch.randn(batch_size, 2, duration_samples, device=model.device)

    segment_info = SegmentInfo(
        meta=None,
        seek_time=None,
        n_frames=duration_samples,
        total_frames=duration_samples,
        sample_rate=config.sample_rate,
        channels=None,
        data_type=None,
        lyrics=None,
    )

    batch = AudioDataResult(
        audio=audio,
        shard=None,
        key=None,
        segment_info=[segment_info] * batch_size,
        index=None,
    )

    lengths = torch.tensor(
        [(s.n_frames / s.sample_rate) * config.sample_rate for s in batch.segment_info],
        device=model.device,
    ).floor()

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        result = model.step(batch, batch_idx=0, return_loss=True)

        z_mel = model.get_mel_latents(batch.audio[:1], config.sample_rate, lengths[:1])
        z_mel = z_mel.permute(0, 2, 1)  # [B, T, D]

        pred_z_audio = model.sample_z_audio_with_z_mel(
            z_mel, t=5, cfg_weight=2.0, schedule_tau=0.3
        )
        print(pred_z_audio.shape)
        pred_z_audio = model.sample_from_audio(
            batch.audio[:1], config.sample_rate, t=5, cfg_weight=2.0, schedule_tau=0.3
        )
        pred_audio = model.decode_audio(pred_z_audio)
        print(pred_z_audio.shape, pred_audio.shape)
