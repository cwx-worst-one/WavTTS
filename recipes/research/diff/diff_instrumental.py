import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from pytorch_lightning.utilities import grad_norm
from tqdm import tqdm
from typing_extensions import Self

from byteformers.models import LlamaConfig
from byteformers.models.llama import LlamaContextModel
from recipes.research.audio_codec.zoo import AudioCodec_7c355ea_64l, AudioCodec_f81b3fa_64l, AudioCodecBase
from recipes.research.diff.cond import NumberConditioner
from recipes.research.diff.diff import (
    DiffConfig,
    DiffResult,
    NullEmb,
    TimeEmbedding,
    UniformDistribution,
    StandardNormalDistribution,
    WeightedMAELoss,
    get_alpha_beta,
    get_steps_from_schedule,
    sequence_mask,
)
from recipes.research.diff.mulan import Mulan
from recipes.research.diff.normalize import FeatureNormalizer, FeatureNormalizerIdentity
from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.optim.lr_scheduler.tri_stage_lr import TriStageLR
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DiffInstrumental(DefaultTrainingBaseModule):

    _solvers = [
        "dpmpp-3m-sde",
        "dpmpp-2m-sde",
        "v-diffusion",
        "rectified-flow",
        "heun",
        "lms",
        "dpmpp-2s-ancestral",
        "dpm-2",
        "dpm-fast",
        "dpm-adaptive",
    ]

    def __init__(self, config: DiffConfig):
        super().__init__()
        self.config = config

        ### Load Models ###
        # self.load_audio_codec(AudioCodec_f81b3fa_64l())

        # stats_fp = "recipes/research/dataset/stats/ShutterStockFeatureParquetDataset_AudioCodec_f81b3fa_64l.stats.pt"
        # self.normalize = FeatureNormalizer(stats_fp)
        self.normalize  = FeatureNormalizerIdentity()
        logger.info(self.normalize)

        ## TODO: write signature check
        self.x_latent_dim = 64

        ### Conditioning Models
        self.seconds_start_emb = NumberConditioner(
            config.cond_n_embd, min_val=0, max_val=512
        )
        self.seconds_total_emb = NumberConditioner(
            config.cond_n_embd, min_val=0, max_val=512
        )

        logger.info("Loading Multi-Modal model checkpoint...")
        self.cond_model = Mulan(
            output_type="seq", text_only=True, text_feature_layer_idx=-2
        )
        self.cond_latent_dim = self.cond_model.n_embd
        if self.cond_latent_dim == config.cond_n_embd:
            self.proj_cond = nn.Identity()
        else:
            self.proj_cond = nn.Linear(
                self.cond_latent_dim, config.cond_n_embd, bias=False
            )

        self.cond_model.eval()
        self.cond_model.freeze()

        ### Diffusion

        ## time sampler and embedder:
        self.uniform_distribution = UniformDistribution(
            vmin=config.min_t, vmax=config.max_t
        )
        self.normal_distribution = StandardNormalDistribution()
        self.time_embedding = TimeEmbedding(config.time_n_embd, bias=False)

        ## null embedder:
        self.null_emb = NullEmb(config.n_embd, config.cfg_dropout, learnable=False)

        ## conditioning projection layers:
        self.proj_prefix = nn.Sequential(
            nn.Linear(config.cond_n_embd, config.n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )
        self.proj_cross_attn = nn.Sequential(
            nn.Linear(config.cond_n_embd, config.n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )
        self.proj_t = nn.Linear(config.time_n_embd, config.n_embd, bias=False)
        self.proj_x = nn.Linear(self.x_latent_dim, config.n_embd, bias=False)

        model_config = LlamaConfig(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            is_causal=False,
            use_rotary_embeddings=True,  # TODO: aliasing may occur longer duration
            interleave_rotary_embeddings=True,  # NOTE: verify this
            use_window_mask=False,
            window_type=0,  # blockwise
            window_size=(-1, -1),  # NOTE: (-1, -1) when using elemwise!
            gradient_checkpointing=True,
        )

        self.model = LlamaContextModel(model_config)
        self.proj_out = nn.Linear(config.n_embd, self.x_latent_dim, bias=False)

        self.criterion = WeightedMAELoss()

        self.sample_rate = config.sample_rate
        self.max_duration = config.max_duration
        self.min_t = config.min_t
        self.max_t = config.max_t
        self.noise_scale_factor = config.noise_scale_factor
        self.initialize_weights()

    @property
    def solvers(self) -> List[str]:
        return self._solvers

    def setup(self, stage: Optional[str] = None, device: Optional[torch.device] = None):
        if device is None:
            self.cond_model.cast_to_rank(self.local_rank)
        else:
            self.cond_model.to(device)

    def initialize_weights(self):
        pass

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.global_step % 500 == 0:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    def load_audio_codec(self, audio_codec: AudioCodecBase, device: torch.device) -> Self:
        logger.info("Loading AudioCodec model checkpoint...")
        self.audio_codec_autocast = False  # TODO codec is trained in fp32
        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            self.audio_codec = audio_codec.to(device)

            #     ## ConvCodec
            #     self.audio_codec = AudioCodec_f81b3fa_64l()
            #     # self.audio_codec = MelCodec_d100dfd_64l()
            #     # self.audio_codec = MelCodec_d100dfd_64l_642k()
            #     # self.audio_codec = MelCodec_6432147_64l_251k()
            #     self.x_latent_dim = self.audio_codec.latent_dim

            assert self.x_latent_dim == self.audio_codec.latent_dim
            assert (
                self.config.sample_rate == self.audio_codec.sample_rate
            ), "Sampling rates do not match"
            self.audio_codec.eval()
            self.audio_codec.freeze()
        return self

    def get_audio_latents(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            with torch.no_grad():
                if self.audio_codec.training:
                    self.audio_codec.eval()

                latents = self.audio_codec.get_z(audio, sample_rate)
                latents = latents.detach()
                return latents

    def decode_audio(self, z: torch.Tensor):
        if self.audio_codec.training:
            self.audio_codec.eval()

        with torch.cuda.amp.autocast(enabled=self.audio_codec_autocast):
            with torch.no_grad():
                z = z.permute(0, 2, 1)  # [B, D, T]
                z = self.normalize.inverse(z)
                return self.audio_codec.decode_z(z)

    def get_multimodal_embs(self, text: List[str]) -> Dict[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if self.cond_model.training:
                    self.cond_model = self.cond_model.eval()

                cond_result = self.cond_model.forward(text=text)
                cond_emb = cond_result.hidden_states.detach()  # [B, T_multimodal, D]
                cond_attn_mask = cond_result.attention_mask

        cond_emb = self.proj_cond(cond_emb)
        return {"embeds": cond_emb, "attention_mask": cond_attn_mask}

    def get_cond_embs(
        self,
        text: List[str],
        seconds_start: List[int],
        seconds_total: List[int],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        ###
        ### Prepare and align conditioning
        ###

        ## prefix (in-context)
        sec_start_result = self.seconds_start_emb.forward(seconds_start, device=device)
        sec_total_result = self.seconds_total_emb.forward(seconds_total, device=device)

        sec_start_emb = sec_start_result["embeds"]
        sec_start_mask = sec_start_result["attention_mask"]

        sec_total_emb = sec_total_result["embeds"]
        sec_total_mask = sec_total_result["attention_mask"]

        prefix_cond = torch.cat((sec_start_emb, sec_total_emb), dim=1)
        prefix_attn_mask = torch.cat((sec_start_mask, sec_total_mask), dim=1)

        ## cross-attention
        cond_result = self.get_multimodal_embs(text)
        cross_attn_cond = torch.cat(
            (sec_start_emb, sec_total_emb, cond_result["embeds"]), dim=1
        )
        cross_attn_mask = torch.cat(
            (sec_start_mask, sec_total_mask, cond_result["attention_mask"]), dim=1
        )

        # assert cross_attn_mask.sum() == cond_attn_mask.sum() + (2 * sec_start_mask.shape[0])
        return {
            "prefix_cond": prefix_cond,
            "prefix_attn_mask": prefix_attn_mask,
            "cross_attn_cond": cross_attn_cond,
            "cross_attn_mask": cross_attn_mask,
        }

    def get_all_embs(
        self,
        seconds_start: List[int],
        seconds_total: List[int],
        text: List[str],
        audio: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        audio_latents: Optional[torch.Tensor] = None,
    ):
        ###
        ### Get latents
        ###
        if audio_latents is None:
            audio_latents = self.get_audio_latents(audio, sample_rate)  # [B, D, T]

        audio_latents = self.normalize(audio_latents)
        audio_latents = audio_latents.transpose(1, 2)  # [B, T, D]

        cond = self.get_cond_embs(
            text, seconds_start, seconds_total, device=audio_latents.device
        )
        return {
            "prefix_cond": cond["prefix_cond"],
            "prefix_attn_mask": cond["prefix_attn_mask"],
            "cross_attn_cond": cond["cross_attn_cond"],
            "cross_attn_mask": cond["cross_attn_mask"],
            "audio_latents": audio_latents,
        }

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sigmas: torch.Tensor,
        attn_mask: torch.Tensor,
        prefix: torch.Tensor,
        prefix_attn_mask: torch.Tensor,
        context: torch.Tensor,
        context_attn_mask: torch.Tensor,
    ):
        residual = noisy_latent

        time_emb = self.time_embedding(sigmas)  # [B, n_embd]
        time_emb = repeat(
            time_emb, "b time_n_embd -> b t time_n_embd", t=1
        )  # [B, T_codec, n_embd]

        time_emb = self.proj_t(time_emb)  # [B, T_codec, n_embd]
        noisy_latent = self.proj_x(noisy_latent)  # [1, T_codec, n_embd]

        # prefix time and global conditioning
        prefix_len = time_emb.shape[1] + prefix.shape[1]
        noisy_latent = torch.cat((time_emb, prefix, noisy_latent), dim=1)

        time_attn_mask = torch.ones(
            time_emb.shape[0], 1, device=prefix_attn_mask.device
        )
        attn_mask = torch.cat((time_attn_mask, prefix_attn_mask, attn_mask), dim=1)

        pred_velocity = self.model.forward(
            noisy_latent,
            context=context,
            attn_mask=attn_mask,
            context_attn_mask=context_attn_mask,
        )  # [B, T_codec, n_embd]
        pred_velocity = pred_velocity[:, prefix_len:]  # remove prefix

        pred_velocity = self.proj_out(pred_velocity)  # [B, T_codec, x_latent_dim]

        ### Apply skip connection (TODO: evaluate impact)
        pred_velocity = residual + pred_velocity  # [B, T_codec, x_latent_dim]

        return pred_velocity

    def forward_with_cfg(
        self,
        noisy_latent: torch.Tensor,
        sigmas: torch.Tensor,
        attn_mask: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]] = None,
    ):
        noisy_latent = torch.cat((noisy_latent, noisy_latent), dim=0)
        attn_mask = torch.cat((attn_mask, attn_mask), dim=0)
        sigmas = torch.cat((sigmas, sigmas), dim=0)

        ### CFG inference
        prefix = self.proj_prefix(cond["prefix_cond"])
        context = self.proj_cross_attn(cond["cross_attn_cond"])

        ### uncond emb
        if uncond is None:
            uncond_context_emb = self.null_emb.forward_cfg(context)
            uncond_prefix_emb = self.null_emb.forward_cfg(prefix)
        else:
            uncond_prefix_emb = self.proj_prefix(uncond["prefix_cond"])
            uncond_context_emb = self.proj_cross_attn(uncond["cross_attn_cond"])

        prefix = torch.cat((prefix, uncond_prefix_emb), dim=0)
        prefix_attn_mask = torch.cat((cond["prefix_attn_mask"], cond["prefix_attn_mask"]), dim=0)

        context = torch.cat((context, uncond_context_emb), dim=0)
        cross_attn_mask = torch.cat((cond["cross_attn_mask"], cond["cross_attn_mask"]), dim=0)

        pred_velocity = self.forward(
            noisy_latent,
            sigmas,
            attn_mask,
            prefix,
            prefix_attn_mask,
            context,
            cross_attn_mask,
        )
        return pred_velocity

    def step(self, batch: AudioDataResult, batch_idx: int, return_loss: bool):
        audio = batch.audio
        batch_size = audio.shape[0]
        seconds_start = [math.floor(s.seek_time) for s in batch.segment_info]
        seconds_total = [math.floor(s.meta.duration) for s in batch.segment_info]
        text = [i["text"] for i in batch.index]

        logger.info(text)

        ### Prepare features

        ## feature dataset:
        lengths = torch.tensor(
            [s.n_frames for s in batch.segment_info], device=audio.device
        )

        # audio:
        # lengths = (lengths / self.sample_rate * self.audio_codec.frame_rate)  # TODO: floor this?

        # feature:
        audio_latents = audio  # TODO: switch
        embs = self.get_all_embs(
            seconds_start,
            seconds_total,
            text,
            audio_latents=audio_latents,
            # audio=audio,
            # sample_rate=self.sample_rate,
        )
        x = embs["audio_latents"]  # [B, T_codec, x_latent_dim]

        ### CFG
        context = embs["cross_attn_cond"]
        context = self.proj_cross_attn(context)
        context = self.null_emb(context)
        context_attn_mask = embs["cross_attn_mask"]

        prefix = embs["prefix_cond"]
        prefix_attn_mask = embs["prefix_attn_mask"]
        prefix = self.proj_prefix(prefix)
        prefix = self.null_emb(prefix)

        ### Diffusion

        ### Create noisy latent and target velocity
        noise = torch.randn_like(x)  # [B, T_codec, x_latent_dim]

        ### Create timestep embedding
        ### Sample continuous timesteps between [min_t, max_t] (dubbed "sigmas")

        if self.config.snr_sampler == "cosine":
            sigmas = self.uniform_distribution(num_samples=batch_size, device=x.device)
            alphas, betas = get_alpha_beta(sigmas[:, None, None])
        elif self.config.snr_sampler == "logit_normal":
            sigmas = self.normal_distribution(num_samples=batch_size, device=x.device)
            sigmas = sigmas.sigmoid()

            alphas = sigmas[:, None, None]
            betas = (1 - alphas)
        else:
            raise NotImplementedError("Choose between: ['logit_normal', 'cosine']")

        noisy_latent = alphas * x + betas * noise
        target_latent = alphas * noise - betas * x

        # T. Chen et. al (2023)
        if self.noise_scale_factor != 1.0:
            x = x * self.noise_scale_factor

        ### Predict velocity from noisy latent with model
        ## feature dataset:
        attn_mask = sequence_mask(lengths, max_length=noisy_latent.shape[1])

        pred = self.forward(
            noisy_latent,
            sigmas,
            attn_mask,
            prefix,
            prefix_attn_mask,
            context,
            context_attn_mask,
        )

        result = DiffResult(
            global_cond=None,
            prefix_cond=embs["prefix_cond"],
            cross_attn_cond=embs["cross_attn_cond"],
            audio_latents=embs["audio_latents"],
        )

        ### MSE loss between predicted and target velocity
        with torch.cuda.amp.autocast(enabled=False):
            result.loss = self.loss(pred, target_latent, sigmas)

        if self._trainer is not None and self.trainer.global_step % 100 == 0:
            latents = self.all_gather(x)
            result.loss["latents/mean"] = latents.mean()
            result.loss["latents/std"] = latents.std()
        return result

    def loss(
        self, pred: torch.Tensor, target: torch.Tensor, sigmas: torch.Tensor
    ) -> LossDict:
        loss = self.criterion.forward(pred, target)  # [B]

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
        return {"loss": loss, "batch_size": pred.shape[0], **mean_losses}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )

        scheduler = TriStageLR(
            optimizer,
            self.config.learning_rate,
            self.config.min_lr,
            self.config.warmup_steps,
            self.config.hold_steps,
            self.config.decay_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def sample_wrapper(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        attn_mask: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_weight: float,
        uncond: Optional[Dict[str, torch.Tensor]] = None,
    ):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # assert self.config.max_t > sigmas > self.config.min_t
            v_pred_cond, v_pred_uncond = self.forward_with_cfg(
                x,
                sigmas,
                attn_mask,
                cond,
                uncond,
            ).chunk(2, dim=0)
            v_pred = cfg_weight * v_pred_cond + (1 - cfg_weight) * v_pred_uncond
            return v_pred

    def sample(
        self,
        cond: Dict[str, torch.Tensor],
        t: int,
        cfg_weight: float,
        schedule_tau: float,
        uncond: Optional[Dict[str, torch.Tensor]],
        init_noise: Optional[torch.Tensor] = None,
        solver: str = "dpmpp-3m-sde",
        duration: float = 60,
        device: str = "cuda",
    ):

        ###
        ### 2. Create noisy latent and target velocity
        ###
        batch_size = cond["prefix_cond"].shape[0]
        n_frames = math.floor(
            min(self.max_duration, duration) * self.audio_codec.frame_rate
        )

        if init_noise is None:
            x = torch.randn((batch_size, n_frames, self.x_latent_dim), device=device)
        else:
            x = init_noise.clone()

        attn_mask = torch.ones(batch_size, n_frames, device=device)

        if solver == "rectified-flow":
            dt = 1.0 / t
            dt = torch.tensor([dt] * batch_size, device=device)[:, None, None]
            sigmas = get_steps_from_schedule(t + 1, schedule="linear", device=device)
            sigmas = repeat(sigmas, "t -> t b", b=batch_size)  # [t, B]

            if self.config.snr_sampler == "cosine":
                alphas, betas = get_alpha_beta(sigmas[:, :, None, None])  # [t, B, :, :]
            elif self.config.snr_sampler == "logit_normal":
                alphas = sigmas[:, :, None, None]
                betas = (1 - alphas)

            for i in range(t):
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    v_pred = self.sample_wrapper(
                        x,
                        sigmas[i],
                        attn_mask,
                        cond,
                        cfg_weight,
                        uncond,
                    )

                # solution #1:
                x = x - dt * v_pred

                # solution #2:
                # x_pred = alphas[i] * x - betas[i] * v_pred
                # noise_pred = betas[i] * x + alphas[i] * v_pred

                # # TODO evaluate https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/diffusion/sampling.py#L32
                # x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

            x_pred = x

        elif solver == "v-diffusion":
            sigmas = get_steps_from_schedule(
                t + 1,
                schedule="cosine",
                start=self.min_t,
                end=self.max_t,
                tau=schedule_tau,
                device=device,
            )
            sigmas = repeat(sigmas, "t -> t b", b=batch_size)  # [t, B]


            if self.config.snr_sampler == "cosine":
                alphas, betas = get_alpha_beta(sigmas[:, :, None, None])  # [t, B, :, :]
            elif self.config.snr_sampler == "logit_normal":
                alphas = sigmas[:, :, None, None]
                betas = (1 - alphas)

            for i in tqdm(range(t)):
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    # TODO: CFG only done when sigmas[i] < 1?
                    v_pred = self.sample_wrapper(
                        x,
                        sigmas[i],
                        attn_mask,
                        cond,
                        cfg_weight,
                        uncond=uncond,
                    )

                # TODO evaluate https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/diffusion/sampling.py#L32
                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
        else:
            import k_diffusion as K

            sigma_min = 0.03
            sigma_max = 500
            denoiser = K.external.VDenoiser(self.sample_wrapper)
            sigmas = K.sampling.get_sigmas_polyexponential(
                t,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=schedule_tau,
                device=device,
            )

            ## scale initial noise by sigma:
            x = x * sigmas[0]
            extra_args = dict(
                attn_mask=attn_mask,
                cond=cond,
                cfg_weight=cfg_weight,
                uncond=uncond,
            )
            if solver == "heun":
                x_pred = K.sampling.sample_heun(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )
            elif solver == "lms":
                x_pred = K.sampling.sample_lms(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )
            elif solver == "dpmpp-2s-ancestral":
                x_pred = K.sampling.sample_dpmpp_2s_ancestral(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )
            elif solver == "dpm-2":
                x_pred = K.sampling.sample_dpm_2(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )
            elif solver == "dpm-fast":
                x_pred = K.sampling.sample_dpm_fast(
                    denoiser,
                    x,
                    sigma_min,
                    sigma_max,
                    t,
                    disable=False,
                    extra_args=extra_args,
                )
            elif solver == "dpm-adaptive":
                x_pred = K.sampling.sample_dpm_adaptive(
                    denoiser,
                    x,
                    sigma_min,
                    sigma_max,
                    rtol=0.01,
                    atol=0.01,
                    disable=False,
                    extra_args=extra_args,
                )
            elif solver == "dpmpp-2m-sde":
                x_pred = K.sampling.sample_dpmpp_2m_sde(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )
            elif solver == "dpmpp-3m-sde":
                x_pred = K.sampling.sample_dpmpp_3m_sde(
                    denoiser, x, sigmas, disable=False, extra_args=extra_args
                )

        if self.noise_scale_factor != 1.0:
            x_pred = x_pred / self.noise_scale_factor
        return x_pred

    def sample_from_text(
        self,
        text: List[str],
        seconds_start: List[int],
        seconds_total: List[int],
        t: int,
        cfg_weight: float,
        schedule_tau: float,
        device: torch.device,
        negative_text: Optional[List[str]] = None,
        init_noise: Optional[torch.Tensor] = None,
        solver: str = "dpmpp-3m-sde",
        duration: float = 60,
    ):
        assert hasattr(self, "audio_codec"), "Use `self.load_audio_codec() first"
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            cond = self.get_cond_embs(
                text, seconds_start, seconds_total, device=device,
            )
            if negative_text is None:
                uncond = None
            else:
                uncond = self.get_cond_embs(
                    negative_text, seconds_start, seconds_total, device=device,
                )

        x_pred = self.sample(
            cond,
            t=t,
            cfg_weight=cfg_weight,
            schedule_tau=schedule_tau,
            uncond=uncond,
            init_noise=init_noise,
            solver=solver,
            duration=duration,
            device=device,
        )
        return x_pred


if __name__ == "__main__":
    config = DiffConfig(
        sample_rate=44100, max_duration=30, n_layer=4, n_embd=1024, n_head=16, snr_sampler="logit_normal"
    )
    model = DiffInstrumental(config).to("cuda")
    model.setup()
    print(model.summarize())

    batch_size = 2
    # audio = torch.randn(
    #     batch_size, 2, config.max_duration * config.sample_rate, device=model.device
    # )

    frame_rate = 25
    x = torch.randn(
        batch_size,
        model.x_latent_dim,
        config.max_duration * frame_rate,
        device=model.device,
    )

    segment_info = SegmentInfo(
        meta=AudioMeta(path=None, duration=30, sample_rate=config.sample_rate),
        data_type="music_vocal",
        seek_time=0.0,
        n_frames=x.shape[2],
        total_frames=x.shape[2],
        sample_rate=config.sample_rate,
        channels=2,
        lyrics=None,
    )
    index = {
        "keywords": "upbeat, technology, technological, pulsing, lively, kawaii, japan, cute, bubblegum, bouncy",
        "genres": "country",
        "instruments": "drums, guitar, piano",
        "bpm": "121",
        "description": "Bouncy and cute",
        "text": "upbeat, technology, technological",
    }

    batch = AudioDataResult(
        audio=x,
        shard=None,
        key=None,
        segment_info=[segment_info] * batch_size,
        index=[index] * batch_size,
    )

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        result = model.step(batch, batch_idx=0, return_loss=True)

        audio_codec = AudioCodec_7c355ea_64l()
        model.load_audio_codec(audio_codec, device="cuda")
        seconds_start = [0]
        seconds_total = [30]
        text = [s["keywords"] for s in batch.index][:1]

        result = model.sample_from_text(
            text, seconds_start, seconds_total, t=10, cfg_weight=1.0, schedule_tau=0.4, device="cuda", solver="v-diffusion"
        )
