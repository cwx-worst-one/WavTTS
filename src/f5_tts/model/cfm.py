"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.eres2net_loss import ERes2NetFeatureLoss
from f5_tts.model.modules import MelSpec, MelSpectrogramLoss, HubertFeatureLoss
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
    masked_mean,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        prediction: str = "flow",       # "flow" | "x_pred"
        loss_space: str = "flow",       # "flow" | "v"
        t_sampling: str = "uniform",    # "uniform" | "logistic_normal"
        P_mean: float = 0.0,
        P_std: float = 1.0,
        t_eps: float = 1e-4,
        noise_scale: float = 1.0,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        aux_mel_loss_start_t: float = 0.0,
        sample_rate: int = 24000,
        use_repa_ctc_loss: bool = False,
        repa_ctc_loss_weight: float = 0.0,
        use_repa_ssl_feature_loss: bool = False,
        repa_ssl_feature_loss_weight: float = 0.0,
        latents_scale: float = 1.0,
        use_aux_hubert_loss: bool = False,
        aux_hubert_loss_weight: float = 1.0,
        aux_hubert_loss_start_t: float = 0.0,
        aux_hubert_model_path: str = "facebook/hubert-large-ll60k",
        aux_hubert_layer_index: int = -1,
        use_aux_eres2net_loss: bool = False,
        aux_eres2net_loss_weight: float = 1.0,
        aux_eres2net_loss_start_t: float = 0.0,
        aux_eres2net_model_path: str = "",
        aux_eres2net_target_sample_rate: int = 16000,
        aux_eres2net_feat_dim: int = 80,
        aux_eres2net_embedding_size: int = 192,
        frontend_type: str = "reshape",  # "reshape" | "conv"
        frontend_cfg: dict | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        
        # wav input
        mel_spec_kwargs = dict(mel_spec_kwargs)

        # wav-only switch + frame_len (won't be passed into MelSpec)
        self.wav_input_only = bool(mel_spec_kwargs.pop("return_wav_only", False))
        self.wav_frame_len = int(mel_spec_kwargs.pop("wav_frame_len", 240))  # e.g., 240 @24k = 100Hz
        self.frontend_type = frontend_type
        self.frontend_cfg = dict(frontend_cfg) if frontend_cfg is not None else {}

        # mel spec
        if self.wav_input_only:
            self.mel_spec = None
            self.num_channels = self.wav_frame_len
        else:
            self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
            self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        self.dim = transformer.dim

        if hasattr(self.transformer, "set_wav_frontend_config"):
            self.transformer.set_wav_frontend_config(
                wav_input_only=self.wav_input_only,
                wav_frame_len=self.wav_frame_len,
                frontend_type=self.frontend_type,
                frontend_cfg=self.frontend_cfg,
            )

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # enhanced flow model settings
        self.prediction = prediction
        self.loss_space = loss_space
        self.t_sampling = t_sampling
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.latents_scale = latents_scale

        # aux mel loss
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_start_t = aux_mel_loss_start_t
        if self.use_aux_mel_loss and self.wav_input_only:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],
                mel_fmax=[None] * 7,
                pow=1.0,
                clamp_eps=1e-5,
                mag_weight=0.0,    # As per config
                log_weight=1.0,    # As per config
                weight=aux_mel_loss_weight
            )
        else:
            self.aux_mel_loss = None

        self.mel_align_to = 1
        if self.wav_input_only and self.aux_mel_loss is not None and hasattr(self.aux_mel_loss, "mel_transforms"):
            hop_lengths = [int(m.hop_length) for m in self.aux_mel_loss.mel_transforms]
            if len(hop_lengths) > 0:
                lcm_hop = hop_lengths[0]
                for h in hop_lengths[1:]:
                    lcm_hop = math.lcm(lcm_hop, h)
                # rand_span_mask is on raw waveform sample axis in wav_input_only mode,
                # so alignment should also be in sample units.
                self.mel_align_to = max(1, lcm_hop)
            
        self.use_aux_hubert_loss = use_aux_hubert_loss
        self.aux_hubert_loss_weight = aux_hubert_loss_weight
        self.aux_hubert_loss_start_t = aux_hubert_loss_start_t
        if self.use_aux_hubert_loss and self.wav_input_only:
            self.aux_hubert_loss = HubertFeatureLoss(
                model_path=aux_hubert_model_path,
                layer_index=aux_hubert_layer_index,
                source_sample_rate=sample_rate,  # Defined in CFM init args
                weight=aux_hubert_loss_weight
            )
        else:
            self.aux_hubert_loss = None

        self.use_aux_eres2net_loss = use_aux_eres2net_loss
        self.aux_eres2net_loss_weight = aux_eres2net_loss_weight
        self.aux_eres2net_loss_start_t = aux_eres2net_loss_start_t
        if self.use_aux_eres2net_loss and self.wav_input_only:
            self.aux_eres2net_loss = ERes2NetFeatureLoss(
                model_path=aux_eres2net_model_path,
                source_sample_rate=sample_rate,
                target_sample_rate=aux_eres2net_target_sample_rate,
                feat_dim=aux_eres2net_feat_dim,
                embedding_size=aux_eres2net_embedding_size,
                weight=aux_eres2net_loss_weight,
            )
        else:
            self.aux_eres2net_loss = None

        self.use_repa_ctc_loss = use_repa_ctc_loss
        self.repa_ctc_loss_weight = repa_ctc_loss_weight
        self.use_repa_ssl_feature_loss = use_repa_ssl_feature_loss
        self.repa_ssl_feature_loss_weight = repa_ssl_feature_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _sample_time(self, batch: int, dtype, device):
        if self.t_sampling == "uniform":
            return torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # JiT: t = sigmoid(N(P_mean, P_std))
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            return torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

    def _x_to_v(self, x_pred, z, t):
        # v_pred = (x_pred - z) / (1 - t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            if self.wav_input_only:
                cond = cond
            else:
                cond = self.mel_spec(cond)
                cond = cond.permute(0, 2, 1)
                assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        cond = cond * self.latents_scale

        wav_mode = self.wav_input_only and cond.ndim == 2
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # keep legacy max_duration semantics for wav path: 4096 means 4096 frame-tokens.
        max_duration_limit = max_duration * self.wav_frame_len if wav_mode else max_duration
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration_limit)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            if wav_mode:
                test_cond = F.pad(cond, (cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            else:
                test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        if wav_mode:
            cond = F.pad(cond, (0, max_duration - cond_seq_len), value=0.0)
        else:
            cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        if wav_mode:
            step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        else:
            cond_mask = cond_mask.unsqueeze(-1)
            step_cond = torch.where(
                cond_mask, cond, torch.zeros_like(cond)
            )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration, length=max_duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            def to_v(pred_x_or_v):
                if self.prediction == "flow":
                    return pred_x_or_v
                else:  # x_pred
                    return self._x_to_v(pred_x_or_v, x, t)

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred, *_ = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    drop_audio_cond=False, drop_text=False, cache=True,
                )
                return to_v(pred)

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg, *_ = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask,
                cfg_infer=True, cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            v_cond = to_v(pred)
            v_uncond = to_v(null_pred)

            # standard CFG in v-space (Ho & Salimans style)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            if wav_mode:
                y0.append(torch.randn(dur, device=self.device, dtype=step_cond.dtype) * self.noise_scale)
            else:
                y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype) * self.noise_scale)
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        out = out / self.latents_scale
        cond_unscaled = cond / self.latents_scale
        out = torch.where(cond_mask, cond_unscaled, out)

        if exists(vocoder) and not self.wav_input_only:
            out = out.permute(0, 2, 1)
            out = vocoder(out)
            
        # ---- wav-only: flatten frames back to waveform (trim to duration*frame_len) ----
        if self.wav_input_only:
            wav_list = []
            for b in range(batch):
                wav_list.append(out[b, : duration[b].item()])
            out = pad_sequence(wav_list, batch_first=True, padding_value=0.0)  # [B, N]

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        zs: list[float["b n d"]],
        zs_lens: list[int["b"]],
        text_lens: int["b"] | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            if self.wav_input_only:
                inp = inp
            else:
                inp = self.mel_spec(inp)
                inp = inp.permute(0, 2, 1)
                assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        if self.wav_input_only and self.aux_mel_loss is not None:
            rand_span_mask = MelSpectrogramLoss._aligned_random_span_mask(
                lengths=lens,
                frac_lengths=frac_lengths,
                align_to=self.mel_align_to,
                max_length=seq_len,
            )
        else:
            rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel / raw wave is x1
        x1 = inp

        x1 = x1 * self.latents_scale

        # x0 is gaussian noise
        x0 = torch.randn_like(x1) * self.noise_scale

        # time step
        time = self._sample_time(batch, dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        if inp.ndim == 2:
            t = time.unsqueeze(-1)
        else:
            t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        if inp.ndim == 2:
            cond = torch.where(rand_span_mask, torch.zeros_like(x1), x1)
        else:
            cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        raw_pred, zs_tilde, zs_tilde_ctc = self.transformer(
            x=φ, cond=cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask,
            zs_lens=zs_lens, lens=lens,
        )

        # interpret prediction
        if self.prediction == "flow":
            v_pred = raw_pred
        elif self.prediction == "x_pred":
            x_pred = raw_pred
            v_pred = self._x_to_v(x_pred, φ, time)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # loss space
        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
        elif self.loss_space == "v":
            # v-loss (same target flow, but v_pred computed from x_pred) & use clamp_min
            denom = (1.0 - time).clamp_min(self.t_eps)
            while denom.ndim < φ.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - φ) / denom
            loss = F.mse_loss(v_pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        loss = loss[rand_span_mask]
        flow_loss = loss.mean()
        total_loss = flow_loss

        aux_mel_loss = torch.tensor(0.0, device=device)     # fixme: 这里的aux_mel_loss只针对x-pred的情况
        if self.use_aux_mel_loss and self.aux_mel_loss is not None and self.wav_input_only:
            aux_mel_mask = time > self.aux_mel_loss_start_t
            if aux_mel_mask.any():
                x1_flat_unscaled = x1[aux_mel_mask] / self.latents_scale
                x1_pred_flat_unscaled = x_pred[aux_mel_mask] / self.latents_scale

                rand_span_mask_aux = rand_span_mask[aux_mel_mask]
                lens_aux = lens[aux_mel_mask]
                aux_mel_loss = self.aux_mel_loss(
                    x1_pred_flat_unscaled,
                    x1_flat_unscaled,
                    frame_mask=rand_span_mask_aux,
                    frame_lengths=lens_aux,
                )
                total_loss = total_loss + aux_mel_loss
            
        aux_hubert_loss = torch.tensor(0.0, device=device)
        if self.use_aux_hubert_loss and self.aux_hubert_loss is not None and self.wav_input_only:
            aux_hubert_mask = time > self.aux_hubert_loss_start_t
            if aux_hubert_mask.any():
                x1_flat_unscaled = x1[aux_hubert_mask] / self.latents_scale
                x1_pred_flat_unscaled = x_pred[aux_hubert_mask] / self.latents_scale
                
                aux_hubert_loss = self.aux_hubert_loss(x1_pred_flat_unscaled, x1_flat_unscaled)
                total_loss = total_loss + aux_hubert_loss

        aux_eres2net_loss = torch.tensor(0.0, device=device)
        if self.use_aux_eres2net_loss and self.aux_eres2net_loss is not None and self.wav_input_only:
            aux_eres2net_mask = time > self.aux_eres2net_loss_start_t
            if aux_eres2net_mask.any():
                x1_flat_unscaled = x1[aux_eres2net_mask] / self.latents_scale
                x1_pred_flat_unscaled = x_pred[aux_eres2net_mask] / self.latents_scale

                aux_eres2net_loss = self.aux_eres2net_loss(x1_pred_flat_unscaled, x1_flat_unscaled)
                total_loss = total_loss + aux_eres2net_loss

        repa_ssl_feature_loss = torch.tensor(0.0, device=device)
        if self.use_repa_ssl_feature_loss and self.repa_ssl_feature_loss_weight > 0.0:
            for i, (z, z_tilde_and_z_len) in enumerate(zip(zs, zs_tilde)):
                z_tilde, z_lens = z_tilde_and_z_len
                z_mask = lens_to_mask(z_lens, length=z.shape[1]).float()
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    cos_sim = F.cosine_similarity(z_j, z_tilde_j, dim=-1)
                    repa_ssl_feature_loss += masked_mean(-cos_sim, z_mask[j])
            repa_ssl_feature_loss /= len(zs) * batch
            total_loss = total_loss + self.repa_ssl_feature_loss_weight * repa_ssl_feature_loss

        repa_ctc_loss = torch.tensor(0.0, device=device)
        if self.use_repa_ctc_loss and self.repa_ctc_loss_weight > 0.0:
            for i, z_tilde_and_z_len_ctc in enumerate(zs_tilde_ctc):
                z_tilde_ctc, z_lens_ctc = z_tilde_and_z_len_ctc
                log_probs = z_tilde_ctc.transpose(1, 0).log_softmax(-1)
                repa_ctc_loss += F.ctc_loss(
                    log_probs, text, z_lens_ctc, text_lens,
                    blank=self.transformer.text_embed.text_embed.num_embeddings,
                    reduction="mean", zero_infinity=True,  # Ignore loss if log(0) happens
                )
            repa_ctc_loss /= len(zs_tilde_ctc)
            total_loss = total_loss + self.repa_ctc_loss_weight * repa_ctc_loss

        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": aux_mel_loss,
            "aux_hubert_loss": aux_hubert_loss,
            "aux_eres2net_loss": aux_eres2net_loss,
            "repa_ssl_feature_loss": repa_ssl_feature_loss,
            "repa_ctc_loss": repa_ctc_loss,
        }

        return total_loss, cond, v_pred, loss_dict
