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
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from wavtts.model.modules import MelSpectrogramLoss
from wavtts.model.utils import (
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
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
        joint_cond_drop_prob=0.0,
        waveform_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        prediction: str = "flow",       # "flow" | "x_pred"
        loss_space: str = "flow",       # "flow" | "v" | "x"
        t_sampling: str = "uniform",    # "uniform" | "logistic_normal"
        P_mean: float = 0.0,
        P_std: float = 1.0,
        time_shift: float = 1.0,
        t_eps: float = 1e-4,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        aux_mel_loss_masked: bool = True,
        sample_rate: int = 16000,
        latents_scale: float = 1.0,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        
        # waveform geometry
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 160))
        self.num_channels = self.wav_frame_len

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.joint_cond_drop_prob = joint_cond_drop_prob

        # transformer
        self.transformer = transformer
        self.dim = transformer.dim

        if hasattr(self.transformer, "set_wav_frame_len"):
            self.transformer.set_wav_frame_len(self.wav_frame_len)

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
        self.time_shift = time_shift
        self.t_eps = t_eps
        self.latents_scale = latents_scale
        self.target_sample_rate = sample_rate
        # aux mel loss
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_masked = aux_mel_loss_masked
        if self.use_aux_mel_loss:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],
                mel_fmax=[None] * 7,
                pow=1.0,
                clamp_eps=1e-5,
                weight=aux_mel_loss_weight,
            )
        else:
            self.aux_mel_loss = None

        self.mask_align_to = 1
        alignments = []
        if self.aux_mel_loss is not None and hasattr(self.aux_mel_loss, "mel_transforms"):
            hop_lengths = [int(m.hop_length) for m in self.aux_mel_loss.mel_transforms]
            if len(hop_lengths) > 0:
                lcm_hop = hop_lengths[0]
                for h in hop_lengths[1:]:
                    lcm_hop = math.lcm(lcm_hop, h)
                alignments.append(lcm_hop)
        if len(alignments) > 0:
            lcm_align = alignments[0]
            for a in alignments[1:]:
                lcm_align = math.lcm(lcm_align, a)
            self.mask_align_to = max(1, lcm_align)
            
    @property
    def device(self):
        return next(self.parameters()).device
    
    def _sample_time(self, batch: int, dtype, device):
        if self.t_sampling == "uniform":
            t = torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # JiT: t = sigmoid(N(P_mean, P_std))
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

        if self.time_shift != 1.0:
            t = t / (t + self.time_shift * (1 - t))

        return t

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
        timestep_mapping="sway_sampling",
        timestep_power=None,
        shift=1.0,
        seed: int | None = None,
        max_duration=4096,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        if cond.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform conditioning [B, N], got {tuple(cond.shape)}")

        cond = cond.to(next(self.parameters()).dtype)
        cond = cond * self.latents_scale

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
        max_duration_limit = max_duration * self.wav_frame_len
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration_limit)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

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

            return v_cond + (v_cond - v_uncond) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        use_epss = use_epss and timestep_mapping == "sway_sampling"

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float32)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float32)

        if timestep_mapping == "uniform":
            pass
        elif timestep_mapping == "sway_sampling":
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        elif timestep_mapping == "power":
            if timestep_power is None:
                raise ValueError("timestep_power must be provided when timestep_mapping='power'")
            t = t.pow(timestep_power)
        else:
            raise ValueError(f"Unknown timestep_mapping: {timestep_mapping}")

        effective_shift = shift
        if effective_shift != 1.0:
            t = t / (t + effective_shift * (1 - t))

        # WavTTS inference uses Euler ODE sampling.
        odeint_kwargs = dict(self.odeint_kwargs)
        odeint_kwargs["method"] = "euler"
        trajectory = odeint(fn, y0, t, **odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        out = out / self.latents_scale
        cond_unscaled = cond / self.latents_scale
        out = torch.where(cond_mask, cond_unscaled, out)

        # ---- wav-only: trim to generated waveform duration ----
        wav_list = []
        for b in range(batch):
            wav_list.append(out[b, : duration[b].item()])
        out = pad_sequence(wav_list, batch_first=True, padding_value=0.0)  # [B, N]

        return out, trajectory

    def forward(
        self,
        inp: float["b nw"],  # raw waveform
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw waveform
        if inp.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform input [B, N], got {tuple(inp.shape)}")

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
        if self.mask_align_to > 1:
            rand_span_mask = MelSpectrogramLoss._aligned_random_span_mask(
                lengths=lens,
                frac_lengths=frac_lengths,
                align_to=self.mask_align_to,
                max_length=seq_len,
            )
        else:
            rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # x1 is raw waveform
        x1 = inp

        x1 = x1 * self.latents_scale

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = self._sample_time(batch, dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask, torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        if self.joint_cond_drop_prob > 0.0:
            joint_drop = random() < self.joint_cond_drop_prob
            drop_audio_cond = joint_drop
            drop_text = joint_drop
        else:
            drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
            if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
                drop_audio_cond = True
                drop_text = True
            else:
                drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        raw_pred, *_ = self.transformer(
            x=φ, cond=cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask,
            lens=lens,
        )

        # interpret prediction
        if self.prediction == "flow":
            v_pred = raw_pred
            x_pred = φ + (1.0 - t) * v_pred
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
        elif self.loss_space == "x":
            loss = F.mse_loss(x_pred, x1, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        loss = loss[rand_span_mask]
        flow_loss = loss.mean()
        total_loss = flow_loss

        aux_mel_loss = torch.tensor(0.0, device=device)
        if self.use_aux_mel_loss and self.aux_mel_loss is not None:
            x1_flat_unscaled = x1 / self.latents_scale
            x1_pred_flat_unscaled = x_pred / self.latents_scale

            aux_mel_kwargs = {}
            if self.aux_mel_loss_masked:
                aux_mel_kwargs.update(
                    frame_mask=rand_span_mask,
                    frame_lengths=lens,
                )

            aux_mel_loss = self.aux_mel_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                **aux_mel_kwargs,
            )
            total_loss = total_loss + aux_mel_loss
            
        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": aux_mel_loss,
        }

        return total_loss, cond, v_pred, loss_dict
