import math
import time
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.utilities import rank_zero_warn
from tqdm import tqdm
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from samantha.utils.hparams import DotDict
from samantha.utils.model_metric import ModelMetric

from ..inference.utils import sample


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def setup(self, stage: str) -> None:
        # Variables for MFU calculation
        self.metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj=self.model,
        )
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            if name == "mulan_centers":
                assert self.extra_params.mulan_num_rvq == self.requires["mulan_centers"].shape[0]

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def seer_rearrange(self, seq, inv=False):
        b = seq.size(0)
        if inv:
            return seq.reshape(b, -1, self.extra_params.n_seers).transpose(2, 1).reshape(b, -1)
        else:
            return seq.reshape(b, self.extra_params.n_seers, -1).transpose(2, 1).reshape(b, -1)

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        t = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch.float())
        exclude_time = time.perf_counter() - t
        logits = self.model(input_ids=input_ids)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        batch_size, seq_len = input_ids.size()[:2]
        self.metric.update(
            batch_size=batch_size,
            stage=self.trainer.state.stage,
            seq_length=seq_len,
            exclude_time=exclude_time,
        )

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.metric.compute(step=self.trainer.global_step)
            self.log_dict(metric, prog_bar=True, sync_dist=True)

        return loss, accu

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch)
        self.log_dict({"tr_loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            for l, a in outputs:
                loss += l
                accu += a
            loss /= len(outputs)
            accu /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if "ln_" in name or "bias" in name:
                print(f"Skip weight decay: {name}")
                params.append({"params": [p], "weight_decay": 0.0})
            else:
                params.append({"params": [p]})
        optimizer = self.hparams.optimizer_cls(params)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def prepare_feature(self, wavs):
        raise NotImplementedError()

    @torch.no_grad()
    def get_mulan_tokens(self, x):
        mulan_embeds = self.requires["mulan_infer_fn"](
            model=self.requires["mulan"], music=x.float(), device=x.device
        )
        mulan_tokens, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        return mulan_tokens

    @torch.no_grad()
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_wav2vec_tokens(self, x):
        wav2vec_tokens = w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x.float(),
            centers=self.requires["semantic_centers"],
            device=x.device,
        )
        return wav2vec_tokens
    
    @torch.no_grad()
    def get_wav2vec_embeds(self, x):
        b, t = x.size()
        feats, feat_mask = self.requires["ssl_frontend"](
            x, torch.LongTensor([t]).repeat([b]).to(x.device)
        )
        wav2vec_embeds, _ = self.requires["semantic"](feats, feat_mask)
        return wav2vec_embeds

    @torch.no_grad()
    def get_wav2vec_embeds_from_tokens(self, x):
        wav2vec_tokens = self.get_wav2vec_tokens(x)
        wav2vec_embeds = self.requires["semantic_centers"][wav2vec_tokens]
        return wav2vec_embeds

    @torch.no_grad()
    def get_mert_embeds(self, x):
        output_emb = self.requires["semantic"](
            x, output_hidden_states=True
        ).hidden_states[12]
        return output_emb


class SemanticModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        wav2vec_ids = self.get_wav2vec_tokens(wavs)

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + torch.arange(self.extra_params.mulan_num_rvq, device=device) * self.extra_params.mulan_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.mulan_num_rvq * self.extra_params.mulan_codebook_size
        )
        input_ids = torch.cat(
            [mulan_ids, sos_ids, wav2vec_ids[:, : -1]], dim=1
        )
        return input_ids, wav2vec_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, hp
    ):
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        mulan_ids = (
            mulan_ids
            + torch.arange(hp.mulan_num_rvq, device=device) * hp.mulan_codebook_size
            + hp.wav2vec_codebook_size
        )
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + hp.mulan_num_rvq * hp.mulan_codebook_size
            + hp.wav2vec_codebook_size
        )

        slice_range = []
        beg = 0
        while True:
            end = beg + hp.semantic_duration * hp.wav2vec_frame_rate
            if end >= hp.duration * hp.wav2vec_frame_rate:
                end = hp.duration * hp.wav2vec_frame_rate
                beg = end - (hp.semantic_duration * hp.wav2vec_frame_rate)
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += hp.semantic_stride * hp.wav2vec_frame_rate
        prev_end = 0

        semantic_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            if cache_len == 0:
                input_ids = torch.cat([mulan_ids, sos_ids], dim=1)
            else:
                prefix_semantic_samples = semantic_samples[:, cur_beg : cur_beg + cache_len]
                input_ids = torch.cat(
                    [mulan_ids, sos_ids, prefix_semantic_samples], dim=1
                )
            past_key_values = None
            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for _ in pbar:
                pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
                model_output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
                predict_logits = logits[:, -1:, :]
                samples = sample(predict_logits, temp=hp.semantic_temperature, mode=hp.sample_mode)
                input_ids = samples
                if semantic_samples is None:
                    semantic_samples = samples
                else:
                    semantic_samples = torch.cat([semantic_samples, samples], dim=1)
        return semantic_samples


class SemanticDiffusionModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        assert self.model.config.n_embd % 2 == 0
        self.rff_freq = nn.Parameter(
            16 * torch.randn([1, int(self.model.config.n_embd // 2)]),
            requires_grad=False
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            mulan_ids, sigmas_embeds, noisy_embeds, target = self.prepare_feature(batch.float())
        mulan_embeds = self.model.transformer.wte(mulan_ids)
        encoder_hidden_states = torch.cat([sigmas_embeds, mulan_embeds], dim=1)
        logits = self.model(
            inputs_embeds=noisy_embeds,
            encoder_hidden_states=encoder_hidden_states,
        )
        if isinstance(logits, dict):
            logits = logits["logits"]
        loss = self.criterion(logits, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log_dict({"tr_loss": loss}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            for l in outputs:
                loss += l
            loss /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    @torch.no_grad()
    def _get_alpha_beta(self, sigmas):
        angle = sigmas * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        wav2vec_embeds = self.get_wav2vec_embeds(wavs)

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + torch.arange(self.extra_params.mulan_num_rvq, device=device) * self.extra_params.mulan_codebook_size
        )

        sigmas = torch.rand([b, 1, 1], device=device)
        # Get noise
        noise = torch.randn_like(wav2vec_embeds)
        # Combine input and noise weighted by half-circle
        alphas, betas = self._get_alpha_beta(sigmas)
        noisy_embeds = alphas * wav2vec_embeds + betas * noise
        target = alphas * noise - betas * wav2vec_embeds
        sigmas_embeds = 2 * math.pi * sigmas * self.rff_freq
        sigmas_embeds = torch.cat(
            [torch.sin(sigmas_embeds), torch.cos(sigmas_embeds)],
            dim=-1
        )
        return mulan_ids, sigmas_embeds, noisy_embeds, target

    @torch.no_grad()
    def predict(
        self, mulan_ids, hp
    ):
        assert hp.duration == hp.semantic_duration
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        mulan_ids = (
            mulan_ids
            + torch.arange(self.extra_params.mulan_num_rvq, device=device) * self.extra_params.mulan_codebook_size
        )
        semantic_samples = torch.randn(
            (b, hp.semantic_duration * hp.wav2vec_frame_rate, hp.wav2vec_n_embd),
            device=device,
        )
        mulan_embeds = self.model.transformer.wte(mulan_ids)
        sigmas = (1.0 - torch.arange(hp.num_diffusion_steps, device=device) / (hp.num_diffusion_steps - 1))[:, None, None]
        alphas, betas = self._get_alpha_beta(sigmas)
        sigmas_embeds = 2 * math.pi * sigmas * self.rff_freq
        sigmas_embeds = torch.cat(
            [torch.sin(sigmas_embeds), torch.cos(sigmas_embeds)],
            dim=-1
        )
        pbar = tqdm(range(hp.num_diffusion_steps - 1))
        for i in pbar:
            pbar.set_description(f"Semantic [{0} - {hp.num_diffusion_steps - 1}]")
            encoder_hidden_states = torch.cat([sigmas_embeds[i : i + 1].expand((b, -1, -1)), mulan_embeds], dim=1)
            samples = self.model(
                inputs_embeds=semantic_samples,
                encoder_hidden_states=encoder_hidden_states,
            )
            if isinstance(samples, dict):
                samples = samples["logits"]
            x_pred = alphas[i] * semantic_samples - betas[i] * samples
            noise_pred = betas[i] * semantic_samples + alphas[i] * samples
            semantic_samples = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return semantic_samples


class SeerSemanticModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        wav2vec_ids = self.get_wav2vec_tokens(wavs)
        wav2vec_ids = self.seer_rearrange(wav2vec_ids)

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = mulan_ids + self.extra_params.wav2vec_codebook_size

        seer_ids = torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
        seer_ids = seer_ids + self.extra_params.wav2vec_codebook_size
    
        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")
        if wav2vec_ids.size(1) % self.extra_params.n_seers != 0:
            raise ValueError("Invalid number of seers.")

        input_ids = torch.cat([mulan_ids, seer_ids, wav2vec_ids[:, : -self.extra_params.n_seers]], dim=1)
        return input_ids, wav2vec_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, sample_len=250, temp=0.9, sample_mode="gumbel"
    ):
        if sample_len % self.model.config.n_seers != 0:
            raise ValueError("Invalid sample length.")
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        mulan_ids = (
            mulan_ids
            + self.extra_params.wav2vec_codebook_size
        )
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + self.extra_params.wav2vec_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")

        input_ids = torch.cat([mulan_ids, seer_ids], dim=1)
        kv_cache = {}
        pbar = tqdm(range(math.ceil(sample_len / self.extra_params.n_seers)))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerSemantic")
            predict_logits = self.model(input_ids, kv_cache=kv_cache)
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return self.seer_rearrange(coarse_samples, inv=True)


class MulanCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        wav2vec_ids = self.get_wav2vec_tokens(wavs)
        wav2vec_ids = wav2vec_ids + num_coarse * self.extra_params.soundstream_codebook_size

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + self.extra_params.wav2vec_codebook_size
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + 1
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )

        input_ids = torch.cat(
            [mulan_ids, sep_ids, wav2vec_ids, sos_ids, soundstream_ids[:, : -1]], dim=1
        )
        return input_ids, None, soundstream_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, wav2vec_ids, hp
    ):
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        num_coarse = hp.num_coarse
        wav2vec_ids = wav2vec_ids + num_coarse * hp.soundstream_codebook_size
        mulan_ids = (
            mulan_ids
            + hp.wav2vec_codebook_size
            + num_coarse * hp.soundstream_codebook_size
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * hp.soundstream_codebook_size
            + hp.wav2vec_codebook_size
        )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * hp.soundstream_codebook_size
            + hp.wav2vec_codebook_size
            + 1
        )

        if not hp.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * hp.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * hp.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * hp.mulan_codebook_size
            )
        else:
            sep_ids = (
                sep_ids
                + hp.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + hp.mulan_codebook_size
            )

        slice_range = []
        beg = 0
        while True:
            end = beg + hp.coarse_duration * hp.soundstream_frame_rate * num_coarse
            if end >= hp.duration * hp.soundstream_frame_rate * num_coarse:
                end = hp.duration * hp.soundstream_frame_rate * num_coarse
                beg = end - hp.coarse_duration * hp.soundstream_frame_rate * num_coarse
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += hp.coarse_stride * hp.soundstream_frame_rate * num_coarse

        prev_end = 0
        coarse_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            semantic_beg = int(cur_beg / hp.soundstream_frame_rate / num_coarse * hp.wav2vec_frame_rate)
            semantic_end = semantic_beg + hp.semantic_duration * hp.wav2vec_frame_rate
            semantic_slice = wav2vec_ids[:, semantic_beg:semantic_end]
            if cache_len == 0:
                input_ids = torch.cat(
                    [mulan_ids, sep_ids, semantic_slice, sos_ids], dim=1
                )
            else:
                prefix_coarse_samples = coarse_samples[:, cur_beg : cur_beg + cache_len]
                input_ids = torch.cat(
                    [
                        mulan_ids,
                        sep_ids,
                        semantic_slice,
                        sos_ids,
                        prefix_coarse_samples,
                    ],
                    dim=1,
                )
            kv_cache = {}
            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(
                    f"Coarse [{cur_beg} - {cur_end}] [{semantic_beg} - {semantic_end}]"
                )
                predict_logits = self.model(input_ids, kv_cache=kv_cache)
                layer_idx = i % num_coarse
                predict_logits = predict_logits[
                    :, -1:, layer_idx * hp.soundstream_codebook_size : (layer_idx + 1) * hp.soundstream_codebook_size
                ] 
                samples = sample(predict_logits, temp=hp.coarse_temperature, mode=hp.sample_mode)
                samples = samples + layer_idx * hp.soundstream_codebook_size
                input_ids = samples
                if coarse_samples is None:
                    coarse_samples = samples
                else:
                    coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples


class CoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        wav2vec_ids = self.get_wav2vec_tokens(wavs)
        wav2vec_ids = wav2vec_ids + num_coarse * self.extra_params.soundstream_codebook_size
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        input_ids = torch.cat(
            [wav2vec_ids, sos_ids, soundstream_ids[:, : -1]], dim=1
        )
        return input_ids, soundstream_ids

    @torch.no_grad()
    def predict(
        self, semantic_samples, hp
    ):
        device = semantic_samples.device
        b = semantic_samples.size(0)
        num_coarse = hp.num_coarse
        soundstream_codebook_size = hp.soundstream_codebook_size
        wav2vec_codebook_size = hp.wav2vec_codebook_size
        soundstream_frame_rate = hp.soundstream_frame_rate
        semantic_frame_rate = hp.wav2vec_frame_rate
        semantic_samples = semantic_samples + num_coarse * soundstream_codebook_size
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=semantic_samples.dtype, device=device)
            + num_coarse * soundstream_codebook_size
            + wav2vec_codebook_size
        )

        slice_range = []
        beg = 0
        while True:
            end = beg + hp.coarse_duration * soundstream_frame_rate * num_coarse
            if end >= hp.duration * soundstream_frame_rate * num_coarse:
                end = hp.duration * soundstream_frame_rate * num_coarse
                beg = end - hp.coarse_duration * soundstream_frame_rate * num_coarse
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += hp.coarse_stride * soundstream_frame_rate * num_coarse

        prev_end = 0
        coarse_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            semantic_beg = int(cur_beg / soundstream_frame_rate / num_coarse * semantic_frame_rate)
            semantic_end = semantic_beg + hp.semantic_duration * semantic_frame_rate
            semantic_slice = semantic_samples[:, semantic_beg : semantic_end]
            if cache_len == 0:
                input_ids = torch.cat([semantic_slice, sos_ids], dim=1)
            else:
                prefix_coarse_samples = coarse_samples[:, cur_beg : cur_beg + cache_len]
                input_ids = torch.cat(
                    [semantic_slice, sos_ids, prefix_coarse_samples], dim=1
                )
            past_key_values = None
            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(f"Coarse [{cur_beg} - {cur_end}] [{semantic_beg} - {semantic_end}]")
                model_output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
                layer_idx = i % num_coarse
                predict_logits = logits[:, -1:, layer_idx * soundstream_codebook_size : (layer_idx + 1) * soundstream_codebook_size]
                samples = sample(predict_logits, temp=hp.coarse_temperature, mode=hp.sample_mode)
                samples = samples + layer_idx * soundstream_codebook_size
                input_ids = samples
                if coarse_samples is None:
                    coarse_samples = samples
                else:
                    coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples


class CoarseCrossAttnModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, encoder_hidden_states, target_ids = self.prepare_feature(batch.float())
        logits = self.model(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        if isinstance(logits, dict):
            logits = logits["logits"]
        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        wav2vec_embeds = self.get_wav2vec_embeds(wavs)
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )
        input_ids = torch.cat(
            [sos_ids, soundstream_ids[:, : -1]], dim=1
        )
        return input_ids, wav2vec_embeds, soundstream_ids

    @torch.no_grad()
    def predict(
        self, semantic_samples, hp
    ):
        device = semantic_samples.device
        b = semantic_samples.size(0)
        num_coarse = hp.num_coarse
        soundstream_codebook_size = hp.soundstream_codebook_size
        soundstream_frame_rate = hp.soundstream_frame_rate
        semantic_frame_rate = hp.wav2vec_frame_rate
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=torch.long, device=device)
            + num_coarse * soundstream_codebook_size
        )

        slice_range = []
        beg = 0
        while True:
            end = beg + hp.coarse_duration * soundstream_frame_rate * num_coarse
            if end >= hp.duration * soundstream_frame_rate * num_coarse:
                end = hp.duration * soundstream_frame_rate * num_coarse
                beg = end - hp.coarse_duration * soundstream_frame_rate * num_coarse
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += hp.coarse_stride * soundstream_frame_rate * num_coarse

        prev_end = 0
        coarse_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            semantic_beg = int(cur_beg / soundstream_frame_rate / num_coarse * semantic_frame_rate)
            semantic_end = semantic_beg + hp.semantic_duration * semantic_frame_rate
            semantic_slice = semantic_samples[:, semantic_beg : semantic_end]
            if cache_len == 0:
                input_ids = sos_ids
            else:
                prefix_coarse_samples = coarse_samples[:, cur_beg : cur_beg + cache_len]
                input_ids = torch.cat(
                    [sos_ids, prefix_coarse_samples], dim=1
                )
            past_key_values = None
            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(f"Coarse [{cur_beg} - {cur_end}] [{semantic_beg} - {semantic_end}]")
                model_output = self.model(
                    input_ids=input_ids,
                    encoder_hidden_states=semantic_slice,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
                layer_idx = i % num_coarse
                predict_logits = logits[:, -1:, layer_idx * soundstream_codebook_size : (layer_idx + 1) * soundstream_codebook_size]
                samples = sample(predict_logits, temp=hp.coarse_temperature, mode=hp.sample_mode)
                samples = samples + layer_idx * soundstream_codebook_size
                input_ids = samples
                if coarse_samples is None:
                    coarse_samples = samples
                else:
                    coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples


class SemanticFreeCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = mulan_ids + num_coarse * self.extra_params.soundstream_codebook_size

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )

        input_ids = torch.cat([mulan_ids, sos_ids, soundstream_ids[:, :-1]], dim=1)
        return input_ids, soundstream_ids

    @torch.no_grad()
    def predict(self, mulan_ids, sample_len=2000, temp=0.9, sample_mode="naive"):
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()

        mulan_ids = mulan_ids + num_coarse * self.extra_params.soundstream_codebook_size
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )
        input_ids = torch.cat([mulan_ids, sos_ids], dim=1)
        self.model.transformer.init_cache()
        pbar = tqdm(range(sample_len))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerCoarse")
            logits = self.model(input_ids)
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1:, layer_idx * 1024 : (layer_idx + 1) * 1024]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        self.model.transformer.deinit_cache()
        return coarse_samples


class SeerCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])
        soundstream_ids = self.seer_rearrange(soundstream_ids)

        wav2vec_ids = (
            self.get_wav2vec_tokens(wavs)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=wav2vec_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.n_seers
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) + sep_ids.size(1) + wav2vec_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")
        if soundstream_ids.size(1) % self.extra_params.n_seers != 0:
            raise ValueError("Invalid length.")

        input_ids = torch.cat(
            [mulan_ids, sep_ids, wav2vec_ids, seer_ids, soundstream_ids[:, : -self.extra_params.n_seers]], dim=1
        )
        return input_ids, soundstream_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, wav2vec_ids, sample_len=2000, temp=0.9, sample_mode="naive"
    ):
        if sample_len % self.extra_params.n_seers != 0:
            raise ValueError("Invalid sample length.")
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()
        wav2vec_ids = wav2vec_ids + num_coarse * self.extra_params.soundstream_codebook_size
        mulan_ids = (
            mulan_ids
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        sep_ids = (
            torch.zeros(size=[b, 1], dtype=wav2vec_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.n_seers
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) + sep_ids.size(1) + wav2vec_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")

        input_ids = torch.cat([mulan_ids, sep_ids, wav2vec_ids, seer_ids], dim=1)
        self.model.transformer.init_cache()
        pbar = tqdm(range(math.ceil(sample_len / self.extra_params.n_seers)))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerCoarse")
            logits = self.model(input_ids)
            layer_idx = i % num_coarse
            predict_logits = logits[
                :, -self.extra_params.n_seers :, layer_idx * 1024 : (layer_idx + 1) * 1024
            ]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        self.model.transformer.deinit_cache()
        return self.seer_rearrange(coarse_samples, inv=True)


class FineModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()
        num_coarse, num_fine = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
        )
        soundstream_ids = self.get_soundstream_tokens(wavs)
        
        # get coarse ids
        coarse_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + (torch.arange(num_coarse, device=device) + num_fine) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            soundstream_ids[:, :, num_coarse : num_coarse + num_fine]
            + torch.arange(num_fine, device=device) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get sos id
        sos_ids = (
            torch.zeros([b, 1], dtype=soundstream_ids.dtype, device=device)
            + (num_coarse + num_fine) * self.extra_params.soundstream_codebook_size
        )
        # final input tokens
        input_tokens = torch.cat([coarse_ids, sos_ids, fine_ids[:, : -1]], dim=1)
        return input_tokens, fine_ids

    @torch.no_grad()
    def predict(
        self, coarse_samples, hp
    ):
        device = coarse_samples.device
        b, _ = coarse_samples.size()
        num_coarse = hp.num_coarse
        num_fine = hp.num_fine
        soundstream_codebook_size = hp.soundstream_codebook_size
        soundstream_frame_rate = hp.soundstream_frame_rate
        coarse_samples = coarse_samples + num_fine * soundstream_codebook_size

        sos_ids = (
            torch.zeros([b, 1], dtype=coarse_samples.dtype, device=device)
            + (num_coarse + num_fine) * soundstream_codebook_size
        )

        slice_range = []
        beg = 0
        while True:
            end = beg + hp.fine_duration * soundstream_frame_rate * num_fine
            if end >= hp.duration * soundstream_frame_rate * num_fine:
                end = hp.duration * soundstream_frame_rate * num_fine
                beg = end - hp.fine_duration * soundstream_frame_rate * num_fine
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += hp.fine_stride * soundstream_frame_rate * num_fine

        prev_end = 0
        fine_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            coarse_beg = int(cur_beg / num_fine * num_coarse)
            coarse_end = coarse_beg + hp.fine_duration * soundstream_frame_rate * num_coarse
            coarse_slice = coarse_samples[:, coarse_beg : coarse_end]
            if cache_len == 0:
                input_ids = torch.cat(
                    [coarse_slice, sos_ids], dim=1
                )
            else:
                prefix_fine_samples = fine_samples[:, cur_beg : cur_beg + cache_len]
                input_ids = torch.cat(
                    [
                        coarse_slice,
                        sos_ids,
                        prefix_fine_samples,
                    ],
                    dim=1,
                )

            past_key_values = None
            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(
                    f"Fine [{cur_beg} - {cur_end}] [{coarse_beg} - {coarse_end}]"
                )
                model_output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
                layer_idx = i % num_fine
                predict_logits = logits[
                    :, -1:, layer_idx * soundstream_codebook_size : (layer_idx + 1) * soundstream_codebook_size
                ] 
                samples = sample(predict_logits, temp=hp.fine_temperature, mode=hp.sample_mode)
                samples = samples + layer_idx * soundstream_codebook_size
                input_ids = samples
                if fine_samples is None:
                    fine_samples = samples
                else:
                    fine_samples = torch.cat([fine_samples, samples], dim=1)
        return fine_samples + num_coarse * soundstream_codebook_size


class SeerFineModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()
        num_coarse, num_fine = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
        )
        soundstream_ids = self.get_soundstream_tokens(wavs)
        
        # get coarse ids
        coarse_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + (torch.arange(num_coarse, device=device) + num_fine) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            soundstream_ids[:, :, num_coarse : num_coarse + num_fine]
            + torch.arange(num_fine, device=device) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        fine_ids = self.seer_rearrange(fine_ids)
        # get seer ids
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + num_fine * self.extra_params.soundstream_codebook_size
        )
        # final input tokens
        input_tokens = torch.cat([coarse_ids, seer_ids, fine_ids[:, : -self.extra_params.n_seers]], dim=1)
        return input_tokens, fine_ids


class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous().float()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return loss


class LanguageModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(
        self,
        config,
        logit_num,
    ):
        super().__init__(config)

        self.transformer = GPT2Model(config)

        self.lm_head = nn.Linear(
            config.n_embd, logit_num, bias=False
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def wte(self, input_ids):
        return self.transformer.wte(input_ids)

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""# noqa
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
