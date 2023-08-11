import math
import time
import sys
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import samantha.utils.hdfs_helper as hh
from einops import rearrange
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm.auto import tqdm

from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.musiclm.transforms.audio import to_energy
from recipes.musiclm.utils.dist import local_zero_first
from samantha.utils.hparams import DotDict

from recipes.musiclm.inference.utils import sample
from recipes.diffusion import rl_inference


DEFAULT_REWARDS = {"mulan_sim": 1.0, "energy_std": 1.0}


class SemanticSequenceTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        seed_model=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["seed_model"])
        self.model = model_cls()
        self.ce_criterion = MaskedCrossEntropy()
        self.extra_params = DotDict(extra_params)
        self.requires = None
        self.val_outputs = dict()
        self.text_log_counter = 0

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        semantic_type = self.extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            self.semantic_token_fn = self.get_wav2vec_tokens
        elif semantic_type == "best_rq":
            self.semantic_token_fn = self.get_best_rq_tokens
        elif semantic_type == "melspec":
            self.semantic_token_fn = self.get_melspec_tokens
        else:
            raise KeyError(f"Invalid semantic_type, got {semantic_type}")
        
        decoder_type = self.extra_params.get("decoder_type", "soundstorm")
        if decoder_type == "soundstorm":
            self.decoder_fn = self.run_soundstorm
        elif decoder_type == "diffusion":
            self.decoder_fn = self.run_diffusion
        else:
            raise KeyError(f"Invalid semantic_type, got {semantic_type}")

        if seed_model is not None:
            if seed_model.startswith("hdfs://"):
                cache_dir = '.module_cache'
                os.makedirs(cache_dir, exist_ok=True)
                local_path = f"{cache_dir}/{os.path.basename(seed_model)}"

                with local_zero_first():
                    if not os.path.exists(local_path):
                        if not hh.get(seed_model, local_path):
                            raise ConnectionError(f"Cannot retrieve file from {seed_model}.")
                seed_model = local_path

            print(f"Loading seed model from {seed_model}")
            state_dict = torch.load(seed_model, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            self.load_state_dict(state_dict=state_dict, strict=False)
            
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def setup(self, stage: str) -> None:
        if stage in {"fit", "validate"} and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        self.requires = {}
        for name, item in self.hparams.required_modules.items():
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            if name == 'soundstorm':
                # Multi-GPU fix: Current soundstorm soundstream/audio_model is scripted to use rank=0. Use direct ss_dec instead.
                self.requires['soundstorm'].audio_model = None
            if name == "mulan_centers":
                assert (
                    self.extra_params.mulan_num_rvq
                    == self.requires["mulan_centers"].shape[0]
                )

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch, mode):
        wavs = None
        text = None
        if isinstance(batch, list):
            if len(batch) == 1:
                if type(batch[0]) == list:
                    text = batch[0]
                else:
                    wavs = batch[0]
            else:
                wavs = batch[0]
                text = batch[1]
        elif isinstance(batch, dict):
            if "audio" in batch:
                wavs = batch["audio"]
            if "text" in batch:
                text = batch["text"]
        if wavs is not None:
            wavs = wavs.float()
            if wavs.dim() == 3:
                wavs = wavs.squeeze(1)
        # t = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids, mulan_ids, mulan_embeds = self.prepare_feature(wavs=wavs, text=text)
        if target_ids is not None:
            B, T = target_ids.size()
        else:
            B = mulan_ids.shape[0]
            T = self.extra_params.semantic_duration * self.extra_params.wav2vec_frame_rate
        beam = self.extra_params.beam_size

        # CE loss
        ce_loss = 0.0
        accu = 0.0
        if target_ids is not None:
            logits = self.model(**input_ids)
            if isinstance(logits, dict):
                logits = logits["logits"]
            elif isinstance(logits, tuple):
                logits = logits[0]
            # CE loss
            x = logits[:, -T:, :]
            ce_loss = self.ce_criterion(x, target_ids)
            accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        # Sequence loss
        # Inference
        ref_samples = None
        add_ref_to_beam = self.extra_params.get("add_ref_to_beam", False)
        if add_ref_to_beam and target_ids is not None and mode == "training":
            ref_samples = target_ids
        samples, mulan_ids, sos_ids = self.beam_inference(
            mulan_ids,
            self.extra_params,
            beam=beam,
            ref_samples=ref_samples,
        )
        # Compute rewards
        rewards, sampled_audio = self.get_reward({
            "samples": samples,
            "mulan_embeds": mulan_embeds,
            "mulan_ids": mulan_ids,
            "batch_size": B,
            "beam_size": beam,
        })
        rewards = rewards.reshape(B, beam)
        # Compute sequence probs
        seq_logits = self.model(
            input_ids=torch.cat([mulan_ids, sos_ids, samples[:, :-1]], dim=1),
        )
        if isinstance(seq_logits, dict):
            seq_logits = seq_logits["logits"]
        elif isinstance(seq_logits, tuple):
            seq_logits = seq_logits[0]
        seq_probs1 = F.log_softmax(seq_logits[:, -T:, :], dim=-1)
        seq_probs2 = torch.gather(
            seq_probs1, -1, samples.unsqueeze(2)
        ).squeeze(2).reshape(B, beam, -1).mean(dim=-1)
        seq_probs3 = F.softmax(seq_probs2, dim=-1)
        seq_loss = -1 * (rewards * seq_probs3).sum(dim=-1).mean()
        skip = torch.any(torch.isnan(seq_probs1)) or torch.any(torch.isnan(seq_probs3))
        if skip:
            print(f"Skipped samples: {samples}")
        return (
            ce_loss,
            accu,
            seq_loss,
            wavs,
            sampled_audio.reshape(B, beam, -1).float(),
            rewards,
            seq_probs3,
            text,
            skip,
        )

    @torch.no_grad()
    def get_reward(self, items):
        sampled_audio = self.decoder_fn(items)
        items["sampled_audio"] = sampled_audio
        reward = 0.0
        for rw_type, rw_weight in self.extra_params.get("rewards", DEFAULT_REWARDS).items():
            reward += rw_weight * self._get_reward(items, rw_type)
        return reward, sampled_audio

    @torch.no_grad()
    def _get_reward(self, items, reward_type):
        sampled_audio = items["sampled_audio"]
        b = items["batch_size"]
        beam = items["beam_size"]
        if reward_type == "mulan_sim":
            sampled_mulan_embeds = self.get_mulan_embeds(sampled_audio)
            # (b, d) --> (b * beam, d)
            mulan_embeds = items["mulan_embeds"].repeat(1, beam).reshape(b * beam, -1)
            # Compute cosine similarity, we want to maximize this
            return F.cosine_similarity(sampled_mulan_embeds, mulan_embeds)
        elif reward_type == "energy_std":
            # Compute negative relative energy std deviation, we want to maximize this
            energy = to_energy(sampled_audio, int(self.extra_params.sample_rate * 0.1))
            neg_rel_std = -1 * energy.std(dim=-1) / energy.mean(dim=-1)
            # Clip relative std to maintain loss scale
            return torch.clamp(neg_rel_std, min=-1, max=0)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    @torch.no_grad()
    def run_soundstorm(self, items):
        samples = items["samples"]
        # Run inference with SoundStorm
        sampled_tokens, _ = self.requires["soundstorm"].iterative_decoding(
            semantic_tokens=samples,
            max_seq_len=self.extra_params.duration * self.extra_params.soundstream_frame_rate,
            #iterations=[48, 24, 12, 8, 4, 4, 4, 4, 2, 2, 2, 2],
            #iterations=[16, 16, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2],
            iterations=[16, 8, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],
            score_strategies=["random"] * 4 + ["maskgit"] * 8,
            temperatures=[0.95] * 12,
        )
        sampled_audio = self.requires["ss_dec"](sampled_tokens).squeeze(1)
        return sampled_audio
        
    @torch.no_grad()
    def run_diffusion(self, items):
        samples = items["samples"]
        mulan_ids = items["mulan_ids"]
        mulan_force_cfg = self.extra_params.get('mulan_force_cfg', 1)
        sampled_audio = rl_inference.run_diffusion(
            self.requires, mulan_ids, samples, 
            diffusion_steps=self.extra_params.diffusion_steps, mulan_force_cfg=mulan_force_cfg
        )
        return sampled_audio.squeeze(1)

    def training_step(self, batch, batch_idx):
        ce_loss, accu, seq_loss, _, _, rewards, seq_probs, text, skip = self._shared_step(
            batch=batch,
            mode="training",
        )
        if text is not None and self.text_log_counter < 5:
            print(f"text: {text}")
            self.text_log_counter += 1
        self.log_dict(
            {
                "ce_loss/train": ce_loss,
                "accuracy/train": accu,
                "seq_loss/train": seq_loss,
                "reward/train_avg_mean": rewards.mean(dim=-1).mean(),
                "reward/train_avg_std": rewards.mean(dim=-1).std(),
                "reward/train_intra_beam_std": rewards.std(dim=-1).mean(),
                "reward/train_max_mean": rewards.max(dim=-1).values.mean(),
                "reward/train_max_std": rewards.max(dim=-1).values.std(),
                "seq_probs/train_max_mean": seq_probs.max(dim=-1).values.mean(),
                "seq_probs/train_max_std": seq_probs.max(dim=-1).values.std(),
            },
            prog_bar=True,
            sync_dist=True,
        )
        if skip:
            print("Skipping update due to NaN...")
            return None
        else:
            ce_weight = self.extra_params.ce_weight
            seq_weight = self.extra_params.seq_weight
            return ce_loss * ce_weight + seq_loss * seq_weight

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ce_loss, accu, seq_loss, wavs, samples, rewards, seq_probs, text, _ = self._shared_step(
            batch=batch,
            mode="validation",
        )
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(
            (ce_loss, accu, seq_loss, wavs, samples, rewards, seq_probs, text)
        )

    def on_validation_epoch_end(self):
        max_log_samples = self.extra_params.get("log_num_val_samples", None)
        max_log_beam_samples = self.extra_params.get("log_num_val_beam_samples", None)
        sample_count = 0
        for dataloader_idx, outputs in self.val_outputs.items():
            prefix = f"val_{dataloader_idx}"
            stats = {
                f"ce_loss/{prefix}": 0,
                f"accuracy/{prefix}": 0,
                f"seq_loss/{prefix}": 0,
                f"reward/{prefix}_avg_mean": 0,
                f"reward/{prefix}_avg_std": 0,
                f"reward/{prefix}_intra_beam_std": 0,
                f"reward/{prefix}_max_mean": 0,
                f"reward/{prefix}_max_std": 0,
                f"seq_probs/{prefix}_max_mean": 0,
                f"seq_probs/{prefix}_max_std": 0,
            }
            for batch_idx, (ce_l, a, seq_l, wavs, samples, rewards, seq_probs, texts) in enumerate(outputs):
                stats[f"ce_loss/{prefix}"] += ce_l
                stats[f"accuracy/{prefix}"] += a
                stats[f"seq_loss/{prefix}"] += seq_l
                stats[f"reward/{prefix}_avg_mean"] += rewards.mean(dim=-1).mean()
                stats[f"reward/{prefix}_avg_std"] += rewards.mean(dim=-1).std()
                stats[f"reward/{prefix}_intra_beam_std"] += rewards.std(dim=-1).mean()
                stats[f"reward/{prefix}_max_mean"] += rewards.max(dim=-1).values.mean()
                stats[f"reward/{prefix}_max_std"] += rewards.max(dim=-1).values.std()
                stats[f"seq_probs/{prefix}_max_mean"] += seq_probs.max(dim=-1).values.mean()
                stats[f"seq_probs/{prefix}_max_std"] += seq_probs.max(dim=-1).values.std()
                for i in range(len(wavs)):
                    if max_log_samples and sample_count >= max_log_samples: 
                        break
                    sample_count += 1

                    self.logger.experiment.add_audio(
                        f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_target",
                        wavs[i],
                        self.global_step,
                        sample_rate=self.extra_params.sample_rate,
                    )
                    # Log highest reward first
                    indices = torch.argsort(rewards[i], descending=True).cpu().tolist()
                    for j in range(len(indices)):
                        idx = indices[j]
                        if max_log_beam_samples and j >= max_log_beam_samples:
                            break
                        self.logger.experiment.add_audio(
                            f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_{j}",
                            samples[i][idx],
                            self.global_step,
                            sample_rate=self.extra_params.sample_rate,
                        )
                        if texts is not None:
                            log_text = f"Reward: {rewards[i][idx].item():.2f} \nText: \n{texts[i]}"
                        else:
                            log_text = f"Reward: {rewards[i][idx].item():.2f}"
                        self.logger.experiment.add_text(
                            f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_{j}",
                            log_text,
                            self.global_step,
                        )
            for key in stats:
                stats[key] /= len(outputs)
            self.log_dict(stats, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if "bias" in name or "layernorm" in name or "ln_" in name:
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
    def prepare_feature(self, wavs, text=None):
        assert wavs is not None or text is not None, "wavs and text can't both be None"
        if wavs is not None:
            device = wavs.device
            b = wavs.shape[0]
            if text is not None:
                mulan_ids, mulan_embeds = self.get_mulan_tokens(text, data_type="text")
            else:
                mulan_ids, mulan_embeds = self.get_mulan_tokens(wavs)
        else:
            device = self.device
            b = len(text)
            mulan_ids, mulan_embeds = self.get_mulan_tokens(text, data_type="text")

        mulan_ids_offset = (
            mulan_ids
            + torch.arange(self.extra_params.mulan_num_rvq, device=device)
            * self.extra_params.mulan_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.mulan_num_rvq * self.extra_params.mulan_codebook_size
        )

        input_ids = None
        wav2vec_ids = None
        if wavs is not None:
            wav2vec_ids = self.semantic_token_fn(wavs)
            input_ids = {
                "input_ids": torch.cat([mulan_ids_offset, sos_ids, wav2vec_ids[:, :-1]], dim=1),
            }

        return input_ids, wav2vec_ids, mulan_ids, mulan_embeds

    @torch.no_grad()
    def get_mulan_embeds(self, x, data_type="music"):
        if data_type == "music":
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"], music=x.float(), device=x.device
            )
        elif data_type == "text":
            # x should be a list of strings
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"], text=x, device=self.requires["mulan"].device
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        return mulan_embeds

    @torch.no_grad()
    def get_mulan_tokens(self, x, data_type="music"):
        mulan_embeds = self.get_mulan_embeds(x, data_type=data_type)
        mulan_tokens, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        return mulan_tokens, mulan_embeds

    @torch.no_grad()
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_wav2vec_embeds(self, x):
        b, t = x.size()
        feats, feat_mask = self.requires["ssl_frontend"](
            x, torch.LongTensor([t]).repeat([b]).to(x.device)
        )
        wav2vec_embeds, _ = self.requires["semantic"](feats, feat_mask)
        return wav2vec_embeds

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
    def get_wav2vec_embeds_from_tokens(self, x):
        wav2vec_tokens = self.get_wav2vec_tokens(x)
        wav2vec_embeds = self.requires["semantic_centers"][wav2vec_tokens]
        return wav2vec_embeds

    @torch.no_grad()
    def beam_inference(self, mulan_ids, hp, beam=1, ref_samples=None):
        """
        Input:
            mulan_ids: (batch_size, mulan_len)
            [optional] ref_samples: (batch_size, seq_len)
                If specified, add these samples to the beam

        Return a tuple of:
            semantic_samples: (batch_size * beam, seq_len)
            mulan_ids: (batch_size * beam, mulan_len)
            sos_ids: (batch_size * beam, 1)
        
        If ref_samples is not None, we'll reduce generation beam by 1 so
        the final returned beam size stays unchanged.
        """
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        mulan_ids = (
            mulan_ids
            + torch.arange(hp.mulan_num_rvq, device=device) * hp.mulan_codebook_size
            + hp.wav2vec_codebook_size
        )
        if ref_samples is not None:
            assert ref_samples.size(0) == b
            assert beam > 1, "Can't use beam size 1 with ref_samples!"
            beam = beam - 1
        # (b, s) --> (b * beam, s)
        mulan_ids = mulan_ids.repeat(1, beam).reshape(b * beam, -1)
        sos_ids = (
            torch.zeros(size=[b * beam, 1], dtype=mulan_ids.dtype, device=device)
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
                prefix_semantic_samples = semantic_samples[
                    :, cur_beg : cur_beg + cache_len
                ]
                input_ids = torch.cat(
                    [mulan_ids, sos_ids, prefix_semantic_samples], dim=1
                )
            gen_length = cur_end - cur_beg - cache_len
            past_key_values = None
            pbar = tqdm(range(gen_length), leave=False)
            for _ in pbar:
                pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
                model_output = self.model(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
                predict_logits = logits[:, -1:, : hp.wav2vec_codebook_size]
                samples, probs = sample(
                    predict_logits,
                    temp=hp.semantic_temperature,
                    mode=hp.sample_mode,
                    return_probs=True,
                )
                input_ids = samples
                if semantic_samples is None:
                    semantic_samples = samples
                else:
                    semantic_samples = torch.cat([semantic_samples, samples], dim=1)
        
        mulan_ids = (
            mulan_ids
            - torch.arange(hp.mulan_num_rvq, device=device) * hp.mulan_codebook_size
            - hp.wav2vec_codebook_size
        )
        # Add ref_samples to generation beam
        if ref_samples is not None:
            # (b * beam, s) -> (b, s) -> (b * (beam + 1), s)
            mulan_ids = mulan_ids.reshape(b, beam, -1)[:, 0, :].repeat(
                1, beam + 1
            ).reshape(b * (beam + 1), -1)
            sos_ids = sos_ids.reshape(b, beam, -1)[:, 0, :].repeat(
                1, beam + 1
            ).reshape(b * (beam + 1), -1)
            # (b * beam, s) -> (b * (beam + 1), s)
            semantic_samples = torch.cat(
                [
                    ref_samples.unsqueeze(1),
                    semantic_samples.reshape(b, beam, -1),
                ],
                dim=1,
            ).reshape(b * (beam + 1), -1)
        return semantic_samples, mulan_ids, sos_ids

    @torch.no_grad()
    def predict(self, mulan_ids, hp):
        semantic_samples, _, _ = self.beam_inference(mulan_ids, hp, beam=1)
        return semantic_samples
    

    @torch.no_grad()
    def generate_audio(self, text, hp=None):
        hp = { **self.extra_params, **hp } if hp is not None else self.extra_params
        hp = DotDict(hp)
        mulan_ids, mulan_embeds = self.get_mulan_tokens(text, data_type="text")
        beam = 1
        B = len(text)
        samples, mulan_ids, sos_ids = self.beam_inference(
            mulan_ids,
            hp,
            beam=beam,
        )
        rewards, sampled_audio = self.get_reward({
            "samples": samples,
            "mulan_embeds": mulan_embeds,
            "mulan_ids": mulan_ids,
            "batch_size": B,
            "beam_size": beam,
        })
        rewards = rewards.reshape(B, beam)
        return sampled_audio, rewards
    