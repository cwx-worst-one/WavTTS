from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import (
    MulanEmbedder,
    LyricsTokenEmbedder,
    WavToVecTokenEmbedder,
    MetadataT5TokenEmbedder,
    SpeakerEmbedder,
    BestRQTokenEmbedder, 
    MulanTagEmbedder,
    DurationEmbedder,
    StructureEmbedder,
    IntensityEmbedder,
    get_mulan_embeds,
)
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
from recipes.bigmusic.utils.rewards import (
    mulan_audio_reward,
    mulan_text_reward,
    wer_reward,
    loudness_reward,
    chord_reward,
    nonvocal_reward,
    structure_reward,
    chorus_sim_reward,
    chorus_presence_reward,
)
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from samantha.utils.hparams import DotDict
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.musiclm.transforms.audio import to_energy
from collections import defaultdict
from itertools import zip_longest

# from recipes.umm.models.bestrq import BestRQMelCTC
from recipes.umm.modules.lit_module import (
    BestRQMelCTC,
    Stage3,
)

from recipes.audio_quality_classifier.models.audio_quality_model.utils import aq_classifier_inference

DEFAULT_REWARDS = {"mulan_sim": 1.0, "wer": 1.0}


class SemanticModule(BaseContinuousEmbedModule):
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
        hidden_size = extra_params['hidden_size']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        embedder_dict = {}
        for emb_type in extra_params.get("input_embedders", ["mulan", "lyrics_tokens"]):
            if emb_type == "mulan":
                mulan_embed_dim = extra_params['mulan_embed_dim']
                mulan_OTF_tag_type = extra_params.get('mulan_tag_type', 'mulan_genres')
                mulan_crop = extra_params.get('mulan_crop', True)
                mulan_average = extra_params.get('mulan_average', True)
                embedder_dict[emb_type] = MulanTagEmbedder(
                    input_dim=mulan_embed_dim,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    mulan_tag_type=mulan_OTF_tag_type,
                    mulan_crop=mulan_crop,
                    mulan_average=mulan_average,
                )
            elif emb_type == "lyrics_tokens":
                lyrics_vocab_size = extra_params['lyrics_codebook_size']
                embedder_dict[emb_type] = LyricsTokenEmbedder(
                    vocab_size=lyrics_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=False,
                )
            elif emb_type == "duration":
                embedder_dict[emb_type] = DurationEmbedder(
                    durations=extra_params["duration"],
                    embedding_dim=hidden_size,
                )
            elif emb_type == "structure":
                embedder_dict[emb_type] = StructureEmbedder(
                    durations=extra_params["duration"],
                    embedding_dim=hidden_size,
                    structure_labels=extra_params["structure_labels"],
                    granularity_in_secs=extra_params["granularity_in_secs"],
                )
            elif emb_type == "intensity":
                embedder_dict[emb_type] = IntensityEmbedder(
                    decimals=extra_params["intensity_decimals"],
                    embedding_dim=hidden_size,
                )
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        else:
            raise NotImplementedError

        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()
        self.log_counter = 0
        self.mulan_counter = 0

    def infer_target_duration(self, batch):
        if "duration" in batch:
            target_duration = batch["duration"]
        else:
            target_duration = batch["target_audio"].shape[-1] // self.extra_params.sample_rate
        return target_duration

    def prepare_mulan_inputs(self, batch, mulan_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        target_duration = self.infer_target_duration(batch)
        target_samples_length = target_duration * self.extra_params.sample_rate
        if 'style_text' in conditions:
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_text'],
                with_sos=True,
                data_type='text',
                target_samples_length=target_samples_length,
            )
        elif 'style_audio' in conditions:
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_audio'].to(self.device),
                with_sos=True,
                data_type='music',
                target_samples_length=target_samples_length,
            )
        elif 'style_tag' in conditions: # using Mulan for on-the-fly MIR tagging
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_audio'].to(self.device),
                mcc_style_text=batch.get('style_text'),
                with_sos=True,
                data_type='tag',
                target_samples_length=target_samples_length,
            )
        else:
            # adding SOS token no matter what so that all parameters get used
            embeds = mulan_embedder.get_sos_embed(batch_size)
        if self.mulan_counter < 10:
            print(f"target_duration: {target_duration}, mulan_emb: {embeds.shape}")
            self.mulan_counter += 1
        return embeds

    def prepare_lyrics_inputs(self, batch, lyrics_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'lyrics_tokens' in conditions:
            embeds = lyrics_embedder.embed(
                self.requires,
                batch['lyrics_tokens'].to(self.device),
                with_sos=True,
            )
        else:
            embeds = lyrics_embedder.get_sos_embed(batch_size)
        return embeds

    def prepare_duration_inputs(self, batch, duration_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'duration' in conditions:
            embeds = duration_embedder.embed(batch["duration"], batch_size)
        else:
            embeds = duration_embedder.empty_embed(batch_size)
        return embeds

    def prepare_structure_inputs(self, batch, structure_embedder):
        batch_size = self.infer_batch_size(batch)
        structure_labels = batch["structure"]
        assert batch_size == len(structure_labels)
        # Dropout if needed
        structure_dropout = self.extra_params.get("structure_dropout", 0.0)
        if self.training and structure_dropout > 0:
            all_keep = [x >= structure_dropout for x in np.random.rand(batch_size)]
            for i, keep in enumerate(all_keep):
                if not keep:
                    structure_labels[i] = None
        # Find target duration
        target_duration = self.infer_target_duration(batch)
        embeds = structure_embedder.embed(structure_labels, target_duration)
        return embeds

    def prepare_intensity_inputs(self, batch, intensity_embedder):
        batch_size = self.infer_batch_size(batch)
        intensity_labels = batch["intensity"]
        assert batch_size == len(intensity_labels)
        target_duration = self.infer_target_duration(batch)
        embeds = intensity_embedder.embed(intensity_labels, target_duration)
        return embeds

    def prepare_inputs_embeddings(self, batch):
        if self.log_counter < 1:
            print(batch)
            self.log_counter += 1

        inputs_embeds = []
        for emb_type, embedder in self.input_embedders.items():
            if emb_type == "mulan":
                emb_inputs = self.prepare_mulan_inputs(batch, embedder)
            elif emb_type == "lyrics_tokens":
                emb_inputs = self.prepare_lyrics_inputs(batch, embedder)
            elif emb_type == "duration":
                emb_inputs = self.prepare_duration_inputs(batch, embedder)
            elif emb_type == "structure":
                emb_inputs = self.prepare_structure_inputs(batch, embedder)
            elif emb_type == "intensity":
                emb_inputs = self.prepare_intensity_inputs(batch, embedder)
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
            inputs_embeds.append(emb_inputs)
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        duration = batch["duration"] if "duration" in batch else hp.duration
        num_tokens = duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode
        sample_thresh = hp.get('sample_thresh', 0.9)

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(
            inputs_embeds,
            num_tokens,
            temperature=temperature,
            beam=beam,
            sample_mode=sample_mode,
            sample_thresh=sample_thresh,
            ref_samples=ref_samples,
        )

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)


class SemanticRLModule(SemanticModule):
    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=MaskedCrossEntropy,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        decoder_type = self.extra_params.get("decoder_type", "diffusion")
        if decoder_type == "diffusion":
            self.decoder_fn = self.run_diffusion
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        self.val_outputs = dict()
        self.positive_qualitative_emb = None
        self.negative_qualitative_emb = None

    def _shared_step(self, batch, mode):
        wavs_gt = batch["target_audio"]
        if wavs_gt.dim() == 2:
            wavs_gt = wavs_gt.unsqueeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            model_inputs, target_ids, inputs_embeds, _, _ = self.prepare_training_inputs(batch, return_all=True)
        B, T = target_ids.size()
        beam = self.extra_params.beam_size

        # CE loss
        logits = self.model(**model_inputs)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -T:, :]
        ce_loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        # Sequence loss
        # Inference
        ref_samples = None
        if self.extra_params.add_ref_to_beam and mode == "training":
            ref_samples = target_ids
        if "duration" in batch:
            duration = batch["duration"]
        elif isinstance(self.extra_params.duration, (list, tuple)):
            duration = self.extra_params.duration[-1]
        else:
            duration = self.extra_params.duration
        num_tokens = duration * self.extra_params.semantic_frame_rate
        sampled_semantic_tokens, model_inputs = self.super_predict(
            inputs_embeds=inputs_embeds,
            num_tokens=num_tokens,
            temperature=self.extra_params.semantic_temperature,
            sample_mode=self.extra_params.sample_mode,
            beam=beam,
            ref_samples=ref_samples,
            rl_training=True,
        )
        sampled_semantic_tokens_processed, eos_index_list = process_eos_indexes(
            sampled_semantic_tokens,
            self,
            sample_rate=self.extra_params.sample_rate,
        )
        # Compute rewards
        rewards, sampled_audio, reward_breakdown = self.get_reward({
            "sampled_semantic_tokens": sampled_semantic_tokens_processed,
            "eos_index_list": eos_index_list,
            "target_audio": wavs_gt,
            "batch_size": B,
            "beam_size": beam,
            "batch": batch,
        })
        # Compute sequence probs
        seq_logits = self.model(**model_inputs)
        if isinstance(seq_logits, dict):
            seq_logits = seq_logits["logits"]
        elif isinstance(seq_logits, tuple):
            seq_logits = seq_logits[0]
        seq_probs1 = F.log_softmax(seq_logits[:, -num_tokens:, :], dim=-1)
        seq_probs2 = torch.gather(
            seq_probs1, -1, sampled_semantic_tokens.unsqueeze(2)
        ).squeeze(2)    # (B * beam, T)
        # Only add up non-eos probs
        if len(eos_index_list) > 0:
            token2wav_rate = self.extra_params.sample_rate // self.extra_params.semantic_frame_rate
            seq_len = eos_index_list // token2wav_rate + 1  # need to add 1 to include <eos>
            for i in range(len(eos_index_list)):
                seq_probs2[i, seq_len[i]:] = 0
            seq_probs2 = (seq_probs2.sum(dim=-1) / seq_len).reshape(B, beam)
        else:
            seq_probs2 = seq_probs2.reshape(B, beam, -1).mean(dim=-1)
        seq_probs3 = F.softmax(seq_probs2, dim=-1)
        seq_loss = -1 * (rewards * seq_probs3).sum(dim=-1).mean()
        skip = torch.any(torch.isnan(seq_probs1)) or torch.any(torch.isnan(seq_probs3))
        if skip:
            print(f"Skipped samples: {sampled_semantic_tokens}")
        return (
            ce_loss,
            accu,
            seq_loss,
            wavs_gt,
            sampled_audio,
            rewards,
            reward_breakdown,
            seq_probs3,
            skip,
        )

    def training_step(self, batch, batch_idx):
        ce_loss, accu, seq_loss, _, _, _, reward_breakdown, seq_probs, skip = self._shared_step(
            batch=batch,
            mode="training",
        )
        stats = {
            "ce_loss/train": ce_loss,
            "accuracy/train": accu,
            "seq_loss/train": seq_loss,
            "seq_probs/train_max_mean": seq_probs.max(dim=-1).values.mean(),
            "seq_probs/train_max_std": seq_probs.max(dim=-1).values.std(),
        }
        for rw_type, rw in reward_breakdown.items():
            stats.update(
                {
                    f"reward_{rw_type}/train_avg_mean": rw.mean(dim=-1).mean(),
                    f"reward_{rw_type}/train_avg_std": rw.mean(dim=-1).std(),
                    f"reward_{rw_type}/train_intra_beam_std": rw.std(dim=-1).mean(),
                    f"reward_{rw_type}/train_max_mean": rw.max(dim=-1).values.mean(),
                    f"reward_{rw_type}/train_max_std": rw.max(dim=-1).values.std(),
                }
            )
        self.log_dict(stats, prog_bar=True, sync_dist=True)
        if skip:
            print("Skipping update due to NaN...")
            return None
        else:
            ce_weight = self.extra_params.ce_weight
            seq_weight = self.extra_params.seq_weight
            return ce_loss * ce_weight + seq_loss * seq_weight

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(
            self._shared_step(
                batch=batch,
                mode="validation",
            )
        )

    def on_validation_epoch_end(self):
        max_log_samples = self.extra_params.get("log_num_val_samples", None)
        max_log_beam_samples = self.extra_params.get("log_num_val_beam_samples", None)
        sample_count = 0
        for dataloader_idx, outputs in self.val_outputs.items():
            prefix = f"val_{dataloader_idx}"
            stats = defaultdict(int)
            for batch_idx, (ce_l, a, seq_l, wavs_gt, wavs_sampled, rewards, reward_breakdown, seq_probs, _) in enumerate(outputs):
                stats[f"ce_loss/{prefix}"] += ce_l
                stats[f"accuracy/{prefix}"] += a
                stats[f"seq_loss/{prefix}"] += seq_l
                stats[f"seq_probs/{prefix}_max_mean"] += seq_probs.max(dim=-1).values.mean()
                stats[f"seq_probs/{prefix}_max_std"] += seq_probs.max(dim=-1).values.std()
                for rw_type, rw in reward_breakdown.items():
                    stats[f"reward_{rw_type}/{prefix}_avg_mean"] += rw.mean(dim=-1).mean()
                    stats[f"reward_{rw_type}/{prefix}_avg_std"] += rw.mean(dim=-1).std()
                    stats[f"reward_{rw_type}/{prefix}_intra_beam_std"] += rw.std(dim=-1).mean()
                    stats[f"reward_{rw_type}/{prefix}_max_mean"] += rw.max(dim=-1).values.mean()
                    stats[f"reward_{rw_type}/{prefix}_max_std"] += rw.max(dim=-1).values.std()
                for i in range(len(wavs_gt)):
                    if max_log_samples and sample_count >= max_log_samples: 
                        break
                    sample_count += 1

                    self.logger.experiment.add_audio(
                        f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_target",
                        wavs_gt[i],
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
                            wavs_sampled[i][idx],
                            self.global_step,
                            sample_rate=self.extra_params.sample_rate,
                        )
                        log_text = ""
                        for rw_type, rw in reward_breakdown.items():
                            log_text += f"{rw_type}={rw[i][idx].item():.2f} "
                        self.logger.experiment.add_text(
                            f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_{j}",
                            log_text,
                            self.global_step,
                        )
            for key in stats:
                stats[key] /= len(outputs)
            self.log_dict(stats, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    @torch.no_grad()
    def run_diffusion(self, items):
        VOCODER_HZ = 125
        sampler = self.requires["sampler"]
        diffusion = self.requires["diffusion"]
        vocoder = self.requires["vocoder"]
        semantic_tokens = items["sampled_semantic_tokens"]
        eos_index_list = items["eos_index_list"]
        # diffusion sampling
        pred_emb = sampler(
            model=diffusion,
            semantic_context=semantic_tokens,
            num_items=semantic_tokens.shape[0],
            num_chunks=1,
            num_steps=self.extra_params.diffusion_steps,
            bf16_portion=self.extra_params.bf16_portion,
            #start=None,
            #show_progress=False,
            angle_schedule="linear",
            schdeule_slope=2.5,
            classifier_free_guidance=2.5,
        ).detach().float()
        # torch.interpolate causes OOM for large batch sizes > 24. chunking to batch of 8 instead.
        # If you see this error, lower batch size:
        # RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.
        duration = pred_emb.shape[-1] // VOCODER_HZ
        batch_chunks = max(1, 240 // duration)
        wavs = torch.cat([vocoder.decode(c).detach() for c in torch.split(pred_emb, batch_chunks)])
        # Zero out samples after <eos>
        if len(eos_index_list) > 0:
            for i in range(len(eos_index_list)):
                wavs[i, :, eos_index_list[i]:] = 0
        return wavs

    @torch.no_grad()
    def get_reward(self, items):
        b = items["batch_size"]
        beam = items["beam_size"]
        batch = items["batch"]
        sampled_audio = self.decoder_fn(items).float()
        if "duration" in batch:
            max_samples = batch["duration"] * self.extra_params.sample_rate
            sampled_audio = sampled_audio[..., :max_samples]
        items["sampled_audio"] = sampled_audio
        reward = 0.0
        reward_breakdown = {}
        for rw_type, rw_weight in self.extra_params.rewards.items():
            if rw_type == "style_sim":
                conditions = self.infer_conditions(batch)
                rw_type = "style_text_sim" if "style_text" in conditions else "mulan_sim"
            rw = self._get_reward(items, rw_type).reshape(b, beam)
            reward += rw_weight * rw
            reward_breakdown[rw_type] = rw
        return reward, sampled_audio.reshape(b, beam, -1), reward_breakdown

    @torch.no_grad()
    def _get_reward(self, items, reward_type):
        sampled_audio = items["sampled_audio"]
        target_audio = items["target_audio"]
        b = items["batch_size"]
        beam = items["beam_size"]
        batch = items["batch"]
        if reward_type == "mulan_sim":
            (
                mulan_sim,
                items["sampled_mulan_embeds"],
                items["target_mulan_embeds"],
            ) = mulan_audio_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                target_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
                sampled_embeds=items.get("sampled_mulan_embeds"),
                target_embeds=items.get("target_mulan_embeds"),
            )
            return mulan_sim
        elif reward_type == "wer":
            if "sampled_lyrics" not in items:
                eos_index_list = items["eos_index_list"]
                items["sampled_lyrics"] = asr_transcribe_lyrics(
                    self.requires,
                    sampled_audio,
                    sample_rate=self.extra_params.sample_rate,
                    sample_lengths=None if len(eos_index_list) == 0 else eos_index_list
                )
            sampled_lyrics = items["sampled_lyrics"]
            if len(sampled_lyrics) != sampled_audio.size(0):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(sampled_lyrics)} (expected {sampled_audio.size(0)}), content={sampled_lyrics}")
                return 0
            return wer_reward(
                sampled_lyrics,
                batch["lyrics"] if "lyrics" in batch else batch["lyrics_text"],
                sampled_audio.device,
            )
        elif reward_type == "style_text_sim":
            (
                style_text_sim,
                items["sampled_mulan_embeds"],
                items["style_text_embeds"],
            ) = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                batch["style_text"],
                device=sampled_audio.device,
                sampled_embeds=items.get("sampled_mulan_embeds"),
                target_embeds=items.get("style_text_embeds"),
            )
            return style_text_sim
        elif reward_type == "loudness_sim":
            return loudness_reward(
                sampled_audio,
                target_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "qualitative_sim":
            qualitative_reward = 0
            if self.extra_params.positive_phrase is not None:
                (
                    positive_sim,
                    items["sampled_mulan_embeds"],
                    self.positive_qualitative_emb,
                ) = mulan_text_reward(
                    self.requires["mulan_infer_fn"],
                    self.requires["mulan"],
                    sampled_audio.squeeze(1),
                    [self.extra_params.positive_phrase],
                    device=sampled_audio.device,
                    sampled_embeds=items.get("sampled_mulan_embeds"),
                    target_embeds=self.positive_qualitative_emb,
                )
                qualitative_reward += positive_sim
            if self.extra_params.negative_phrase is not None:
                (
                    negative_sim,
                    items["sampled_mulan_embeds"],
                    self.negative_qualitative_emb,
                ) = mulan_text_reward(
                    self.requires["mulan_infer_fn"],
                    self.requires["mulan"],
                    sampled_audio.squeeze(1),
                    [self.extra_params.negative_phrase],
                    device=sampled_audio.device,
                    sampled_embeds=items.get("sampled_mulan_embeds"),
                    target_embeds=self.negative_qualitative_emb,
                )
                qualitative_reward -= negative_sim
            return qualitative_reward
        elif reward_type == "nonvocal":
            if "sampled_lyrics" not in items:
                eos_index_list = items["eos_index_list"]
                items["sampled_lyrics"] = asr_transcribe_lyrics(
                    self.requires,
                    sampled_audio,
                    sample_rate=self.extra_params.sample_rate,
                    sample_lengths=None if len(eos_index_list) == 0 else eos_index_list
                )
            sampled_lyrics = items["sampled_lyrics"]
            if len(sampled_lyrics) != sampled_audio.size(0):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(sampled_lyrics)} (expected {sampled_audio.size(0)}), content={sampled_lyrics}")
                return 0
            return nonvocal_reward(sampled_lyrics, device=sampled_audio.device)
        elif reward_type == "chord":
            # Use genre-specific chord LM if possible, otherwise fall back to default LM
            chord_lm_keys = []
            for i in range(sampled_audio.size(0)):
                style_metadata = batch.get("style_metadata")
                genres = []
                if style_metadata is not None:
                    genres = style_metadata[i // beam].get("genres", [])
                    genres = ["_".join(g.lower().split()) for g in genres]
                    genres = [g for g in genres if g in self.requires["chord_lms"]]
                if len(genres) == 0 and "default" in self.requires["chord_lms"]:
                    genres = ["default"]
                # Final filter
                genres = [g for g in genres if self.requires["chord_lms"][g] is not None]
                chord_lm_keys.append(genres)
            return chord_reward(
                self.requires["chord"],
                self.requires["chord_lms"],
                sampled_audio,
                chord_lm_keys=chord_lm_keys,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "structure":
            return structure_reward(
                self.requires["structure"],
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "chorus_sim":
            return chorus_sim_reward(
                sampled_audio,
                batch["structure"],
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "chorus_presence":
            return chorus_presence_reward(
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


class SemanticT5Module(BaseContinuousEmbedModule):
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
        
        hidden_size = extra_params['hidden_size']
        lyrics_vocab_size = extra_params['lyrics_codebook_size']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            'style_tokens': MetadataT5TokenEmbedder(embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        else:
            raise NotImplementedError
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def prepare_inputs_embeddings(self, batch):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        with_sos=True
        # convert inputs to conditions
        inputs_embeds = []
        if 'lyrics_tokens' in conditions:
            embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['lyrics_tokens'].get_sos_embed(batch_size))
        if 'style_tokens' in conditions:
            embeds = self.input_embedders['style_tokens'].embed(self.requires, batch['style_tokens'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(self.input_embedders['style_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

def process_eos_indexes(semantic_samples, semantic_module: SemanticModule, sample_rate=24000):
    semantic_frame_rate = semantic_module.extra_params.semantic_frame_rate
    eos_id = semantic_module.target_embedder.eos_id
    eos_index_list = []
    if eos_id is not None:
        eos_padding_id = 0
        eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
        semantic_samples[eos_mask] = eos_padding_id
        token2wav_rate = int(sample_rate / semantic_frame_rate)
        eos_index_list = ((semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0).bool().sum(axis=1) * token2wav_rate
    return semantic_samples, eos_index_list

def truncate_wav_to_eos(wavs, eos_index_list):
    truncated_wavs = []
    for i, (eos, wav) in enumerate(zip_longest(eos_index_list, wavs)):
        if eos is not None:
            wav = wav[:eos]
        truncated_wavs.append(wav)
    return truncated_wavs