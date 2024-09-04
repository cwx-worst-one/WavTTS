import torch

import torchaudio
from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
from recipes.bigmusic.utils.rewards import (
    mulan_audio_reward,
    mulan_text_reward,
    wer_reward,
    chord_reward,
    nonvocal_reward,
    structure_reward,
    chorus_sim_reward,
    chorus_presence_reward,
    loudness_reward,
    audio_metrics_reward,
    semantic_diversity_reward,
    chroma_reward,
    anchor_points_sim_reward,
    mulan_temporal_reward,
    chroma_temporal_reward
)


def infer_conditions(batch):
    if type(batch["conditions"]) == list:
        assert (
            len(set(list(map(tuple, batch["conditions"])))) == 1
        ), "Make sure that all conditions in the batch are the same"
        conditions = batch['conditions'][0].split(',')
    else:
        conditions = batch['conditions'].split(',')
    return conditions


TARGET_SAMPLE_RATE=24000
class Reranker:
    def __init__(
        self,
        rewards,
        required_modules,
        local_rank,
        cache_dir,
        sample_rate=24000,
    ):
        self.rewards = rewards
        self.local_rank = local_rank
        self.requires = {}
        for _, module in required_modules.items():
            self.requires.update(
                module["initializer"](
                    module["hpath"],
                    local_rank=local_rank,
                    cache_dir=cache_dir,
                )
            )
        if any([x in rewards for x in ["style_audio", "style_text", "qualitative", "qualitative_cn", "style_sim"]]):
            assert "mulan" in self.requires
            assert "mulan_infer_fn" in self.requires
        if "structure" in rewards:
            assert "structure" in self.requires
        if "chord" in rewards:
            assert "chord" in self.requires
            assert "chord_lms" in self.requires

        self.device = torch.device(f'cuda:{local_rank}')
        if sample_rate != TARGET_SAMPLE_RATE:
            self.resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE).to(self.device)

    def compute_rewards(self, sampled_audio, eos_index_list, batch, extra_params, sample_rate):
        rewards = torch.zeros(len(sampled_audio)).to(self.device)
        rewards_breakdown = [{ 'weighted': {}, 'unweighted': {} } for _ in range(len(sampled_audio))]
        for rw_type, rw_weight in self.rewards.items():
            if rw_weight == 0:
                continue
            if rw_type == "style_sim":
                conditions = infer_conditions(batch)
                rw_type = "style_text" if "style_text" in conditions else "style_audio"
            rw = self._get_reward(rw_type, sampled_audio, eos_index_list, batch, extra_params, sample_rate)
            rewards += rw_weight * rw
            for i in range(len(sampled_audio)):
                rewards_breakdown[i]['unweighted'][rw_type] = rw[i].item()
                rewards_breakdown[i]['weighted'][rw_type] = rw[i].item() * rw_weight
                rewards_breakdown[i]['weighted']['total'] = rewards[i].item()
        return rewards, rewards_breakdown

    def _get_reward(
        self,
        rw_type,
        sampled_audio,
        eos_index_list,
        batch,
        extra_params,
        sample_rate=24000
    ):
        sampled_audio = sampled_audio.to(self.device)
        if rw_type == "style_audio":
            return mulan_audio_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                batch["style_audio"],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )[0]
        elif rw_type == "style_text":
            return mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                batch["style_text"],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )[0]
        elif rw_type == "qualitative":
            positive_phrase = extra_params.get(
                "positive_phrase",
                "cd quality, catchy, memorable",
            )
            positive_reward = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                [positive_phrase],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )[0]
            negative_phrase = extra_params.get(
                "negative_phrase",
                "noisy, boring, forgettable",
            )
            negative_reward = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                [negative_phrase],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )[0]
            return positive_reward - negative_reward
        elif rw_type == "wer":
            lyrics_hyp = asr_transcribe_lyrics(
                self.requires,
                sampled_audio,
                sample_lengths=None if len(eos_index_list) == 0 else eos_index_list,
                sample_rate=sample_rate,
            )
            if len(lyrics_hyp) != len(sampled_audio):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(lyrics_hyp)} (expected {len(sampled_audio)}), content={lyrics_hyp}")
                return 0
            return wer_reward(
                lyrics_hyp,
                [normalize_text(x) for x in batch["lyrics"]],
                device=sampled_audio.device,
            )
        elif rw_type == "nonvocal":
            lyrics_hyp = asr_transcribe_lyrics(
                self.requires,
                sampled_audio,
                sample_lengths=None if len(eos_index_list) == 0 else eos_index_list,
                sample_rate=sample_rate,
            )
            if len(lyrics_hyp) != len(sampled_audio):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(lyrics_hyp)} (expected {len(sampled_audio)}), content={lyrics_hyp}")
                return 0
            return nonvocal_reward(lyrics_hyp, device=sampled_audio.device)
        elif rw_type == "structure":
            return structure_reward(
                self.requires["structure"],
                sampled_audio,
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "chorus_sim":
            return chorus_sim_reward(
                sampled_audio,
                batch["structure"],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "chorus_presence":
            return chorus_presence_reward(
                sampled_audio,
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "chord":
            # TODO: enable genre-specific chord LM
            return chord_reward(
                self.requires["chord"],
                self.requires["chord_lms"],
                sampled_audio,
                [["default"]] * len(sampled_audio),
                sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "loudness_sim":
            return loudness_reward(
                sampled_audio,
                batch["style_audio"],
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "audio_metrics":
            return audio_metrics_reward(
                sampled_audio,
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "semantic_diversity":
            return semantic_diversity_reward(
                batch["sampled_semantic_tokens"],
                device=sampled_audio.device,
            )
        elif rw_type == "chroma":
            return chroma_reward(
                sampled_audio,
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "anchor_points_sim":
            return anchor_points_sim_reward(    
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                sample_rate=sample_rate,
                device=sampled_audio.device,
                mulan_hpath=self.hparams.required_modules["mulan"]["hpath"],
            )
        elif rw_type == "mulan_temporal":
            mulan_temporal = mulan_temporal_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                device=sampled_audio.device,
                sample_rate=sample_rate,
            )
            return mulan_temporal
        elif rw_type == "chroma_temporal":
            chroma_temporal = chroma_temporal_reward(
                sampled_audio.squeeze(1),
                sample_rate=sample_rate,
                device=sampled_audio.device,
            )
            return chroma_temporal
        else:
            raise ValueError(f"Unknown reward type: {rw_type}")

    def rerank(self, sampled_audio, eos_index_list, batch, extra_params):
        original_audio = sampled_audio.clone()
        if extra_params.sample_rate != TARGET_SAMPLE_RATE and getattr(self, "resampler", False):
            sampled_audio = self.resampler(sampled_audio.to(self.device))
        if len(sampled_audio.shape) == 3: # convert stereo to mono for rewards
            sampled_audio = sampled_audio.mean(1, keepdims=False)

        rewards, rewards_breakdown = self.compute_rewards(
            sampled_audio, eos_index_list, batch, extra_params, TARGET_SAMPLE_RATE
        )
        beam = extra_params.beam_size
        assert len(sampled_audio) % beam == 0
        bsz = len(sampled_audio) // beam
        rewards = rewards.reshape(bsz, beam)
        indices = rewards.argsort(dim=-1, descending=True).cpu().flatten().tolist()
        reranked_sampled_audio = []
        reranked_eos_index_list = []
        reranked_rewards_breakdown = []
        reranked_semantic_tokens = []
        for i in range(len(indices)):
            idx = (i // beam) * beam + indices[i]
            reranked_sampled_audio.append(original_audio[idx])
            if len(eos_index_list) > 0:
                reranked_eos_index_list.append(eos_index_list[idx])
            reranked_rewards_breakdown.append(rewards_breakdown[idx])
            reranked_semantic_tokens.append(batch["sampled_semantic_tokens"][idx])
        reranked_sampled_audio = torch.stack(reranked_sampled_audio)
        reranked_eos_index_list = torch.hstack(reranked_eos_index_list)
        batch["sampled_semantic_tokens"] = torch.stack(reranked_semantic_tokens)
        return (
            reranked_sampled_audio,
            reranked_eos_index_list,
            reranked_rewards_breakdown,
        )


def init_reranker(hpath, local_rank, cache_dir, sample_rate=24000):
    reranker = Reranker(
        rewards=hpath["rewards"],
        required_modules=hpath["required_modules"],
        local_rank=local_rank,
        cache_dir=cache_dir,
        sample_rate=sample_rate,
    )
    return {"reranker": reranker}