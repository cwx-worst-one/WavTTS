import torch

from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
from recipes.bigmusic.utils.rewards import (
    mulan_text_reward,
    wer_reward,
    chord_reward,
    nonvocal_reward,
    structure_reward,
    chorus_sim_reward,
)


class Reranker:
    def __init__(
        self,
        rewards,
        required_modules,
        local_rank,
        cache_dir,
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
        if "style_text" in rewards or "qualitative" in rewards:
            assert "mulan" in self.requires
            assert "mulan_infer_fn" in self.requires
        if "structure" in rewards:
            assert "structure" in self.requires
        if "chord" in rewards:
            assert "chord" in self.requires
            assert "chord_lms" in self.requires

    def compute_rewards(self, sampled_audio, eos_index_list, batch, extra_params):
        rewards = torch.zeros(len(sampled_audio)).to(sampled_audio.device)
        rewards_breakdown = [{} for _ in range(len(sampled_audio))]
        for rw_type, rw_weight in self.rewards.items():
            if rw_weight == 0:
                continue
            rw = self._get_reward(rw_type, sampled_audio, eos_index_list, batch, extra_params)
            rewards += rw_weight * rw
            for i in range(len(sampled_audio)):
                rewards_breakdown[i][rw_type] = rw[i].item()
        return rewards, rewards_breakdown

    def _get_reward(self, rw_type, sampled_audio, eos_index_list, batch, extra_params):
        if rw_type == "style_text":
            return mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                batch["style_text"],
                device=sampled_audio.device,
            )[0]
        elif rw_type == "qualitative":
            # TODO: make phrase configurable
            positive_reward = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                ["cd quality, catchy, memorable"],
                device=sampled_audio.device,
            )[0]
            negative_reward = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio,
                ["noisy, boring, forgettable"],
                device=sampled_audio.device,
            )[0]
            return positive_reward - negative_reward
        elif rw_type == "wer":
            lyrics_hyp = asr_transcribe_lyrics(
                self.requires,
                sampled_audio,
                sample_lengths=None if len(eos_index_list) == 0 else eos_index_list,
                sample_rate=extra_params.sample_rate
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
                sample_rate=extra_params.sample_rate
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
                sample_rate=extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "chorus_sim":
            return chorus_sim_reward(
                sampled_audio,
                [[x for x in y if x[0] == "chorus"] for y in batch["structure"]],
                sample_rate=extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif rw_type == "chord":
            # TODO: enable genre-specific chord LM
            return chord_reward(
                self.requires["chord"],
                self.requires["chord_lms"],
                sampled_audio,
                [["default"]] * len(sampled_audio),
                extra_params.sample_rate,
                device=sampled_audio.device,
            )
        else:
            raise ValueError(f"Unknown reward type: {rw_type}")

    def rerank(self, sampled_audio, eos_index_list, batch, extra_params):
        rewards, rewards_breakdown = self.compute_rewards(
            sampled_audio, eos_index_list, batch, extra_params
        )
        beam = extra_params.beam_size
        assert len(sampled_audio) % beam == 0
        bsz = len(sampled_audio) // beam
        rewards = rewards.reshape(bsz, beam)
        indices = rewards.argsort(dim=-1, descending=True).cpu().flatten().tolist()
        reranked_sampled_audio = []
        reranked_eos_index_list = []
        reranked_rewards_breakdown = []
        for i in range(len(indices)):
            idx = (i // beam) * beam + indices[i]
            reranked_sampled_audio.append(sampled_audio[idx])
            if len(eos_index_list) > 0:
                reranked_eos_index_list.append(eos_index_list[idx])
            reranked_rewards_breakdown.append(rewards_breakdown[idx])
        reranked_sampled_audio = torch.vstack(reranked_sampled_audio)
        reranked_eos_index_list = torch.hstack(reranked_eos_index_list)
        return (
            reranked_sampled_audio,
            reranked_eos_index_list,
            reranked_rewards_breakdown,
        )


def init_reranker(hpath, local_rank, cache_dir):
    reranker = Reranker(
        rewards=hpath["rewards"],
        required_modules=hpath["required_modules"],
        local_rank=local_rank,
        cache_dir=cache_dir,
    )
    return {"reranker": reranker}