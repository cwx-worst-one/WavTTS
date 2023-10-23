import numpy as np
import torch
import torch.nn.functional as F
from recipes.bigmusic.utils.metrics_asr import (
    edit_distance,
    remove_punc_case,
)
from torchaudio.functional import loudness, resample


def _infer_batch_beam(sampled, ref):
    assert len(sampled) % len(ref) == 0
    batch_size = len(ref)
    beam = len(sampled) // len(ref)
    return batch_size, beam


@torch.no_grad()
def mulan_audio_reward(
    mulan_infer_fn,
    mulan_model,
    sampled_audio,  # (batch_size * beam, T)
    target_audio,   # (batch_size, T)
    device,
    sampled_embeds=None,    # (batch_size * beam, D)
    target_embeds=None,     # (batch_size, D)
    shift_seconds=5,
):
    if sampled_embeds is None:
        sampled_embeds = mulan_infer_fn(
            model=mulan_model,
            music=sampled_audio.float(),
            device=device,
            shift_seconds=shift_seconds,
        )
    if target_embeds is None:
        target_embeds = mulan_infer_fn(
            model=mulan_model,
            music=target_audio.float(),
            device=device,
            shift_seconds=shift_seconds,
        )
    batch_size, beam = _infer_batch_beam(sampled_embeds, target_embeds)
    # (batch_size, D) --> (batch_size * beam, D)
    target_embeds_reshaped = target_embeds.repeat(1, beam).reshape(batch_size * beam, -1)
    sim = F.cosine_similarity(sampled_embeds, target_embeds_reshaped)
    return sim, sampled_embeds, target_embeds


@torch.no_grad()
def mulan_text_reward(
    mulan_infer_fn,
    mulan_model,
    sampled_audio,  # (batch_size * beam, T)
    target_text,   # (batch_size,)
    device,
    sampled_embeds=None,    # (batch_size * beam, D)
    target_embeds=None,     # (batch_size, D)
    shift_seconds=5,
):
    if sampled_embeds is None:
        sampled_embeds = mulan_infer_fn(
            model=mulan_model,
            music=sampled_audio.float(),
            device=device,
            shift_seconds=shift_seconds,
        )
    if target_embeds is None:
        target_embeds = mulan_infer_fn(
            model=mulan_model,
            text=target_text,
            device=device,
        )
    batch_size, beam = _infer_batch_beam(sampled_embeds, target_embeds)
    # (batch_size, D) --> (batch_size * beam, D)
    target_embeds_reshaped = target_embeds.repeat(1, beam).reshape(batch_size * beam, -1)
    sim = F.cosine_similarity(sampled_embeds, target_embeds_reshaped)
    return sim, sampled_embeds, target_embeds


@torch.no_grad()
def wer_reward(
    sampled_lyrics, # (batch_size * beam,)
    ref_lyrics, # (batch_size,)
    device,
):
    batch_size, beam = _infer_batch_beam(sampled_lyrics, ref_lyrics)
    wer = torch.zeros(batch_size * beam).to(device)
    for i in range(batch_size):
        ref = remove_punc_case(ref_lyrics[i])
        for j in range(beam):
            idx = i * beam + j
            hyp = remove_punc_case(sampled_lyrics[idx])
            if ref != "" and hyp != "":
                # Cap WER at 100% for more stable range
                wer[idx] = min(1.0, edit_distance(ref, hyp).edits() / len(ref))
            elif (ref == "" and hyp != "") or (ref != "" and hyp == ""):
                # Default to 100% WER
                wer[idx] = 1.0
    # Return negative WER as reward (higher is better)
    return -1 * wer


@torch.no_grad()
def nonvocal_reward(sampled_lyrics, device):
    num_words = torch.zeros(len(sampled_lyrics)).to(device)
    for i in range(len(num_words)):
        hyp = remove_punc_case(sampled_lyrics[i])
        num_words[i] = len(hyp.split())
    # Use tanh to keep the range between (0, 1) and make the penalty
    # increase exponentially with the number of words. We divide the
    # number of words by 4 to make the range less extreme.
    num_words = torch.tanh(num_words / 4)
    # Return negative number of words, we want to maximize this
    return -1 * num_words


@torch.no_grad()
def loudness_reward(
    sampled_audio,  # (batch_size * beam, T)
    target_audio,   # (batch_size, T)
    sample_rate,
    device,
):
    batch_size, beam = _infer_batch_beam(sampled_audio, target_audio)
    sampled_loudness = loudness(sampled_audio.cpu(), sample_rate=sample_rate)
    target_loudness = loudness(target_audio.cpu(), sample_rate=sample_rate)
    # (batch_size,) --> (batch_size * beam,)
    target_loudness = target_loudness.reshape(batch_size, 1).repeat(1, beam).reshape(batch_size * beam)
    # loudness is in LKFS (dB scale), so we use sigmoid to measure the difference
    # With sigmoid, a difference of 1dB is considered relatively small, while a
    # difference of 3dB is considered large.
    loudness_diff = torch.sigmoid(target_loudness - sampled_loudness)
    # We normalize the range to (0, 1) to be consistent with other rewards
    loudness_diff = (2 * loudness_diff - 1).abs().to(device)
    # Sometimes loudness function returns NaN, in which case we just assign
    # a hardcoded reward of 1dB loudness difference.
    loudness_diff[torch.isnan(loudness_diff)] = 0.4621
    # 1 - loudness_diff is loudness similarity, we want to maximize this
    return 1 - loudness_diff


@torch.no_grad()
def chord_reward(
    chord_model,
    chord_lms,
    sampled_audio,
    chord_lm_keys,
    sample_rate,
    device,
):
    if sample_rate != chord_model._sample_rate:
        resampled_audio = resample(
            sampled_audio,
            orig_freq=sample_rate,
            new_freq=chord_model._sample_rate,
        )
    else:
        resampled_audio = sampled_audio
    chord_labels = chord_model.predict_step(
        batch=(resampled_audio, None),
        batch_idx=0,
    )
    chord_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    for i in range(len(chord_labels)):
        if len(chord_lm_keys[i]) == 0:
            continue
        chord_seq = [x[-1] for x in chord_labels[i] if x[-1] != "N"]
        # If too few chords, use lowest reward possible (0)
        if len(chord_seq) < 4 or len(set(chord_seq)) < 2:
            continue
        # Use average token probability as reward
        chord_seq_str = " ".join(chord_seq)
        scores = []
        for key in chord_lm_keys[i]:
            score = chord_lms[key].score(chord_seq_str, eos=False)
            scores.append(10.0 ** (score / len(chord_seq)))
        chord_rewards[i] = np.mean(scores)
    return chord_rewards