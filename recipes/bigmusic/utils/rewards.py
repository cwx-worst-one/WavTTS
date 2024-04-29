import numpy as np
import math
import torch
import torch.nn.functional as F
from recipes.bigmusic.datasets.transforms.structure import (
    ChorusDetectionTransform,
    IntensityTransform,
)
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length
from recipes.bigmusic.utils.metrics_asr import (
    edit_distance,
    remove_punc_case,
)
from recipes.bigmusic.lightning.embedding_modules import get_bestrq_umm_tokens
from recipes.musiclm.inference.utils import dump_wav
from torchaudio.functional import loudness, resample
import librosa
from scipy.stats import entropy

import json
import euler
euler.install_thrift_import_hook()
import base_thrift
import sami_thrift

CLIENT = euler.Client(
    sami_thrift.SamiService,
    target="sd://lab.sami.gateway?cluster=release_thrift",
    timeout=1200,
)
ACCESS_KEY = "ATUBJrWuzl"


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
    sample_rate,
    device,
    sampled_embeds=None,    # (batch_size * beam, D)
    target_embeds=None,     # (batch_size, D)
    shift_seconds=5,
    min_audio_duration=10,
    max_audio_duration=None,
):
    min_audio_length = min_audio_duration * sample_rate
    max_audio_length = max_audio_duration * sample_rate if max_audio_duration is not None else None
    if sampled_embeds is None:
        if sampled_audio.shape[-1] < min_audio_length:
            sampled_audio = crop_pad_to_seq_length(sampled_audio, min_audio_length)
        elif max_audio_length and sampled_audio.shape[-1] > max_audio_length:
            sampled_audio = random_crop_pad_to_seq_length(sampled_audio, max_audio_length)
        sampled_embeds = mulan_infer_fn(
            model=mulan_model,
            music=sampled_audio.float(),
            device=device,
            shift_seconds=shift_seconds,
        )
    if target_embeds is None:
        if target_audio.shape[-1] < min_audio_length:
            target_audio = crop_pad_to_seq_length(target_audio, min_audio_length)
        elif max_audio_length and target_audio.shape[-1] > max_audio_length:
            target_audio = random_crop_pad_to_seq_length(target_audio, max_audio_length)
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
    # Return inverse WER as reward (higher is better)
    return 1 - wer


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
    return 1 - num_words


@torch.no_grad()
def loudness_reward(
    sampled_audio,  # (batch_size * beam, T)
    target_audio,   # (batch_size, T)
    sample_rate,
    device,
):
    if sampled_audio.dim() == 2:
        sampled_audio = sampled_audio.unsqueeze(1)
    if target_audio.dim() == 2:
        target_audio = target_audio.unsqueeze(1)
    batch_size, beam = _infer_batch_beam(sampled_audio, target_audio)
    sampled_loudness = loudness(sampled_audio.float().cpu(), sample_rate=sample_rate)
    target_loudness = loudness(target_audio.float().cpu(), sample_rate=sample_rate)
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
        chord_seq = [x[2] for x in chord_labels[i] if x[2] != "N"]
        # If too few chords, use lowest reward possible (0)
        if len(chord_seq) < 4 or len(set(chord_seq)) < 2:
            continue
        # Use average token probability as reward
        chord_seq_str = " ".join(chord_seq)
        scores = []
        for key in chord_lm_keys[i]:
            score = chord_lms[key].score(chord_seq_str, eos=False)
            scores.append(10.0 ** (score / len(chord_seq)))
        # Cap score at 0.25 to prevent very common chords from dominating
        # Apply a smoothing function to increase score gap between chords
        chord_rewards[i] = min(np.mean(scores) * 4, 1.0) ** 1.5
    return chord_rewards


@torch.no_grad()
def chord_prob_reward(
    chord_model,
    sampled_audio,
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

    chord_labels = []
    for single_sample in resampled_audio:   # split batch to run, otherwise out of memory...
        chord_label = chord_model.predict_step(
            batch=(single_sample[None, ...], None),
            batch_idx=0,
        )[0]
    chord_labels.append(chord_label)

    chord_prob_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    for i in range(len(chord_labels)):
        assert len(chord_labels[i][0]) == 4    # make sure there is prob as the 4th item
        probs = [x[-1] if x[2] != "N" else 0 for x in chord_labels[i]]
        weights = [x[1]-x[0] for x in chord_labels[i]]
        chord_prob_rewards[i] = sum(d * w for d, w in zip(probs, weights)) / sum(weights)   # weighted by the duration of each chord

    return chord_prob_rewards

STRUCTURE_TO_SCORE = {
    "verse": 0.5,
    "chorus": 0.5,
    "intro-verse": 0.75,
    "intro-chorus": 0.75,
    "verse-chorus": 1.0,
    "chorus-verse": 1.0,
}


@torch.no_grad()
def structure_reward(
    structure_model,
    sampled_audio,
    sample_rate,
    device,
):
    def dedup(lst):
        lst_dedup = []
        for x in lst:
            if len(lst_dedup) == 0 or lst_dedup[-1] != x:
                lst_dedup.append(x)
        return lst_dedup

    if sample_rate != structure_model._sampling_rate:
        resampled_audio = resample(
            sampled_audio,
            orig_freq=sample_rate,
            new_freq=structure_model._sampling_rate,
        )
    else:
        resampled_audio = sampled_audio
    all_structure_labels = structure_model.predict_step(
        batch=(resampled_audio,),
        batch_idx=0,
    )
    filtered_structure_labels = []
    for structure_labels in all_structure_labels:
        structure_labels = [
            x for x in structure_labels if x["funct_name"] in {"intro", "verse", "chorus"}
        ]
        structure_labels = dedup([x["funct_name"] for x in structure_labels])
        filtered_structure_labels.append(structure_labels)
    structure_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    for i, structure_labels in enumerate(filtered_structure_labels):
        structure_str = "-".join(structure_labels)
        if structure_str in STRUCTURE_TO_SCORE:
            structure_rewards[i] = STRUCTURE_TO_SCORE[structure_str]
    return structure_rewards


@torch.no_grad()
def chorus_sim_reward(
    sampled_audio,
    ref_choruses,
    sample_rate,
    device,
    debug=False,
):
    def _get_reward(hyps, refs):
        if refs is None:
            return 1.0
        if len(refs) == 0:
            return 1 - math.tanh(len(hyps))
        rewards = []
        matched_hyps = set()
        for i, (ref_name, ref_st, ref_en) in enumerate(refs):
            max_reward = 0.0
            matched_j = None
            for j, (hyp_name, hyp_st, hyp_en) in enumerate(hyps):
                if ref_name != hyp_name or j in matched_hyps:
                    continue
                overlap = max(0, min(hyp_en, ref_en) - max(hyp_st, ref_st))
                if overlap > 0:
                    distance = (abs(ref_st - hyp_st) + abs(ref_en - hyp_en)) / 2
                    # Normalize between [0, 1]
                    reward = 1 - math.tanh(distance / 10)
                    if reward > max_reward:
                        max_reward = reward
                        matched_j = j
            rewards.append(max_reward)
            if matched_j is not None:
                matched_hyps.add(matched_j)
        # Insertions will reduce reward
        denom = len(refs) + len(hyps) - len(matched_hyps)
        return np.sum(rewards) / denom

    _, beam = _infer_batch_beam(sampled_audio, ref_choruses)
    # TODO: make params configurable
    chorus_detection = ChorusDetectionTransform(sample_rate=sample_rate)
    hyp_choruses = [chorus_detection.find_chorus(audio) for audio in sampled_audio]

    chorus_sim_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    for i, hyps in enumerate(hyp_choruses):
        refs = ref_choruses[i // beam]
        if refs is not None:
            refs = [x for x in refs if x[0] == "chorus"]
        reward = _get_reward(hyps, refs)
        chorus_sim_rewards[i] = reward
        if debug:
            print(f"refs: {refs}, hyps: {hyps}, reward: {reward}")
    return chorus_sim_rewards


@torch.no_grad()
def chorus_presence_reward(sampled_audio, sample_rate, device):
    # TODO: make params configurable
    chorus_detection = ChorusDetectionTransform(sample_rate=sample_rate)
    hyp_choruses = [chorus_detection.find_chorus(audio) for audio in sampled_audio]
    chorus_presence_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    for i, hyps in enumerate(hyp_choruses):
        if len(hyps) > 0:
            chorus_presence_rewards[i] = 1
    return chorus_presence_rewards


def get_audio_metrics(audio_bytes):
    payload = json.dumps(
        {
            "extra": {
                "cutoff_freq": True,
                "phase_check": True,
                "rms_stats": True,
                "clipping": True,
                "loudness": True,
            }
        }
    )
    request = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key=ACCESS_KEY,
        method="AudioMetrics",
        payload=payload,
        data=audio_bytes,
        version="v4",
    )
    try:
        response = CLIENT.Invoke(request)
        metric = json.loads(response.payload)
    except Exception as ex:
        print(f"AudioMetrics exception: {ex}, payload: {payload}")
        metric = {}
    return metric


def get_audio_metrics_score(metrics):
    # Clipping
    def _clip_score(clip):
        if clip.get("rate", 0) >= 5e-5:
            return 0
        for ch in ["left", "right"]:
            if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
                return 0
        return 1
    # Loudness
    def _loudness_score(loudness):
        if (
            loudness.get("integrated_loudness", -7) > -5
            or loudness.get("max_mom_loud", -7) >= 0
            or loudness.get("max_short_term_loud", -7) >= 0
        ):
            return 0
        return 1
    # RMS stats
    def _rms_score(rms_stats):
        if rms_stats.get("peak", 0) > 3:
            return 0
        for ch in ["left", "right"]:
            if (
                rms_stats.get(f"{ch}_total", -10) > -5
                or rms_stats.get(f"{ch}_total", -10) < -40
                or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
            ):
                return 0
        return 1
    # Cutoff frequency
    def _cutoff_freq_score(cutoff_freq):
        for ch in ["left", "right"]:
            if (
                cutoff_freq.get(f"rel_{ch}", 48000) < 15000
                and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
                and cutoff_freq.get(f"band_std_{ch}", 10) < 5
            ):
                return 0
        return 1
    # Phase
    def _phase_score(phase):
        if (
            phase.get("has_phase_issue", False)
            or abs(phase.get("rms_downmix_diff", 0.1)) > 3
        ):
            return 0
        return 1
    score = 0.0
    score += _clip_score(metrics.get("clipping", {}))
    score += _loudness_score(metrics.get("loudness", {}))
    score += _rms_score(metrics.get("rms_stats", {}))
    score += _cutoff_freq_score(metrics.get("cutoff_frequency", {}))
    score += _phase_score(metrics.get("phase_check", {}))
    return score / 5


@torch.no_grad()
def audio_metrics_reward(sampled_audio, sample_rate, device):
    # TODO: support stereo input
    if sampled_audio.dim() == 3:
        sampled_audio = sampled_audio.squeeze(1)
    rewards = torch.zeros(sampled_audio.size(0)).to(device)
    sampled_audio = sampled_audio.float().cpu().numpy()
    for i in range(len(sampled_audio)):
        audio_bytes = dump_wav(sampled_audio[i], sr=sample_rate)
        metrics = get_audio_metrics(audio_bytes)
        if len(metrics) == 0:
            continue
        rewards[i] = get_audio_metrics_score(metrics)
    return rewards

@torch.no_grad()
def intensity_sim_reward(
    sampled_audio,
    target_intensity,
    sample_rate,
    device,
    target_audio=None,
    calculation_mode="mean",
    intensity_hz=1,
    resize_mode="resample",
    normalize=False,

):
    intensity_transform = IntensityTransform(
        sample_rate=sample_rate,
        calculation_mode=calculation_mode,
        intensity_hz=intensity_hz,
        normalize=normalize,
        remove_silence=True
    )
    if target_intensity is None and target_audio is not None:
        target_intensity = [intensity_transform.get_intensity(audio) for audio in target_audio]
    _, beam = _infer_batch_beam(sampled_audio, target_intensity)

    intensity_sim_rewards = torch.zeros(sampled_audio.size(0)).to(device)
    hyp_intensity = [intensity_transform.get_intensity(audio) for audio in sampled_audio]

    for i, hyp in enumerate(hyp_intensity):
        ref = target_intensity[i // beam]

        # ensure dimensions match.
        if ref.shape[-1] != hyp.shape[-1]:
            target_size = min(ref.shape[-1], hyp.shape[-1])
            if resize_mode == 'resample':
                ref = F.interpolate(ref.view(1,1,-1), size=target_size).view(-1)
                hyp = F.interpolate(hyp.view(1,1,-1), size=target_size).view(-1)
            elif resize_mode == 'crop':
                ref = ref[..., :target_size]
                hyp = hyp[..., :target_size]
        intensity_sim_rewards[i] = 1 - (ref - hyp).abs().mean()
    return intensity_sim_rewards


@torch.no_grad()
def semantic_diversity_reward(umm_tokens, device):
    semantic_diversity_rewards = torch.zeros(umm_tokens.size(0)).to(device)
    for i in range(len(umm_tokens)):
        semantic_diversity_rewards[i] = float(len(umm_tokens[i].unique())) / umm_tokens.shape[-1]
    return semantic_diversity_rewards


@torch.no_grad()
def semantic_diversity_sim_reward(hyp_umm_tokens, ref_umm_tokens, device):
    _, beam = _infer_batch_beam(hyp_umm_tokens, ref_umm_tokens)
    semantic_diversity_sim_rewards = torch.zeros(hyp_umm_tokens.size(0)).to(device)
    for i, hyp in enumerate(hyp_umm_tokens):
        ref = ref_umm_tokens[i // beam]
        hyp_diversity = float(len(hyp.unique())) / hyp.shape[-1]
        ref_diversity = float(len(ref.unique())) / ref.shape[-1]
        semantic_diversity_sim_rewards[i] = 1 - abs(hyp_diversity - ref_diversity)
    return semantic_diversity_sim_rewards


@torch.no_grad()
def _chroma_stats(audio, sample_rate):
    if len(audio.shape) == 2:
        audio = audio.squeeze(0)
    chroma = librosa.feature.chroma_stft(
        y=audio.float().cpu().numpy(),
        sr=sample_rate,
        hop_length=sample_rate // 4,    # 0.25s
    )
    melody = chroma.argmax(axis=0)
    probs = np.bincount(melody) / len(melody)
    return probs.max(), entropy(probs)


@torch.no_grad()
def chroma_reward(
    sampled_audio,
    sample_rate,
    device,
    prob_thresh=0.35,
    entropy_thresh=1.7,
):
    chroma_rewards = torch.zeros(len(sampled_audio)).to(device)
    for i in range(len(sampled_audio)):
        max_prob, prob_entropy = _chroma_stats(sampled_audio[i], sample_rate)
        if max_prob < prob_thresh:
            chroma_rewards[i] += 0.5
        if prob_entropy > entropy_thresh:
            chroma_rewards[i] += 0.5
    return chroma_rewards


@torch.no_grad()
def chroma_sim_reward(
    sampled_audio,
    target_audio,
    sample_rate,
    device,
):
    _, beam = _infer_batch_beam(sampled_audio, target_audio)
    ref_stats = [
        _chroma_stats(target_audio[i], sample_rate) for i in range(len(target_audio))
    ]
    hyp_stats = [
        _chroma_stats(sampled_audio[i], sample_rate) for i in range(len(sampled_audio))
    ]
    chroma_sim_rewards = torch.zeros(len(sampled_audio)).to(device)
    for i in range(len(sampled_audio)):
        hyp_max_prob, hyp_prob_entropy = hyp_stats[i]
        ref_max_prob, ref_prob_entropy = ref_stats[i // beam]
        # Don't penalize if generated music has more variety than reference
        if hyp_max_prob <= ref_max_prob:
            max_prob_diff = 0
        else:
            max_prob_diff = min(1.0, abs(ref_max_prob - hyp_max_prob) / ref_max_prob)
        if hyp_prob_entropy >= ref_prob_entropy:
            prob_entropy_diff = 0
        else:
            prob_entropy_diff = min(1.0, abs(ref_prob_entropy - hyp_prob_entropy) / ref_prob_entropy)
        chroma_sim_rewards[i] = 1 - 0.5 * (max_prob_diff + prob_entropy_diff)
    return chroma_sim_rewards

@torch.no_grad()
def anchor_points_sim_reward(
    mulan_infer_fn,
    mulan_model,
    sampled_audio,  # (batch_size * beam, T)
    sample_rate,
    device,
    shift_seconds=5,
    min_audio_duration=10,
    max_audio_duration=None,
):

    binary_center = np.load("/mnt/bn/audio-diffusion/peng/binary_center.npy")
    min_audio_length = min_audio_duration * sample_rate
    max_audio_length = max_audio_duration * sample_rate if max_audio_duration is not None else None
    if sampled_audio.shape[-1] < min_audio_length:
        sampled_audio = crop_pad_to_seq_length(sampled_audio, min_audio_length)
    elif max_audio_length and sampled_audio.shape[-1] > max_audio_length:
        sampled_audio = crop_pad_to_seq_length(sampled_audio, max_audio_length)
    sampled_embeds = mulan_infer_fn(
        model=mulan_model,
        music=sampled_audio.float(),
        device=device,
        shift_seconds=shift_seconds,
    )
    sampled_embeds = sampled_embeds.float().cpu()
    binary_center = torch.from_numpy(binary_center).float()
    binary_center = binary_center / binary_center.norm(dim=1, keepdim=True)
    similarity = torch.matmul(sampled_embeds, binary_center.T)
    anchor_rewards = similarity[:, 0] - similarity[:, 1]
    anchor_rewards = anchor_rewards.to(device)
    return anchor_rewards
