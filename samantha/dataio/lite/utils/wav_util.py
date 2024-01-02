import io
import logging
import pickle

import librosa
import numpy as np
import torch

from .mel import mel_spectrogram
from .phone_to_id import PhoneToId, get_duration_frames_wds

logger = logging.getLogger(__name__)


def trim_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset.

    origin_wav_len = len(wav)
    wav = np.pad(wav, (5400, 5400))

    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    # num_sil_samples = int(8 * 300)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - 3200, 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + 5400, len(wav))

    trimmed = wav[start_idx:stop_idx]

    head_trim_nums = 5400 - start_idx
    tail_trim_nums = stop_idx - 5400 - origin_wav_len

    return trimmed, head_trim_nums, tail_trim_nums


def sequence_mask(seq_lens, max_len=None, device="cpu"):
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    return mask.float()


def get_duration_wds(alignments, phonemes, mel_len, hop_ms):
    if alignments is not None and len(alignments) > 0:
        duration_frames = get_duration_frames_wds(alignments, phonemes, hop_ms)
        if duration_frames is not None and abs(sum(duration_frames) - mel_len) > 2:
            duration_frames = None
        if duration_frames is not None:
            pad_len = sum(duration_frames) - mel_len
            duration_frames[-1] = duration_frames[-1] - pad_len
            if duration_frames[-1] < 0:
                duration_frames = None
    else:
        duration_frames = None

    return duration_frames


def get_text_duration(
    sample, meta_obj, acoustic_len, hop_ms, mask_use_alignment, use_text
):
    # load phone seq & duration
    text = sample["text"]
    lab = meta_obj.get("labels", "")
    utt_id = sample["__key__"]
    data_dict = {}
    labels = lab  # .decode()
    if labels is None:
        return None
    labels = list(filter(lambda x: x != "", labels.split("\n")))
    if len(labels) < 2:
        return None
    if len(labels[-1].split("\t")) == 2:
        last_duration = labels[-1].split("\t")[-1]
        last_line = labels[-2]
        labels = labels[:-2]
        last_line = "\t".join(last_line.split("\t")[:-1] + [last_duration])
        labels.append(last_line)
    try:
        text_id_phones_tones = PhoneToId().convert_tacolab_to_text_id(labels)
    except Exception as e:
        logger.warning(f"{utt_id} get_text_duration exception: {e}")
        return None
    if text_id_phones_tones is None:
        logger.warning(f"{utt_id} get_text_duration failed ...")
        return None
    else:
        text_id, phones, tones, word_segs, alignments = text_id_phones_tones

    if mask_use_alignment:
        duration = get_duration_wds(alignments, phones, acoustic_len, hop_ms)
        if duration is None:
            return None
        data_dict["duration_ori"] = duration  # for masking methods
        data_dict["duration"] = duration

    if use_text:
        text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)
        data_dict["phone"] = text_id[0, :]
        data_dict["tone"] = text_id[1, :]
        data_dict["word_seg"] = text_id[2, :]

    data_dict["utt_id"] = utt_id
    data_dict["lab"] = labels
    data_dict["text"] = text
    return data_dict


def get_bn(sample):
    bn = pickle.loads(sample["bns"])
    bn = torch.from_numpy(bn)
    bn_stats = bn
    out_dim = bn.shape[1] // 2
    m, logs = torch.split(bn, out_dim, dim=-1)
    bn = m + torch.randn_like(m) * torch.exp(logs)
    return bn, bn_stats


def get_wav(sample, max_wav_len, audio_resampler, wav_amp_aug):
    # sourcery skip: raise-specific-error
    wav, _ = librosa.load(io.BytesIO(sample["wav"]), sr=None)
    if len(wav.shape) >= 2:
        wav = wav[0]
    if wav.shape[0] > max_wav_len or wav.shape[0] < 12000:
        return None

    wav = wav.astype(np.float32)
    wav = torch.FloatTensor(wav)
    if audio_resampler is not None:
        wav = audio_resampler(wav)
    scale = max(0.001, torch.max(torch.abs(wav)))
    wav = wav / scale * 0.95
    if wav_amp_aug is not None and wav_amp_aug["use"]:
        wav_aug_scale = (
            np.random.rand() * (wav_amp_aug["max"] - wav_amp_aug["min"])
            + wav_amp_aug["min"]
        )
        wav = wav * wav_aug_scale
    return wav


def get_mel(wav, wav_divide, mel_config, umm_hop_size, mel_norm_mean, mel_norm_std):
    # handle umm & mel length
    crop_wav_len = wav.shape[0] % wav_divide
    max_wav_len = wav.shape[0] - crop_wav_len
    # if crop_wav_len > 0:
    #    wav = torch.nn.functional.pad(
    #       wav, (0, self.wav_divide - crop_wav_len), "constant", 0)
    # max_wav_len = wav.shape[0]
    max_mel_len = max_wav_len // mel_config["hop_size"]
    max_umm_len = max_wav_len // umm_hop_size
    mel = mel_spectrogram(wav.unsqueeze(0), **mel_config).squeeze(0)
    mel = (mel - mel_norm_mean) / mel_norm_std
    return mel, max_mel_len, max_umm_len
