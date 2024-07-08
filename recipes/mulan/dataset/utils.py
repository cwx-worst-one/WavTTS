import hashlib
import random
import re

import numpy as np
import torch

HDFS_BASE = "/home/byte_speech_sv/jingsong.gao/"


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8

def process_audio(data, vocal_segments=None):
    audio = data["audio.npy"]
    if audio.dtype == np.int16:
        audio = (audio / 32768.0).astype("float32")
    if len(audio.shape) == 1:
        audio = audio[None, :]

    music_len = 24000 * 10
    if vocal_segments:
        # avoid vocal segments
        # [{"start": 6610, "end": 8320}, {"start": 14960, "end": 16540}] in ms
        # 1. get nonvocal segments
        audio_length = audio.shape[-1]
        non_vocal_segments = []
        for i, segment in enumerate(vocal_segments):
            segment["start"] = segment["start"] // 1000 * 24000
            segment["end"] = segment["end"] // 1000 * 24000
            if i == 0 and segment["start"] > 0:
                non_vocal_segments.append({"start": 0, "end": segment["start"]})
            elif i > 0:
                prev_segment = vocal_segments[i - 1]
                non_vocal_segments.append({"start": prev_segment["end"], "end": segment["start"]})

            if i == len(vocal_segments) - 1 and segment["end"] < audio_length:
                non_vocal_segments.append({"start": segment["end"], "end": audio_length})
        valid_segments = [segment for segment in non_vocal_segments if segment["end"] - segment["start"] >= music_len]
        # 2. random select one nonvocal segment and crop at random position
        if not valid_segments:
            return None
        segment = random.choice(valid_segments)
        start_idx = random.randint(segment["start"], segment["end"] - music_len)
    else:
        if audio.shape[-1] < music_len:
            audio = np.pad(audio, ((0, 0), (0, music_len - audio.shape[-1])), "constant")
            start_idx = 0
        elif audio.shape[-1] > music_len * 4: # trim intro / outro
            start_idx = random.randint(music_len, audio.shape[-1] - music_len * 2)
        else:
            start_idx = random.randint(0, audio.shape[-1] - music_len)

    audio = torch.from_numpy(audio[..., start_idx : start_idx + music_len]).float()
    data["audio"] = audio
    #print("data_audio", data["audio"])
    # delete full audio from the sample, otherwise 
    # it will take a lot of memory in sample buffer
    del data["audio.npy"]

    return data



def tokenize_text(tokenizer, mode="train", seq_len=150):
    def _tokenize_text(data):
        text = data["text"]

        # Skip empty text
        if text == "":
            return None

        # Tokenize the text
        encodings = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        data["input_ids"] = encodings["input_ids"]
        if "token_type_ids" in encodings:
            # T5 does not have token_type_ids
            data["token_type_ids"] = encodings["token_type_ids"]
        data["attention_mask"] = encodings["attention_mask"]

        # Randomly knock out tokens for training
        # if mode == "train":
        #     # scheme 1:
        #     original_attention_masks = data["attention_mask"]
        #     mask_of_mask = torch.rand(original_attention_masks.shape)
        #     sampled_mask = (
        #         mask_of_mask > 0.05
        #     ) * original_attention_masks  # TODO random knock out 5%
        #     data["attention_mask"] = sampled_mask

        return data

    return _tokenize_text


def collate_fn(batch):
    keys = ["audio", "input_ids", "attention_mask", "token_type_ids", "music_id"]
    out_batch = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            if k in b:
                out_batch[k].append(b[k])

    for k, v in out_batch.items():
        if k in ["music_id"]:
            out_batch[k] = torch.tensor(v)
        elif k in b:
            out_batch[k] = torch.cat(v)

    return out_batch


# def collate_ymv_fn(batch):
#     keys = ["audio", "input_ids", "attention_mask", "token_type_ids"]
#     out_batch = {k: [] for k in keys}
#     for b in batch:
#         for k in keys:
#             out_batch[k].append(b[k])

#     for k, v in out_batch.items():
#         out_batch[k] = torch.cat(v)

#     return out_batch



def select_datasets(datasets, selected_names):
    return [datasets[name] for name in selected_names]


def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)
