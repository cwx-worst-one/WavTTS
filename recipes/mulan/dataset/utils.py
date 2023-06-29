import hashlib
import random
import re

import numpy as np
import torch

HDFS_BASE = "/home/byte_speech_sv/jingsong.gao/"


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8


def process_audio(data):
    audio = data["audio.npy"]
    audio = (audio / 32768.0).astype("float32")
    if len(audio.shape) == 1:
        audio = audio[None, :]

    music_len = 24000 * 10
    if audio.shape[-1] < music_len:
        audio = np.pad(audio, ((0, 0), (0, music_len - audio.shape[-1])), "constant")
        start_idx = 0
    else:
        start_idx = random.randint(0, audio.shape[-1] - music_len)

    audio = torch.from_numpy(audio[..., start_idx : start_idx + music_len]).float()
    data["audio"] = audio

    # delete full audio from the sample, otherwise 
    # it will take a lot of memory in sample buffer
    del data["audio.npy"]

    return data


def tokenize_text(tokenizer, mode="train"):
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
            max_length=400,
            return_tensors="pt",
        )
        data["input_ids"] = encodings["input_ids"]
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
            out_batch[k].append(b[k])

    for k, v in out_batch.items():
        if k in ["music_id"]:
            out_batch[k] = torch.tensor(v)
        else:
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
