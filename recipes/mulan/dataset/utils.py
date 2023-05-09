import hashlib
import random

import numpy as np
import torch

HDFS_BASE = "/home/byte_speech_sv/jingsong.gao/"


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8


def process_audio(data):
    audio = data["audio.npy"]
    audio = (audio / 32768.0).astype("float32")

    music_len = 24000 * 10
    if audio.shape[-1] < music_len:
        audio = np.pad(audio, ((0, 0), (0, music_len - audio.shape[-1])), "constant")
        start_idx = 0
    else:
        start_idx = random.randint(0, audio.shape[-1] - music_len)

    audio = torch.from_numpy(audio[..., start_idx : start_idx + music_len]).float()
    data["audio"] = audio

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
            max_length=200,
            return_tensors="pt",
        )
        data["input_ids"] = encodings["input_ids"]
        data["token_type_ids"] = encodings["token_type_ids"]
        data["attention_mask"] = encodings["attention_mask"]

        # Randomly knock out tokens for training
        if mode == "train":
            # scheme 1:
            original_attention_masks = data["attention_mask"]
            mask_of_mask = torch.rand(original_attention_masks.shape)
            sampled_mask = (
                mask_of_mask > 0.05
            ) * original_attention_masks  # TODO random knock out 5%
            data["attention_mask"] = sampled_mask

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


def select_datasets(datasets, selected_names):
    return [datasets[name] for name in selected_names]
