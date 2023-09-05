import hashlib
import random

import numpy as np
import torch


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8


def process_audio(data):
    audio = data["audio.npy"]
    if audio.dtype == np.int16:
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


def pad_seq_embeds(
    text_field="text_embeds.npy",
    text_output_field="text_embeds",
    seq_len=250,
):
    def _pad_seq_embeds(data):
        text_embeds = data[text_field]
        assert len(text_embeds.shape) == 2
        text_embeds = torch.from_numpy(text_embeds).float()
        # pad to seq_len
        if text_embeds.shape[0] < seq_len:
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            attention_mask[text_embeds.shape[0] :] = 0
            text_embeds = torch.cat(
                [
                    text_embeds,
                    torch.zeros(
                        seq_len - text_embeds.shape[0], text_embeds.shape[-1]
                    ).float(),
                ],
                dim=0,
            )
        else:
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            text_embeds = text_embeds[:seq_len, :]
        data[text_output_field] = text_embeds
        data[f"{text_output_field}_mask"] = attention_mask
        del data[text_field]
        return data
    return _pad_seq_embeds
