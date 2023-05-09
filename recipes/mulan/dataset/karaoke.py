import random

import numpy as np
import pyloudnorm as pyln
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

KARAOKE_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/karaoke/karaoke_{i:04d}.tar"
    for i in range(485)
]


class KaraokeDataset(IterableDataset):
    def __init__(self, mixed_batch_size=72, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.meter = pyln.Meter(24000)
        self.mode = mode
        self.dataset = (
            wds.WebDataset(KARAOKE_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .batched(mixed_batch_size, collation_fn=self._collate_fn)
            .unlisted()
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        source = meta["data_source"]
        music_id = meta["music_id"]
        ins_name = str(music_id).split("_")[-1].lower()
        # remove all the number in name
        ins_name = "".join([i for i in ins_name if not i.isdigit()])
        # remove parentheses
        ins_name = ins_name.replace("(", "").replace(")", "")
        # remoe dash
        ins_name = ins_name.replace("-", "")
        # remove left right center stero mono
        ins_name = (
            ins_name.replace("left", "")
            .replace("right", "")
            .replace("center", "")
            .replace("stereo", "")
            .replace("mono", "")
        )
        # make all spaces to one space
        ins_name = " ".join(ins_name.split())

        # remove singer names in vocal
        if "lead vocal" in ins_name:
            ins_name = "lead vocal"
        if "vocals" in ins_name:
            # remove all char after vocals
            ins_name = ins_name.split("vocals")[0] + "vocals"

        data["text"] = ins_name
        data["data_source"] = source
        data["music_id"] = utils.fix_hash(music_id)
        return data

    def _collate_fn(self, batch):
        out_batch = {}
        original_batch_size = len(batch)
        # gather to list
        for b in batch:
            # remove silence audio
            db = self.meter.integrated_loudness(b["audio"].numpy().T)
            if db < -50 or np.isneginf(db):
                continue
            else:
                for k, v in b.items():
                    if k in ["audio", "text", "music_id"]:
                        out_batch[k] = out_batch.get(k, []) + [v]

        # Random mix
        # 1. gather the names and put audio indx to list
        # 2. random choose 1 to max(ins_name_dict) names to mix
        # 3. random selct audio from the list
        # 4. mix the audio
        data_dict = {}
        for i, name in enumerate(out_batch["text"]):
            # remove + in name
            name = name.split(" + ")
            if isinstance(name, list):
                for ins_name in name:
                    data_dict[ins_name] = data_dict.get(ins_name, []) + [i]
            else:
                data_dict[name] = data_dict.get(name, []) + [i]

        ins_names = list(data_dict.keys())

        mixed_batch = []

        for i in range(original_batch_size):
            picked_ins_names = random.sample(
                ins_names,
                random.randint(min(2, len(ins_names) - 1), min(6, len(ins_names))),
            )
            mixed_text = " ".join(picked_ins_names)
            sorted_text = " ".join(sorted(picked_ins_names))
            mixed_audio = torch.zeros_like(out_batch["audio"][0])
            for ins_name in picked_ins_names:
                mixed_audio += out_batch["audio"][random.choice(data_dict[ins_name])]

            # tokenize text
            tokenized_text = self.tokenizer(
                mixed_text.lower(),
                padding="max_length",
                truncation=True,
                max_length=200,
                return_tensors="pt",
            )
            input_ids = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]
            token_type_ids = tokenized_text["token_type_ids"]
            if self.mode == "train":
                # scheme 1:
                original_attention_masks = attention_mask
                mask_of_mask = torch.rand(original_attention_masks.shape)
                sampled_mask = (
                    mask_of_mask > 0.05
                ) * original_attention_masks  # TODO random knock out 5%
                attention_mask = sampled_mask

            mixed_batch.append(
                {
                    "audio": mixed_audio,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "music_id": utils.fix_hash(sorted_text),
                }
            )
        return mixed_batch

    def __iter__(self):
        return iter(self.dataset)
