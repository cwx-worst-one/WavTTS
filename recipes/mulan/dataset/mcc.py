import pickle
import random

import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

MCC_GPT_GEN_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_gpt_generated/mcc_gpt_generated_{i:04d}.tar"  # noqa
    for i in range(201)
] + [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_gpt_generated/mcc_gpt_generated_{i:02d}.tar"  # noqa
    for i in range(12)
]
AED_GPT_GEN_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/aed700k_gpt_generated/aed700k_gpt_generated_{i:04d}.tar"  # noqa
    for i in range(274)
]
MCC_N2M_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_n2m/mcc_n2m_{i:04d}.tar"
    for i in range(6379)
]


class MCCGPTGenDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_GPT_GEN_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        data["text"] = meta["gpt_generated_caption"]
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)


class AEDGPTGenDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(AED_GPT_GEN_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_aed.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]

        text = ""
        if random.random() < 0.5:
            text = meta["gpt_generated_caption"]
        else:
            if str(meta["music_id"]) in self.chatgpt_g4:
                tags = self.chatgpt_g4[str(meta["music_id"])]
                # get all the items into list
                tags = [v for v in tags.values()]
                # randomize the order
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag
        data["text"] = text

        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)


class MCCN2MDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_N2M_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_mcc.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]
        # n2m is a list of 3 matched texts
        # data["text"] = random.choice(n2m_texts)
        text = ""
        if random.random() < 0.5:
            n2m_texts = meta["n2m"]
            text += random.choice(n2m_texts)
        else:
            if str(meta["music_id"]) in self.chatgpt_g4:
                tags = self.chatgpt_g4[str(meta["music_id"])]
                # get all the items into list
                tags = [v for v in tags.values()]
                # randomize the order
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag
        data["text"] = text
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]

        return data

    def __iter__(self):
        return iter(self.dataset)
