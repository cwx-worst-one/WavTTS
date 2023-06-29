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


def filter_g4_category_labels(s):
    if ":" not in s:
        return s
    x = s.index(":")
    prefix = s[:x].lower()
    for k in ["genre", "mood", "theme", "epoch", "musician", "level", "experience"]:
        if k in prefix:
            return s[x + 1 :].strip()
    return s


class MCCGPTGenDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_GPT_GEN_URLS, **kwargs)
            .decode()
            .to_tuple("__key__", "meta.json", "audio.npy")
            .map(lambda x: {"chunk_id": x[0], "meta.json": x[1], "audio.npy": x[2]})
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
    def __init__(self, mode="train", verbose=False, **kwargs):
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(AED_GPT_GEN_URLS, **kwargs)
            .decode()
            .to_tuple("__key__", "meta.json", "audio.npy")
            .map(lambda x: {"chunk_id": x[0], "meta.json": x[1], "audio.npy": x[2]})
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_aed.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)
        with open("assets/n2m_aed.pkl", "rb") as f:
            self.n2m = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]

        text = ""
        if random.random() < 0.5:
            if data["chunk_id"] in self.n2m:
                tags = list(self.n2m[data["chunk_id"]])
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag + " "
                text = text.strip()
                if self.verbose:
                    chunk_id = data["chunk_id"]
                    print(
                        f"AED N2M sample - chunk_id = {chunk_id}, n2m candidates is {tags}, final text is {text}"
                    )
            else:
                if self.verbose:
                    print(f"chunk_id not found in this AED example!")

            if text == "":
                text = meta["gpt_generated_caption"]
        else:
            if str(meta["music_id"]) in self.chatgpt_g4:
                tags = self.chatgpt_g4[str(meta["music_id"])]
                # get all the items into list
                tags = [
                    filter_g4_category_labels(tags[k])
                    for k in tags
                    if "instrument" not in k.lower() and "place" not in k.lower()
                ]
                tags = [s for s in tags if len(s) > 0]
                # randomize the order
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag + " "
                if self.verbose:
                    chunk_id = data["chunk_id"]
                    print(
                        f"AED G4 sample - chunk_id = {chunk_id}, G4 candidates is {tags}, final text is {text}"
                    )
        data["text"] = text.strip()
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)


class MCCN2MDataset(IterableDataset):
    def __init__(self, mode="train", verbose=False, **kwargs):
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_N2M_URLS, **kwargs)
            .decode()
            .to_tuple("__key__", "meta.json", "audio.npy")
            .map(lambda x: {"chunk_id": x[0], "meta.json": x[1], "audio.npy": x[2]})
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_mcc.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)
        with open("assets/n2m_mcc.pkl", "rb") as f:
            self.n2m = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]
        # n2m is a list of 3 matched texts
        # data["text"] = random.choice(n2m_texts)
        text = ""
        if random.random() < 0.5:
            if data["chunk_id"] in self.n2m:
                tags = list(self.n2m[data["chunk_id"]])
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag + " "
                text = text.strip()
                if self.verbose:
                    chunk_id = data["chunk_id"]
                    print(
                        f"MCC N2M sample - chunk_id = {chunk_id}, N2M candidates is {tags}, final text is {text}"
                    )
        else:
            if str(meta["music_id"]) in self.chatgpt_g4:
                tags = self.chatgpt_g4[str(meta["music_id"])]
                # get all the items into list
                tags = [
                    filter_g4_category_labels(tags[k])
                    for k in tags
                    if "instrument" not in k.lower() and "place" not in k.lower()
                ]
                tags = [s for s in tags if len(s) > 0]
                # randomize the order
                random.shuffle(tags)
                for tag in tags:
                    if random.random() < 0.8:
                        text += tag + " "
                if self.verbose:
                    chunk_id = data["chunk_id"]
                    print(
                        f"MCC G4 sample - chunk_id = {chunk_id}, G4 candidates is {tags}, final text is {text}"
                    )
        data["text"] = text.strip()
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]

        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":

    g4_dataset = MCCGPTGenDataset()
    g4_loader = wds.WebLoader(g4_dataset, num_workers=1, batch_size=None)

    for batch in g4_loader:
        print(batch)
        break

    n2m_dataset = MCCN2MDataset()
    n2m_loader = wds.WebLoader(n2m_dataset, num_workers=1, batch_size=None)
    for batch in n2m_loader:
        print(batch)
        break
