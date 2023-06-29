import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

URL2IDX_Map = {
    "vocal": {
        "mcc600k": "/mnt/bn/audio-diffusion/data/mcc_pgc_600k/gpt_url2idx.txt",
        "metacritic": "/mnt/bn/audio-diffusion/data/metacritic/vocal_url2idx.txt",
        "everynoise": "/mnt/bn/audio-diffusion/data/everynoise/vocal_url2idx.txt",
    },
    "nonvocal": {
        "metacritic": "/mnt/bn/audio-diffusion/data/metacritic/non_vocal_url2idx.txt",
        "everynoise": "/mnt/bn/audio-diffusion/data/everynoise/non_vocal_url2idx.txt",
    }
}

class MMEDataset(IterableDataset):
    def __init__(self, name="mcc600k", audio_type="vocal", mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.name = name
        self.audio_type = audio_type
        url2idx = URL2IDX_Map[audio_type][name]
        self.dataset = (
            IndexedWebDataset(url2idx, **kwargs)
            .decode()
            .map(process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        if self.name == "mcc600k":
            meta = data['__index_data__']
            data["text"] = meta["gpt_text"]
            data["music_id"] = int(data["__key__"])
            return data
        elif self.name == "metacritic":
            meta = data['__index_data__']
            critic_content = json.loads(meta['critic_content'])
            if len(critic_content) > 0:
                random.shuffle(critic_content)
                data["text"] = meta["desc"] + " " + critic_content[0]
            else:
                data["text"] = meta["desc"]
            data["music_id"] = int(data["__key__"])
            return data
        elif self.name == "everynoise":
            meta = data['__index_data__']
            data["text"] = meta["genre"]
            data["music_id"] = int(data["__key__"])
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    for audio_type in ["vocal", "nonvocal"]:
        for name in ["mcc600k", "metacritic", "everynoise"]:
            if name == "mcc600k" and audio_type == "nonvocal":
                continue
            print(f"Testing {audio_type} {name}")
            d = MMEDataset(name=name, audio_type=audio_type, mode="train")
            for item in d:
                print(item.keys())
                break
