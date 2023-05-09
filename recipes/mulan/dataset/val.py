import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

pd.set_option("mode.chained_assignment", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextMusicDataset(Dataset):
    def __init__(
        self,
        path,
        csv_name,  # meta csv name in the tar.gz
        text_key="text",
        music_feature_path="audio",
        music_feature_type="npy",
        max_text_len=200,
        max_music_len=1024,
    ):

        # df_playlist
        # columns: music_id, text
        # each row is a text and its corresponding music_id
        self.df = pd.read_csv(f"{path}/{csv_name}")

        # set the max text length
        self.max_text_len = max_text_len
        self.max_music_len = max_music_len
        self.text_key = text_key

        self.music_feature_path = f"{path}/{music_feature_path}"
        self.music_feature_type = music_feature_type

        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.encodings_text = tokenizer(
            self.df[self.text_key].tolist(), truncation=True, padding=True
        )

    def __getitem__(self, index):
        # each data instance is a query text
        # index the row
        row = self.df.iloc[index]

        music_id = row["music_id"]

        if self.music_feature_type == "npz":
            raw_music_file = np.load(f"{self.music_feature_path}/{music_id}.npz")
            audio = raw_music_file["audio"]
        else:
            audio = np.load(f"{self.music_feature_path}/{music_id}.npy")

        input_ids = torch.tensor(self.encodings_text["input_ids"][index])
        token_type_ids = torch.tensor(self.encodings_text["token_type_ids"][index])
        attention_mask = torch.tensor(self.encodings_text["attention_mask"][index])

        item = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "audio": torch.tensor((audio / 32768.0).astype("float32")).squeeze(0),
        }
        return item

    def __len__(self):
        return len(self.df)
