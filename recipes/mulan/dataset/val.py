import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
        max_text_len=250,
        tok_path=None,
    ):

        # df_playlist
        # columns: music_id, text
        # each row is a text and its corresponding music_id
        self.df = pd.read_csv(f"{path}/{csv_name}")

        # set the max text length
        self.max_text_len = max_text_len
        self.text_key = text_key

        self.music_feature_path = f"{path}/{music_feature_path}"
        self.music_feature_type = music_feature_type
        
        if tok_path is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            tokenizer.pad_token = tokenizer.eos_token

        self.encodings_text = tokenizer(
            self.df[self.text_key].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
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

        item = {
            "audio": torch.tensor((audio / 32768.0).astype("float32")).squeeze(0),
        }

        item["input_ids"] = torch.tensor(self.encodings_text["input_ids"][index])
        if "token_type_ids" in self.encodings_text:
            item["token_type_ids"] = torch.tensor(self.encodings_text["token_type_ids"][index])
        item["attention_mask"] = torch.tensor(self.encodings_text["attention_mask"][index])

        return item

    def __len__(self):
        return len(self.df)

def qq_collate_fn(batch):
  return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'token_type_ids': torch.stack([x['token_type_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'audio': torch.stack([x['audio'] for x in batch])
}

class MultiLanTextMusicDataset(Dataset):
    def __init__(
        self,
        path,
        csv_name,  # meta csv name in the tar.gz
        text_key="text_chinese",
        audio_key="music_id",
        music_feature_path="audio",
        music_feature_type="npy",
        max_text_len=200,
        tokenizer_name="bert-base-chinese",
    ):

        # df_playlist
        # columns: music_id, text
        # each row is a text and its corresponding music_id
        self.df = pd.read_csv(f"{path}/{csv_name}")

        # set the max text length
        self.max_text_len = max_text_len
        self.text_key = text_key
        self.audio_key = audio_key

        self.music_feature_path = f"{path}/{music_feature_path}"
        self.music_feature_type = music_feature_type

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings_text = tokenizer(
            self.df[self.text_key].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )

    def __getitem__(self, index):
        # each data instance is a query text
        # index the row
        row = self.df.iloc[index]

        music_id = row[self.audio_key]

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

if __name__=="__main__":
    ds = MultiLanTextMusicDataset(
        path="data/qq",
        csv_name="qq_meta_val.csv",
        text_key="desc",
        audio_key="song_id",
        tokenizer_name="bert-base-chinese" # bert-base-multilingual-cased
    )
    dataloader = DataLoader(ds, batch_size=200, num_workers=1, collate_fn=qq_collate_fn)
    import pdb
    for i, batch in enumerate(dataloader):
        print(i, batch["input_ids"].shape)
