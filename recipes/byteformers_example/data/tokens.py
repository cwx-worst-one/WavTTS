import random

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, fp: str, seq_len: int):
        self.data = np.memmap(fp, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.total = len(self.data)

    @staticmethod
    def prepare(text: str, tokenizer, out_fp: str):
        ids = np.array(tokenizer.encode(text), dtype=np.uint16)
        ids.tofile(out_fp)

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        idx = random.randint(0, self.total - self.seq_len)
        x = torch.from_numpy(self.data[idx : idx + self.seq_len + 1].astype(np.int64))
        y = x[1:]
        x = x[:-1]
        return x, y
