from torch.utils.data import Dataset
import pandas as pd
import csv
import os


class InferenceDataset(Dataset):
    def __init__(self, items):
        self.items = items

    @classmethod
    def from_prompt_path(cls, prompt_path, skip_header=True):
        items = []
        _, ext = os.path.splitext(prompt_path)
        skipped_header = False
        if ext == ".csv":
            with open(prompt_path, "r") as f:
                for ary in csv.reader(f):
                    if skip_header and not skipped_header:
                        skipped_header = True
                        continue
                    assert len(ary) >= 2, f"Invalid csv line: {ary}"
                    items.append({"category": ary[0], "text": ary[1]})
        else:
            with open(prompt_path, "r") as fp:
                for line in fp.readlines():
                    items.append(
                        {
                            "category": "demos",
                            "text": line.strip()
                        }
                    )
        return InferenceDataset(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
