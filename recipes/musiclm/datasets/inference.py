from torch.utils.data import Dataset
import pandas as pd
import os


class InferenceDataset(Dataset):
    def __init__(self, items):
        self.items = items

    @classmethod
    def from_prompt_path(cls, prompt_path):
        items = []
        _, ext = os.path.splitext(prompt_path)
        if ext == ".csv":
            with open(prompt_path, "r") as f:
                skipped_header = False
                for line in f:
                    # Skip first line (header)
                    if not skipped_header:
                        skipped_header = True
                        continue
                    ary = line.strip().split(",")
                    if len(ary) < 2:
                        print(f"Skipping invalid csv line: {line.strip()}")
                        continue
                    items.append(
                        {
                            "category": ary[0],
                            "text": ",".join(ary[1:]),
                        }
                    )
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
