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
            df = pd.read_csv(prompt_path)
            for _, row in df.iterrows():
                items.append(
                    {
                        "category": row["category"],
                        "text": row["text"]
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
