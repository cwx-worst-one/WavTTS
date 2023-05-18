from torch.utils.data import Dataset
import pandas as pd


class InferenceDataset(Dataset):
    def __init__(self, prompt_path):
        self.items = []
        df = pd.read_csv(prompt_path)
        for _, row in df.iterrows():
            self.items.append(
                {
                    "category": row["category"],
                    "text": row["text"]
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
