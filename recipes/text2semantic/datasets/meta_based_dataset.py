import os
from torch.utils.data import Dataset


class MetaBasedDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                uttid, prompt_text, prompt_wav_path, text = line.strip().split("|")
                if not os.path.isabs(prompt_wav_path):
                    prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                assert os.path.exists(prompt_wav_path)
                meta.append([uttid, prompt_text, prompt_wav_path, text])
        return meta
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        return self.meta[index]