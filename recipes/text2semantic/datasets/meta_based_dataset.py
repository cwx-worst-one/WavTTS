import os
from torch.utils.data import Dataset


class MetaBasedDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        if meta_lst is None or meta_lst == '':
            return meta

        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                if len(line.strip().split("|")) == 5:
                    uttid, prompt_text, prompt_wav_path, text, spk = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    assert os.path.exists(prompt_wav_path)
                    meta.append([uttid, prompt_text, prompt_wav_path, text, spk])
                elif len(line.strip().split("|")) == 4:
                    uttid, prompt_text, prompt_wav_path, text = line.strip().split("|")
                    if not os.path.isabs(prompt_wav_path):
                        prompt_wav_path = os.path.join(os.path.dirname(meta_lst), prompt_wav_path)
                    assert os.path.exists(prompt_wav_path)
                    meta.append([uttid, prompt_text, prompt_wav_path, text])
                elif len(line.strip().split("|")) == 2:
                    uttid, text = line.strip().split("|")
                    meta.append([uttid, text])
        return meta
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        return self.meta[index]