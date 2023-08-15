from torch.utils.data import Dataset


class MetaBasedDataset(Dataset):
    def __init__(self, meta_lst):
        self.meta = self._parse_meta(meta_lst)

    def _parse_meta(self, meta_lst):
        meta = []
        with open(meta_lst, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split("|")
                meta.append(line)
        return meta
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        return self.meta[index]