from torch.utils.data import Dataset


class ValDataset(Dataset):
    def __init__(self, param1, param2):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
