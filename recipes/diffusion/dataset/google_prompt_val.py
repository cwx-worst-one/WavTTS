import glob
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

ASSET_PATH = 'recipes/diffusion/assets/google_prompts/*/*.npy'

class GooglePromptVal(Dataset):
    def __init__(self, num_samples=80):
        path_list = glob.glob(ASSET_PATH)[:num_samples]

        token_list = []
        for path in path_list:
            tokens = np.load(path)
            token_list.append(torch.from_numpy(tokens))
        
        self.name_list = [str(Path(path).stem).split('.')[0].replace('-', ' ') for path in path_list]
        self.token_list = token_list

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):

        return self.token_list[idx], self.name_list[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = GooglePromptVal()

    token, name = dataset.__getitem__(0)

    loader = DataLoader(dataset, batch_size=16, num_workers=1)

    for i, d in enumerate(loader):
        token, name = d
        print(token.shape)
        print(type(name))
        print(list(name))
        assert 1==2
        # print(len(token), name)
    
    # print(token.shape, name)
