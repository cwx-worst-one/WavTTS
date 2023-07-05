from torch.utils.data import Dataset
import torchaudio
from glob import glob


class PlasticDataset(Dataset):
    def __init__(self):
        self.items = [
            torchaudio.load(wav_path)[0] for wav_path in glob("/mnt/bn/audio-diffusion/data/plastic_love/*.wav")
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
