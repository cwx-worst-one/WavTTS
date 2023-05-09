import torch.nn as nn
from torchaudio_augmentations.compose import Compose


class InverseCompose(Compose):
    def inverse(self, x):
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x


class InverseTransform(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def inverse(self, x):
        pass
