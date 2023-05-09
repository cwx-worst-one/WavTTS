import torch
from torchvision.transforms import Normalize as NormalizeBase
from torchvision.transforms.functional import invert

from .inverse import InverseTransform


class Invert(InverseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return invert(x)

    def inverse(self, x):
        return invert(x)


class Normalize(InverseTransform):
    def __init__(self, mean, std, inplace: bool = False):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.fn = NormalizeBase(self.mean.tolist(), self.std.tolist(), inplace=inplace)
        self.inverse_fn = NormalizeBase(
            (-self.mean / self.std).tolist(), (1.0 / self.std).tolist(), inplace=inplace
        )

    def inverse(self, x):
        return self.inverse_fn(x)

    def forward(self, x):
        return self.fn(x)


class ImagePostProcessTransform(InverseTransform):
    def __init__(
        self,
        image_x: int,
        image_y: int,
        n_channels: int,
        power: float = 1.0,
        max_norm_value: float = 30e6,
    ):
        super().__init__()
        self.image_x = image_x
        self.image_y = image_y
        self.n_channels = n_channels
        self.power = power
        self.max_norm_value = max_norm_value

    def forward(self, image):
        # normalize
        image = image / (image.max() + 1e-8)
        image = image.pow(self.power)

        if image.ndim == 3:
            image = image[:, None]
        image = image.expand([-1, self.n_channels, -1, -1])
        return image

    def inverse(self, image):
        # NOTE: the max_value is -very- necessary for the InvMelScaler to work!!!
        image = image.mean(dim=1)
        image = image.pow(1 / self.power)
        return image * self.max_norm_value
