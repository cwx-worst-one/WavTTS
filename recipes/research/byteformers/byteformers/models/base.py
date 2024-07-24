from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from byteformers.models.utils import evaluate_model


class BaseModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    def to_torchscript(self):
        with evaluate_model(self):
            self = self.apply(
                lambda m: remove_parametrizations(m, "weight") if is_parametrized(m) else m
            )
            return torch.jit.script(self.cpu())
