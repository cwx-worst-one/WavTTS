import torch
from contextlib import contextmanager
from copy import deepcopy

import torch.nn as nn


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

@contextmanager
def evaluate_model(model):
    _training = model.training
    try:
        model.eval()
        yield model
    finally:
        if _training:
            model.train()

def checkpoint(module, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)