from copy import deepcopy

import torch.nn as nn


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
