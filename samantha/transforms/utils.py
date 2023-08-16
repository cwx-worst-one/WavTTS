import inspect
from hashlib import md5

import torch.nn as nn


def get_transform_version(transform: nn.Module):
    transform_src = inspect.getsource(type(transform))
    return md5(transform_src.encode("utf-8")).hexdigest()
