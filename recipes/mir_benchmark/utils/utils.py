import torch
from typing import List


def split_by_node(src):
    for s in src:
        yield s


def get_tags(probs: torch.Tensor, tags: List[str], thresholds: torch.Tensor):
    """
    probs: (batch, n_classes)
    thresholds: (n_classes,)
    """
    mask = (probs >= thresholds).cpu()
    return [
        [tags[j] for j in range(len(tags)) if mask[i][j]] for i in range(probs.size(0))
    ]
