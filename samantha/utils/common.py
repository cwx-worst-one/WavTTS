import random
import subprocess

import numpy as np
import torch
from lightning_fabric.utilities.seed import seed_everything


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
