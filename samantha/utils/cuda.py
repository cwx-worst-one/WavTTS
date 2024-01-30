from contextlib import contextmanager

import torch
import torch.backends.cuda
import torch.backends.cudnn


def get_compute_capability():
    cc = torch.cuda.get_device_properties(0)
    return cc.major + cc.minor * 0.1


@contextmanager
def torch_allow_tf32(enable_matmul=None, enable_cudnn=None):
    allow_matmul = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn = torch.backends.cudnn.allow_tf32

    if enable_matmul is not None:
        torch.backends.cuda.matmul.allow_tf32 = enable_matmul
    if enable_cudnn is not None:
        torch.backends.cudnn.allow_tf32 = enable_cudnn

    yield

    torch.backends.cuda.matmul.allow_tf32 = allow_matmul
    torch.backends.cudnn.allow_tf32 = allow_cudnn
