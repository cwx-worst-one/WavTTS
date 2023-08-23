import torch


def get_compute_capability():
    cc = torch.cuda.get_device_properties(0)
    return cc.major + cc.minor * 0.1
