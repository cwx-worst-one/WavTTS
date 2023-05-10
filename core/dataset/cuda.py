'''
Cuda Transforms.
'''
import torch
from core.utils import get_local_rank


def to_cuda(data):
    '''
    to cuda ops.
    mv all torch.tensor to gpu, by recursive function:
    Args:
        data(any): origin data.
    Return:
        any: data that all tensor on gpu.
    '''
    if isinstance(data, torch.Tensor):
        data = data.cuda(get_local_rank(), non_blocking=True)
        return data
    if isinstance(data, (list, tuple)):
        return [to_cuda(item) for item in data]
    if isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    return data


def pin_memory(data):
    '''
    put the fetched data Tensors in pinned memory
    enable faster data transfer to CUDA-enabled GPUs
    '''
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    if isinstance(data, (list, tuple)):
        return [pin_memory(item) for item in data]
    if isinstance(data, dict):
        return {k: pin_memory(v) for k, v in data.items()}
    return data
