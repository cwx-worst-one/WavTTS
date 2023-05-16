''' checkpoint. '''

import os.path as osp

import torch
from core.extensions import mpu
from .logging import warning
from .path import mkdir_or_exist


def _merge(states):
    '''merge multi dict.'''
    model_state = dict()
    for k in states[0].keys():
        vs = [state[k] for state in states if k in state]
        if not isinstance(vs[0], torch.Tensor):
            v = _merge(vs)
        else:
            v = sum(vs)
            if len(vs) > 1:
                mode = 'floor' if v.is_floating_point() else 'trunc'
                v = v.div(len(vs), rounding_mode=mode)
        model_state[k] = v
    return model_state


def load_checkpoint(model, filenames, map_location=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filenames (str or list(str)): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict: The loaded checkpoint.
    """
    if not isinstance(filenames, list):
        filenames = [filenames]
    if not filenames:
        return None

    map_location = 'cpu'
    # load all checkpoints in filenames list
    model_states = []
    for filename in filenames:
        checkpoint = torch.load(filename, map_location)
        model_states.append(checkpoint['model'])
    model_state = _merge(model_states)
    checkpoint['model'] = None

    # load model_state
    model.cpu()
    msg = model.load_state_dict(model_state, strict=False)
    model.cuda()
    err_msg = []
    if msg.unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(msg.unexpected_keys)}\n')
    if msg.missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(msg.missing_keys)}\n')
    if err_msg:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        warning('\n'.join(err_msg))

    return checkpoint


def weights_to_cpu(data):
    """Copy a model state_dict to cpu.

    Args:
        state_dict(dict): Model weights on GPU.

    Returns:
        dict: Model weights on CPU.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu()
        return data
    if isinstance(data, (list, tuple)):
        return [weights_to_cpu(item) for item in data]
    if isinstance(data, dict):
        return {k: weights_to_cpu(v) for k, v in data.items()}
    return data


def weights_to_cuda(data):
    """Copy a model state_dict to cuda.

    Args:
        state_dict(dict): Model weights on CPU.

    Returns:
        dict: Model weights on CUDA.
    """
    if isinstance(data, torch.Tensor):
        data = data.cuda()
        return data
    if isinstance(data, (list, tuple)):
        return [weights_to_cuda(item) for item in data]
    if isinstance(data, dict):
        return {k: weights_to_cuda(v) for k, v in data.items()}
    return data


def save_checkpoint(filename, state_dict):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        filename (str): Checkpoint filename.
        state_dict (any): The state dictionary to save.
    """

    mkdir_or_exist(osp.dirname(filename))
    # immediately flush buffer
    with open(filename, 'wb') as f:
        torch.save(state_dict, f)
        f.flush()
