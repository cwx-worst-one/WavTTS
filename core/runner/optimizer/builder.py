''' optimizer builder. '''
# pylint: disable=missing-function-docstring,no-name-in-module

import copy
import inspect
import torch
from core.utils import Registry, build_from_cfg


OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_torch_optimizers():
    '''register torch optimizer.'''
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


def register_apex_optimizers():
    '''register apex optimizer.'''
    apex_optimizers = []
    try:
        # pylint: disable=import-outside-toplevel
        from apex import optimizers
    except Exception:
        return apex_optimizers
    for module_name in dir(optimizers):
        if module_name.startswith('__'):
            continue
        _optim = getattr(optimizers, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            apex_optimizers.append(module_name)
    return apex_optimizers


def register_other_optimizers():
    pass


TORCH_OPTIMIZERS = register_torch_optimizers()
APEX_OPTIMIZERS = register_apex_optimizers()
register_other_optimizers()


def build_optimizer_constructor(cfg):
    '''build opt constructor.'''
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    '''build opt.'''
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
    optim_constructor = build_optimizer_constructor(
        dict(type=constructor_type, optimizer_cfg=optimizer_cfg)
    )
    optimizer = optim_constructor(model)
    return optimizer
