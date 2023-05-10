''' for Debug Nan. '''

import math
import torch
from core.utils import get_rank, logging
from .hook import HOOKS, Hook


@HOOKS.register_module()
class NanDebugHook(Hook):
    '''
    register debug nan hook recursively.
    '''

    def before_run(self, runner):
        invoke_debug_nan_hook(runner.solution, 'solution')

    def after_iter(self, runner):
        for name, p in runner.solution.named_parameters():
            max_val = recursive_abs_max(p.grad)
            if not math.isfinite(max_val):
                logging.error(
                    'rank %d: parameter %s\' grad not finite : %r', get_rank(), name, max_val
                )


def invoke_debug_nan_hook(model, name, cur_depth=1):
    '''invoke model hook.'''
    hook = ValueCheckHook(name)
    model.register_forward_hook(hook.post_fwd)
    model.register_backward_hook(hook.post_bwd)

    for child_name, module in model.named_children():
        invoke_debug_nan_hook(module, '{}.{}'.format(name, child_name), cur_depth + 1)


class ValueCheckHook:
    '''model hook for nvtx.'''

    _list = []

    def __init__(self, name):
        '''init.'''
        self.name = name

    def post_fwd(self, _module, _inputs, outputs):
        '''post forward.'''
        max_val = recursive_abs_max(outputs)
        if not math.isfinite(max_val):
            logging.error('rank %d: %s fwd out not finite : %r', get_rank(), self.name, max_val)

    def post_bwd(self, _module, _grad_input, grad_output):
        '''post backward.'''
        max_val = recursive_abs_max(grad_output)
        if not math.isfinite(max_val):
            logging.error(
                'rank %d: %s bwd activation grad not finite : %r', get_rank(), self.name, max_val
            )


def recursive_abs_max(inputs):
    '''recursive abs max.'''
    if isinstance(inputs, torch.Tensor):
        return inputs.abs().max().item()
    if isinstance(inputs, (list, tuple)):
        return max(recursive_abs_max(v) for v in inputs)
    if isinstance(inputs, dict):
        return max(recursive_abs_max(v) for v in inputs.values())
    return 0  # skip None and unknown type
