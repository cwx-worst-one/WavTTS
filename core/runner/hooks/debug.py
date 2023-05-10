'''Hook for debug'''
from core.utils import get_rank, logging
from .hook import HOOKS, Hook


@HOOKS.register_module()
class DebugHook(Hook):
    '''
    Debug Hook.
    Usage:
        set in config file and
        execute debug.sh
        ```
        train=dict(
        dolphin_debug_depth=6,
        )
        ```
    '''

    def __init__(self, max_depth):
        '''
        max_depth: the hook record forward, pre forward and backward
        of max depth
        '''
        self.max_depth = max_depth
        logging.info("Debug start, model max_depth %d", self.max_depth)

    def before_run(self, runner):
        register_model_recoder_hook(runner.solution, 'Solution', self.max_depth)


def register_model_recoder_hook(model, name, max_depth, cur_depth=1):
    '''
    register model record hook recursively.
    '''
    if cur_depth > max_depth:
        return
    hook = ModelRecorderHook(name)
    model.register_forward_pre_hook(hook.pre_fwd)
    model.register_forward_hook(hook.post_fwd)
    model.register_backward_hook(hook.post_bwd)

    for child_name, module in model.named_children():
        register_model_recoder_hook(
            module, "{}.{}".format(name, child_name), max_depth, cur_depth + 1
        )


class ModelRecorderHook:
    '''model record hook for record forward and pre_forward'''

    def __init__(self, name):
        '''init.'''
        self.name = name
        self.rank = get_rank()

    def pre_fwd(self, _module, _inputs):
        '''pre forward.'''
        logging.all_rank_info("rank %d: %s pre forward ", self.rank, self.name)

    def post_fwd(self, _module, _inputs, _output):
        '''post forward.'''
        logging.all_rank_info("rank %d: %s forward ", self.rank, self.name)

    def post_bwd(self, _module, _inputs, _output):
        '''post backward.'''
        logging.all_rank_info("rank %d: %s backward ", self.rank, self.name)
