''' Hook for Arnold profiler 2.0. '''

import ctypes
import torch
from packaging import version
from core.utils import logging
from .hook import HOOKS, Hook


@HOOKS.register_module()
class ProfilerHook(Hook):
    '''
    Arnold profiler 2.0 Hook.
    Usage:
        set profiler in config file or cmd line.
        ```
        profiler=dict(
            use_arnold_profiler_profiler=True,
            start_step=0,
            end_step=10,
        )
        ```
    '''

    def __init__(self, **kwargs):
        '''init.'''
        self._cudart = ctypes.CDLL("libcudart.so")
        self.start_step = kwargs.get('profile_start_step', 2)
        self.end_step = kwargs.get('profile_end_step', 10)
        self.model_depth = kwargs.get('profile_model_depth', 4)
        logging.info(
            "Start CudaProfiler start step %d end step %d model depth %d",
            self.start_step,
            self.end_step,
            self.model_depth,
        )

    def before_run(self, runner):
        register_nvtx_hook(
            runner,
            runner.solution,
            'Solution',
            self.start_step,
            self.end_step,
            self.model_depth,
        )

    def before_iter(self, runner):
        '''
        Do profiler check_and_profiler.
        '''
        if runner.iter == self.start_step:
            self._cudart.cudaProfilerStart()
            # pylint: disable=protected-access
            if version.parse(torch.__version__) < version.parse('1.8.0'):
                torch.autograd._enable_profiler(
                    torch.autograd.ProfilerConfig(
                        torch.autograd.ProfilerState.NVTX,  # other is CPU.CUDA
                        False,  # record input shape
                    )
                )
            else:
                torch.autograd._enable_profiler_legacy(
                    torch.autograd.ProfilerConfig(
                        torch.autograd.ProfilerState.NVTX,  # other is CPU.CUDA
                        False,  # record input shape
                        False,  # profile_memory
                        False,  # with_stack
                        False,  # with_flops
                    )
                )

        if runner.iter > self.end_step or runner.iter < self.start_step:
            return
        torch.cuda.nvtx.range_push('iteration {}'.format(runner.iter + 1))

    def after_iter(self, runner):
        torch.cuda.nvtx.range_pop()
        if runner.iter == self.end_step:
            # pylint: disable=protected-access
            if version.parse(torch.__version__) < version.parse('1.8.0'):
                torch.autograd._disable_profiler()
            else:
                torch.autograd._disable_profiler_legacy()
            self._cudart.cudaProfilerStop()
        if runner.iter > self.end_step or runner.iter < self.start_step:
            return


def register_nvtx_hook(runner, model, name, start_step, end_step, max_depth, cur_depth=1):
    '''
    register nvtx hook recursively.
    '''
    if cur_depth > max_depth:
        return

    hook = NVTXHook(runner, name, start_step, end_step, cur_depth)
    model.register_forward_pre_hook(hook.pre_fwd)
    model.register_forward_hook(hook.post_fwd)

    for child_name, module in model.named_children():
        register_nvtx_hook(
            runner, module, child_name, start_step, end_step, max_depth, cur_depth + 1
        )


class NVTXHook:
    '''model hook for nvtx.'''

    def __init__(self, runner, name, start_step, end_step, depth):
        '''init.'''
        self.runner = runner
        self.name = name
        self.start_step = start_step
        self.end_step = end_step
        self.depth = depth

    def pre_fwd(self, _module, _inputs):
        '''pre forward.'''
        if self.runner.iter > self.end_step or self.runner.iter < self.start_step:
            return
        text = '{}, depth={}'.format(self.name, self.depth)
        torch.cuda.nvtx.range_push(text)

    def post_fwd(self, _module, _inputs, _outputs):
        '''post forward.'''
        if self.runner.iter > self.end_step or self.runner.iter < self.start_step:
            return
        torch.cuda.nvtx.range_pop()
