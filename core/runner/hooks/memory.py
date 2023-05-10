''' memory hook. '''
import os
import torch

from core.utils import logging
from .hook import HOOKS, Hook


@HOOKS.register_module()
class EmptyCacheHook(Hook):
    '''EmptyCacheHook.'''

    def __init__(self, before_epoch=False, after_epoch=True, after_iter=False):
        '''init.'''
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, runner):
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, runner):
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, runner):
        if self._after_epoch:
            torch.cuda.empty_cache()

    def after_train_iter(self, runner):
        if runner.iter % 100 == 0:
            torch.cuda.empty_cache()


@HOOKS.register_module()
class MemMonitorHook(Hook):
    '''MemMonitorHook.'''

    def __init__(self, interval):
        '''init.'''
        self.interval = interval

    def before_iter(self, runner):
        if runner.iter % self.interval != 1:
            return
        try:
            cmd = 'ps xh -o rsz'
            used_mem = 0
            ret = os.popen(cmd).read().split('\n')
            for r in ret:
                if r:
                    used_mem += int(r)
            used_mem /= 1024**2
            total_mem = int(os.getenv('MY_MEM_LIMIT')) / 1024**3
            free_mem = total_mem - used_mem
            if used_mem / total_mem > 0.95:
                logging.warning(
                    'Memory high usage warning in iter[%d], total = %.2fGB'
                    ' used = %.2fGB free = %.2fGB percent = %.2f%%',
                    runner.iter,
                    total_mem,
                    used_mem,
                    free_mem,
                    used_mem / total_mem * 100,
                )
        except Exception:
            return
