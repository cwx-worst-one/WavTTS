''' flops hook. '''
from core.extensions import FlopsProfiler
from core.runner.metric import SpeedMeter

from .hook import HOOKS, Hook


@HOOKS.register_module()
class FlopsHook(Hook):
    '''FlopsHook.'''

    def __init__(self, *_args, **kwargs):
        '''flops hook'''
        super().__init__()
        self.output_freq = kwargs.get('flop_output_freq', 5000)
        self.record_iters = kwargs.get('flop_record_iters', 100)
        self.record_offset = 37
        self.is_profile = False
        assert self.output_freq >= self.record_iters

    def before_run(self, runner):
        self.prof = FlopsProfiler(runner.solution)
        runner.train_log_buffer.add_meter('flops', SpeedMeter)
        runner.train_log_buffer.get_meter('flops').set_enable_state(self.is_profile)

    def after_run(self, runner):
        if self.is_profile:
            self.prof.end_profile()
            self.is_profile = False

    def before_train_iter(self, runner):
        if runner.iter % self.output_freq == self.record_offset:
            self.prof.start_profile()
            self.is_profile = True
            runner.train_log_buffer.get_meter('flops').set_enable_state(self.is_profile)
        if self.is_profile:
            self.prof.reset_profile()

    def after_train_iter(self, runner):
        if self.is_profile:
            flops = self.prof.get_total_flops() / 1e12
            runner.train_log_buffer.update({'flops': flops})
        if (
            runner.iter % self.output_freq == self.record_offset + self.record_iters
            and self.is_profile
        ):
            self.prof.end_profile()
            self.is_profile = False
            runner.train_log_buffer.get_meter('flops').set_enable_state(self.is_profile)
