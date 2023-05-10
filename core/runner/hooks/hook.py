''' hook. '''

from core.utils import Registry

HOOKS = Registry('hook')


class Hook:
    '''Hook class.'''

    def before_run(self, runner):
        '''before run.'''

    def after_run(self, runner):
        '''after run.'''

    def before_epoch(self, runner):
        '''before epoch.'''

    def after_epoch(self, runner):
        '''after_epoch.'''

    def before_iter(self, runner):
        '''before_iter.'''

    def after_iter(self, runner):
        '''after_iter.'''

    def before_train_epoch(self, runner):
        '''before_train_epoch.'''
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        '''before_val_epoch.'''
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        '''after_train_epoch.'''
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        '''after_val_epoch.'''
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        '''before_train_iter.'''
        self.before_iter(runner)

    def before_val_iter(self, runner):
        '''before_val_iter.'''
        self.before_iter(runner)

    def after_train_iter(self, runner):
        '''after_train_iter.'''
        self.after_iter(runner)

    def after_val_iter(self, runner):
        '''after_val_iter.'''
        self.after_iter(runner)

    # pylint: disable=no-self-use
    def every_n_epochs(self, runner, n):
        '''every_n_epochs.'''
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        '''every_n_inner_iters.'''
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        '''every_n_iters.'''
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        '''end_of_epoch.'''
        return runner.inner_iter + 1 == len(runner.data_loader)
