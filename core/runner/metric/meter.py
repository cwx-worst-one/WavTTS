''' Metric class. '''


class AbstractMeter:
    '''Abstract  Meter.'''

    eps = 1e-5

    def __init__(self):
        '''init.'''
        self._value = 0.0
        self._count = 0
        self._sum = 0.0

    @property
    def value(self):
        '''value.'''
        return self._value

    @property
    def count(self):
        '''count sum'''
        return self._count

    @property
    def sum(self):
        '''sum.'''
        return self._sum

    def reset(self):
        '''reset.'''
        self._value = 0
        self._count = 0
        self._sum = 0

    def add(self, val, count=1):
        '''add.'''
        raise NotImplementedError

    def add_val(self, val):
        '''add val.'''
        raise NotImplementedError

    def add_count(self, count):
        '''add count.'''
        raise NotImplementedError

    def get(self):
        '''average.'''
        return self.do_get(self._sum, self._count, self.eps)

    @staticmethod
    def do_get(_sum, _count, _eps=1e-5):
        '''average.'''
        raise NotImplementedError


class BaseMeter(AbstractMeter):
    '''Base Meter.'''

    need_reduce_sum = True
    need_reduce_count = True

    def add(self, val, count=1):
        '''add.'''
        self._count += count
        self._sum += val
        self._value = val

    def add_val(self, val):
        '''add val.'''
        self._sum += val
        self._value = val

    def add_count(self, count):
        '''add count.'''
        self._count += count

    @staticmethod
    def do_get(sum_val, count_val, eps=1e-5):
        '''average.'''
        return sum_val / (count_val + eps)


class WeightedMeter(AbstractMeter):
    '''Weighted Meter'''

    need_reduce_sum = True
    need_reduce_count = True

    def add(self, val, count=1):
        '''add'''
        self._count += count
        self._sum += val * count

    def add_val(self, val):
        '''add val.'''
        raise RuntimeError('add_count is not supported')

    def add_count(self, count):
        '''add count.'''
        raise RuntimeError('add_count is not supported')

    @staticmethod
    def do_get(sum_val, count_val, eps=1e-5):
        '''average'''
        return sum_val / (count_val + eps)


class ExponentialWeightedMeter(AbstractMeter):
    '''Exponential Weighted Meter'''

    need_reduce_sum = False
    need_reduce_count = False

    def __init__(self, value=300.0, momentum=0.9):  # pylint: disable=super-init-not-called
        self._value = value
        self._count = 0.0
        self._sum = 0.0
        self._momentum = momentum

    def add(self, val, _count=1):
        '''add'''
        self._value = self._momentum * self._value + (1 - self._momentum) * val
        self._sum = self._value

    def add_val(self, val):
        '''add val.'''
        raise RuntimeError('add_val is not supported')

    def add_count(self, count):
        '''add count.'''
        raise RuntimeError('add_count is not supported')

    @staticmethod
    def do_get(sum_val, _count_val, _eps=1e-5):
        '''average'''
        return sum_val


class SpeedMeter(AbstractMeter):
    '''Speed Meter'''

    need_reduce_sum = True
    need_reduce_count = True
    # fps and tps count have been reduced , so they need to be multiply by world_size

    def __init__(self, is_enable=True):
        super().__init__()
        self.is_enable = is_enable

    def set_enable_state(self, state):
        '''set enable state'''
        self.is_enable = state

    def add(self, val, count=1):
        '''add'''
        if not self.is_enable:
            return
        self._count += count
        self._sum += val
        self._value = val

    def add_val(self, val):
        '''add val.'''
        if not self.is_enable:
            return
        self._sum += val
        self._value = val

    def add_count(self, count):
        '''add count.'''
        if not self.is_enable:
            return
        self._count += count

    @staticmethod
    def do_get(sum_val, count_val, eps=1e-5):
        '''average'''
        return sum_val / (count_val + eps)


# TODO(cch) maxmeter and realmeter support reduce
class MaxMeter(AbstractMeter):
    '''Max Meter'''

    need_reduce_sum = False
    need_reduce_count = False

    def add(self, val, **_kwargs):
        '''add'''
        self._value = val
        self._sum = max(self._sum, val)

    def add_val(self, val):
        '''add val.'''
        self._value = val
        self._sum = max(self._sum, val)

    def add_count(self, count):
        '''add count.'''

    @staticmethod
    def do_get(sum_val, _count_val, _eps=1e-5):
        '''average'''
        return sum_val


class SumMeter(AbstractMeter):
    '''Max Meter'''

    need_reduce_sum = True
    need_reduce_count = False

    def add(self, val, _count=1):
        '''add'''
        self._value = val
        self._sum += val

    def add_val(self, val):
        '''add val.'''
        self._value = val
        self._sum += val

    def add_count(self, count):
        '''add count.'''

    @staticmethod
    def do_get(sum_val, _count_val, _eps=1e-5):
        '''average'''
        return sum_val


class RealMeter(AbstractMeter):
    '''Real Meter'''

    need_reduce_sum = False
    need_reduce_count = False

    def add(self, val, _count=1):
        '''add'''
        self._value = val
        self._sum = val

    def add_val(self, val):
        '''add val.'''
        self._value = val
        self._sum = val

    def add_count(self, count):
        '''add count.'''

    @staticmethod
    def do_get(sum_val, _count_val, _eps=1e-5):
        '''average'''
        return sum_val
