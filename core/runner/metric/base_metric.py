'''Base Metric'''
from collections import OrderedDict
import numpy as np
import torch
from core.utils import get_world_size, dist_allreduce, ReduceOp


class BaseMetric:
    '''Base Metric.

    .. note::
    base metric 是所有 metric 的基类,为metric的书写提供了基本的思路,其中_name2type中可以重定义你需要的参数.

    When user define a new Metic, just need 2 step.
        1. Define _name2type. like _name2type = {'loss': BaseMeter, 'fps': SpeedMeter}
        2. implement update function, which process the data from your model.
    '''

    _name2type = {}

    def __init__(self, deny_names=()):
        '''
        init the metric.
        '''
        self._meters = OrderedDict()
        self._output_buffer = OrderedDict()
        self._init_meters(deny_names)

    def _init_meters(self, deny_names):
        '''mertrics init'''
        for name, cls_type in self._name2type.items():
            if name in deny_names:
                continue
            self._meters[name] = cls_type()

    def add_meter(self, name, cls_type):
        '''add meter'''
        self._meters[name] = cls_type()

    def get_meter(self, name):
        '''get meter'''
        if name not in self._meters:
            raise RuntimeError('Metric: unexpected key', name)
        return self._meters[name]

    @torch.no_grad()
    def update(self, _data_dict):
        '''
        update user defined data to this metric by self._add function
        '''
        raise NotImplementedError

    def _add(self, key, val, count):
        '''add a meter value and count.'''
        if key not in self._meters:
            raise RuntimeError('Metric: unexpected key', key)
        self._output_buffer.clear()
        if count is None:
            self._meters[key].add_val(val)
        elif val is None:
            self._meters[key].add_count(count)
        else:
            self._meters[key].add(val, count)

    def reset(self):
        '''
        clear previous data.
        '''
        self._output_buffer.clear()
        for meter in self._meters.values():
            meter.reset()

    def get_value(self, key):
        '''get value'''
        if key not in self._meters:
            raise RuntimeError('Metric: unexpected key', key)
        if len(self._output_buffer) == 0:
            self._cal_output()
        return self._output_buffer[key]

    def get(self, copy=False):
        '''
        get all output value

        Args:
            copy(bool): whether you will modify the output dict.
        Return:
            (dict): a dict with all metric value.
        '''
        if len(self._output_buffer) == 0:
            self._cal_output()
        if copy:
            return self._output_buffer.copy()
        return self._output_buffer

    def lazy_get(self):
        '''
        get all output value by latest calculate
        in case of extra dist sync.
        '''
        return self._output_buffer

    def _cal_output(self):
        '''
        calculate outputs and set to output buffer.
        '''
        self._output_buffer.clear()
        world_size = get_world_size()
        if world_size == 1:
            self._cal_local_output()
        else:
            self._cal_dist_output()

    def _cal_local_output(self):
        '''
        calculate outputs for this rank and set to output buffer.
        '''
        for name, meter in self._meters.items():
            value = meter.get()
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.item()
            self._output_buffer[name] = value

    def _cal_dist_output(self):
        '''
        calculate outputs for this rank and set to output buffer.
        '''
        # record which value need to do reduce.
        value_buf = {}
        sum_to_reduce, count_to_reduce = [], []
        sum_to_reduce_name, count_to_reduce_name = [], []
        for name, meter in self._meters.items():
            value_buf[name] = [meter.sum, meter.count]
            if meter.need_reduce_sum:
                sum_to_reduce.append(meter.sum)
                sum_to_reduce_name.append(name)
            if meter.need_reduce_count:
                count_to_reduce.append(meter.count)
                count_to_reduce_name.append(name)

        # do dist reduce
        if len(sum_to_reduce_name) > 0 or len(count_to_reduce_name) > 0:
            lst = sum_to_reduce + count_to_reduce
            tensor = torch.as_tensor(lst, dtype=torch.float32, device='cuda')
            dist_allreduce(tensor, name='reduce_metric_sums', op=ReduceOp.SUM)
            tensor = tensor.cpu()
            sum_num = len(sum_to_reduce)
            sum_tensor, count_tensor = tensor[:sum_num], tensor[sum_num:]
            for name, sum_val in zip(sum_to_reduce_name, sum_tensor):
                value_buf[name][0] = sum_val
            for name, count_val in zip(count_to_reduce_name, count_tensor):
                value_buf[name][1] = count_val

        for (name, lst), meter in zip(value_buf.items(), self._meters.values()):
            sum_val, count_val = lst
            value = meter.do_get(sum_val, count_val)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.item()
            self._output_buffer[name] = value
