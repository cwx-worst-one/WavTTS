'''Asr Metric'''
import math
import torch
from core.utils import get_rank, get_world_size, logging
from .base_metric import BaseMetric
from .meter import SpeedMeter, WeightedMeter, BaseMeter, SumMeter, RealMeter


class SidMetric(BaseMetric):
    '''Sid Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        margin = softmax margin if existed
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
        margin=RealMeter,
        margin_lambda=RealMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        self._output_buffer.clear()
        assert isinstance(data_dict, dict)

        rank = get_rank()
        world_size = get_world_size()

        skipped_key = []
        for key in data_dict.keys():
            if (
                key
                not in (
                    'tgt_size',
                    'frame_size',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = data_dict.get('time', self._meters['time'].value)
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = data_dict.get('time', self._meters['time'].value)
            elif key == 'utt_num':
                val = val * world_size
                count = 1
            elif key in ('loss', 'nll_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key == 'flops':
                count = data_dict.get('time', self._meters['time'].value)
            else:
                count = 1
            if math.isfinite(count) and math.isfinite(val):
                self._add(key, val, count)
            else:
                skipped_key.append(key)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: sid_metric: value is not finite for keys: %r', rank, skipped_key
            )
