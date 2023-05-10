'''Aed Metric'''
import math
import torch
from core.utils import get_rank, get_world_size, logging
from .base_metric import BaseMetric
from .meter import SpeedMeter, WeightedMeter, BaseMeter, RealMeter


class EmotionMetric(BaseMetric):
    '''Emotion Metric

    ::

        Calculation Formula
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        UA=WeightedMeter,
        UF=WeightedMeter,
        WP=WeightedMeter,
        WA=WeightedMeter,
        WF=WeightedMeter,
        CCC=WeightedMeter,
        RMSE=WeightedMeter,
    )

    def update(self, data_dict):
        '''update'''
        self._output_buffer.clear()

        assert isinstance(data_dict, dict)

        rank = get_rank()

        skipped_key = []
        for key in data_dict.keys():
            if key not in self._meters:
                continue
            val = data_dict[key]
            if key in ('loss', 'nll_loss', 'acc'):
                count = data_dict.get('tgt_size')
            else:
                count = 1
            if math.isfinite(count) and math.isfinite(val):
                self._add(key, val, count)
            else:
                skipped_key.append(key)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class TopHitsMetric(BaseMetric):
    '''Top hits for classification metrics'''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        ce_loss=WeightedMeter,
        l2_loss=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        top1=BaseMeter,
        topk=BaseMeter,
        gnorm=BaseMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
        self._output_buffer.clear()
        rank = get_rank()
        world_size = get_world_size()
        skipped_key = []

        for key in data_dict.keys():
            if key not in ('tgt_size', 'frame_size', 'time') and key not in self._meters:
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                key = 'fps'
                val = val * world_size
                count = None
            elif key in ('loss', 'ce_loss', 'l2_loss', 'top1', 'topk'):
                count = data_dict['tgt_size']
            elif key == 'flops':
                count = None
            else:
                count = 1

            if key == 'time':
                # time is updated in different callings.
                for k, meter in self._meters.items():
                    if not isinstance(meter, SpeedMeter):
                        continue
                    self._add(k, None, val)
            if (count is None or math.isfinite(count)) and (val is None or math.isfinite(val)):
                self._add(key, val, count)
            else:
                skipped_key.append(key)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class RecallMetric(TopHitsMetric):
    '''RecallMetric'''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        ce_loss=WeightedMeter,
        l2_loss=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        top1=BaseMeter,
        topk=BaseMeter,
        gnorm=BaseMeter,
        mAP=RealMeter,
        mAUC=RealMeter,
        eer=RealMeter,
        d_prime=RealMeter,
    )
