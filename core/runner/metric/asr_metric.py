'''Asr Metric'''
# pylint:disable=too-many-lines
import math
import torch
from core.utils import get_rank, get_world_size, logging
from .base_metric import BaseMetric
from .meter import SpeedMeter, WeightedMeter, BaseMeter, SumMeter, ExponentialWeightedMeter


class AsrMetric(BaseMetric):
    '''Asr Metric

    ::

        Calculation Formula
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        ctc_loss=WeightedMeter,
        las_loss=WeightedMeter,
        distiller_loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                    'time',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in ('loss', 'nll_loss', 'ctc_loss', 'las_loss', 'distiller_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('dist')
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


class AsrCaseMetric(AsrMetric):
    '''Metric for causal&non-causal joint training
    Calculation Formula
    average(tps) = sum(target_size) / sum(time)
    avreage(fps) = sum(frame_size) / sum(time)
    average(loss) = sum(loss * target_size) / sum(target_size)
    average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)

    average(c_loss) = sum(c_loss * c_target_size) / sum(c_target_size)
    average(nc_loss) = sum(nc_loss * nc_target_size) / sum(nc_target_size)

    average(c_nll_loss) = sum(nll_loss * c_target_size) / sum(c_target_size)
    average(nc_nll_loss) = sum(nll_loss * nc_target_size) / sum(nc_target_size)

    average(c_acc) = sum(c_acc * c_target_size) / sum(c_target_size)
    average(nc_acc) = sum(nc_acc * nc_target_size) / sum(nc_target_size)

    avreage(c_cer) = sum(c_error_char_num)  / sum(c_total_char_num)
    avreage(nc_cer) = sum(nc_error_char_num)  / sum(nc_total_char_num)
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
        c_loss=WeightedMeter,
        c_nll_loss=WeightedMeter,
        c_ctc_loss=WeightedMeter,
        c_las_loss=WeightedMeter,
        c_acc=WeightedMeter,
        c_cer=BaseMeter,
        nc_loss=WeightedMeter,
        nc_nll_loss=WeightedMeter,
        nc_ctc_loss=WeightedMeter,
        nc_las_loss=WeightedMeter,
        nc_acc=WeightedMeter,
        nc_cer=BaseMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                    'time',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in ('loss', 'nll_loss'):
                count = data_dict.get('tgt_size')
            elif key in ('c_loss', 'c_nll_loss', 'c_acc', 'c_ctc_loss', 'c_las_loss'):
                count = data_dict.get('c_tgt_size')
            elif key in ('nc_loss', 'nc_nll_loss', 'nc_acc', 'nc_ctc_loss', 'nc_las_loss'):
                count = data_dict.get('nc_tgt_size')
                count = data_dict.get('nc_tgt_size')
            elif key == 'c_cer':
                count = data_dict.get('c_dist')
            elif key == 'nc_cer':
                count = data_dict.get('nc_dist')
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


class SDTMetric(BaseMetric):
    '''SDT Metric for rnnt mbr and las mwer

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        average(sdt_loss) = sum(sdt_loss * batch_size) / sum(batch_size)
        average(avg_wer) = sum(avg_wer * batch_size) / sum(batch_size)
        average(avg_best_wer) = sum(avg_best_wer * batch_size) / sum(batch_size)
        average(avg_worst_wer) = sum(avg_worst_wer * batch_size) / sum(batch_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        sdt_loss=WeightedMeter,
        avg_wer=WeightedMeter,
        avg_best_wer=WeightedMeter,
        avg_worst_wer=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in ('loss', 'nll_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key in ('sdt_loss', 'avg_wer', 'avg_worst_wer', 'avg_best_wer'):
                count = data_dict.get('utt_num')
            elif key in ('cer',):
                count = data_dict.get('dist')
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
                'rank %d: sdt_metric: value is not finite for keys: %r', rank, skipped_key
            )


class UniversalAsrMetric(BaseMetric):
    '''Asr Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        loss_offline=WeightedMeter,
        nll_loss_offline=WeightedMeter,
        loss_stream=WeightedMeter,
        nll_loss_stream=WeightedMeter,
        distillation_loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        cer_offline=BaseMeter,
        cer_stream=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint: disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key == 'utt_num':
                val = val * world_size
                count = 1
            elif key in (
                'loss',
                'nll_loss_offline',
                'loss_offline',
                'nll_loss_stream',
                'loss_stream',
                'acc',
                'distillation_loss',
            ):
                count = data_dict.get('tgt_size')
            elif key in ('cer_offline', 'cer_stream', 'cer'):
                count = data_dict.get('dist')
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


class PipelineAsrMetric(AsrMetric):
    '''Asr Metric for pipeline runner

    .. note::
        the metric is based on AsrMetric,and add the function _cal_output()
    '''

    def _cal_output(self):
        '''
        calculate outputs and set to output buffer.
        '''
        self._output_buffer.clear()
        self._cal_local_output()


class Wav2vecMetric(BaseMetric):
    '''Wav2vec Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(contrastive_loss) = sum(contrastive_loss * target_size) / sum(target_size)
        average(diversity_loss) = sum(diversity_loss * target_size) / sum(target_size)
        average(feature_l2_loss) = sum(feature_l2_loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        contrastive_loss=WeightedMeter,
        diversity_loss=WeightedMeter,
        feature_l2_loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in (
                'loss',
                'acc',
                'contrastive_loss',
                'feature_l2_loss',
                'diversity_loss',
            ):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('dist')
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
                'rank %d: wav2vec_metric: value is not finite for keys: %r', rank, skipped_key
            )


class BestrqMetric(BaseMetric):
    '''Wav2vec Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        frame_num=SumMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in (
                'loss',
                'acc',
            ):
                count = data_dict.get('tgt_size')
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
                'rank %d: bestrq_metric: value is not finite for keys: %r', rank, skipped_key
            )


# pylint:disable=too-many-branches
class MostMetric(BaseMetric):
    '''Most Metric
    Calculation Formula
    average(acc) = sum(acc * target_size) / sum(target_size)
    average(loss) = sum(loss * target_size) / sum(target_size)
    average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
    average(tps) = sum(target_size) / sum(time)
    avreage(fps) = sum(frame_size) / sum(time)
    avreage(cer) = sum(error_char_num)  / sum(total_char_num)
    dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        paired_loss=WeightedMeter,  # paired RNN-T loss
        dur_loss=WeightedMeter,  # duration loss
        mm_loss=WeightedMeter,  # modality matching loss
        speech_loss=WeightedMeter,  # speech reconstruction loss
        text_loss=WeightedMeter,  # text reconstruction loss
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,  # paired RNN-T cer
        text_cer=BaseMeter,  # text reconstruction cer
        gnorm=BaseMeter,
        utt_num=SumMeter,
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
                    'time',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key.endswith('loss'):
                denorm = '_'.join([key.split('_')[0], 'denorm'])
                count = data_dict.get(denorm)
            elif key.endswith('cer'):
                denorm = key[:-3] + 'dist'
                count = data_dict.get(denorm)
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

class BestrqMetric(BaseMetric):
    '''Wav2vec Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        frame_num=SumMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in (
                'loss',
                'acc',
            ):
                count = data_dict.get('tgt_size')
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
                'rank %d: bestrq_metric: value is not finite for keys: %r', rank, skipped_key
            )


class HuBERTMetric(BaseMetric):
    '''HuBERT Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        loss_m=WeightedMeter,
        loss_u=WeightedMeter,
        acc_m=WeightedMeter,
        acc_u=WeightedMeter,
        fps=SpeedMeter,
        gnorm=BaseMeter,
        frame_num=SumMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
            if key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key.startswith('loss') or key.startswith('acc'):
                denorm = '_'.join(['cnt'] + key.split('_')[1:])
                count = data_dict[denorm]
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
                'rank %d: bestrq_metric: value is not finite for keys: %r', rank, skipped_key
            )



# TODO: refactor this implementation to corresponding with common asr metrics
class AsrCifTrainMetric(BaseMetric):
    '''Asr Metric for CIF training

    ::

        Calculation Formula:
        loss = (1 - momentum) * curr_loss + momentum * loss
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=ExponentialWeightedMeter,
        ce_loss=ExponentialWeightedMeter,
        ctc_loss=ExponentialWeightedMeter,
        quantity_loss=ExponentialWeightedMeter,
        hybrid_ce_loss=ExponentialWeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

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
            elif key in ('loss', 'ce_loss', 'ctc_loss', 'quantity_loss', 'acc', 'hybrid_ce_loss'):
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
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class AsrCifValMetric(BaseMetric):
    '''Asr Metric for CIF validation

    ::

        Calculation Formula:
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(ce_loss) = sum(ce_loss * target_size) / sum(target_size)
        average(ctc_loss) = sum(ctc_loss * target_size) / sum(target_size)
        average(quantity_loss) = sum(quantity_loss * target_size) / sum(target_size)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        ce_loss=WeightedMeter,
        ctc_loss=WeightedMeter,
        quantity_loss=WeightedMeter,
        hybrid_ce_loss=WeightedMeter,
        acc=WeightedMeter,
        cer=BaseMeter,
        total_error_count=SumMeter,
        sub_error_count=SumMeter,
        del_error_count=SumMeter,
        ins_error_count=SumMeter,
        total_count=SumMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        utt_num=SumMeter,
    )

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
            elif key in ('loss', 'ce_loss', 'ctc_loss', 'quantity_loss', 'hybrid_ce_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('total_count')
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
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class UniversalAsrCifTrainMetric(BaseMetric):
    '''Asr Metric for CIF training

    ::

        Calculation Formula:
        loss = (1 - momentum) * curr_loss + momentum * loss
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=ExponentialWeightedMeter,
        stream_ce_loss=ExponentialWeightedMeter,
        stream_ctc_loss=ExponentialWeightedMeter,
        stream_quantity_loss=ExponentialWeightedMeter,
        nonstream_ce_loss=ExponentialWeightedMeter,
        nonstream_ctc_loss=ExponentialWeightedMeter,
        nonstream_quantity_loss=ExponentialWeightedMeter,
        kld_on_ctc=ExponentialWeightedMeter,
        kld_on_ce=ExponentialWeightedMeter,
        stream_acc=WeightedMeter,
        nonstream_acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

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
            elif key in (
                'loss',
                'stream_ce_loss',
                'stream_ctc_loss',
                'stream_quantity_loss',
                'nonstream_ce_loss',
                'nonstream_ctc_loss',
                'nonstream_quantity_loss',
                'kld_on_ctc',
                'kld_on_ce',
                'stream_acc',
                'nonstream_acc',
            ):
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
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class UniversalAsrCifValMetric(BaseMetric):
    '''Asr Metric for CIF validation

    ::

        Calculation Formula:
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(ce_loss) = sum(ce_loss * target_size) / sum(target_size)
        average(ctc_loss) = sum(ctc_loss * target_size) / sum(target_size)
        average(quantity_loss) = sum(quantity_loss * target_size) / sum(target_size)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=ExponentialWeightedMeter,
        stream_ce_loss=ExponentialWeightedMeter,
        stream_ctc_loss=ExponentialWeightedMeter,
        stream_quantity_loss=ExponentialWeightedMeter,
        nonstream_ce_loss=ExponentialWeightedMeter,
        nonstream_ctc_loss=ExponentialWeightedMeter,
        nonstream_quantity_loss=ExponentialWeightedMeter,
        kld_on_ctc=ExponentialWeightedMeter,
        kld_on_ce=ExponentialWeightedMeter,
        stream_acc=WeightedMeter,
        nonstream_acc=WeightedMeter,
        stream_cer=BaseMeter,
        stream_total_error_count=SumMeter,
        stream_sub_error_count=SumMeter,
        stream_del_error_count=SumMeter,
        stream_ins_error_count=SumMeter,
        stream_total_count=SumMeter,
        nonstream_cer=BaseMeter,
        nonstream_total_error_count=SumMeter,
        nonstream_sub_error_count=SumMeter,
        nonstream_del_error_count=SumMeter,
        nonstream_ins_error_count=SumMeter,
        nonstream_total_count=SumMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        utt_num=SumMeter,
    )

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
            elif key in (
                'loss',
                'stream_ce_loss',
                'stream_ctc_loss',
                'stream_quantity_loss',
                'nonstream_ce_loss',
                'nonstream_ctc_loss',
                'nonstream_quantity_loss',
                'kld_on_ctc',
                'kld_on_ce',
                'astream_cc',
                'nonstream_acc',
            ):
                count = data_dict.get('tgt_size')
            elif key in ('stream_cer',):
                count = data_dict.get('stream_total_count')
            elif key in ('nonstream_cer',):
                count = data_dict.get('nonstream_total_count')
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
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )


class LasMetric(BaseMetric):
    '''Las Metric

    ::

        Calculation Formula
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(ctc_loss) = sum(ctc_loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        ctc_loss=WeightedMeter,
        nll_loss=WeightedMeter,
        acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                    'time',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in ('loss', 'nll_loss', 'acc', 'ctc_loss'):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('dist')
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
                'rank %d: las_metric: value is not finite for keys: %r', rank, skipped_key
            )


class KmeansMetric(BaseMetric):
    '''k-means Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        distance=WeightedMeter,
        fps=SpeedMeter,
        frame_num=SumMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
        self._output_buffer.clear()

        assert isinstance(data_dict, dict)

        rank = get_rank()
        world_size = get_world_size()

        skipped_key = []
        for key in data_dict.keys():
            if key not in ('frame_size',) and key not in self._meters:
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in ('distance', 'loss'):
                count = data_dict.get('frame_size')
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
                'rank %d: bestrq_metric: value is not finite for keys: %r', rank, skipped_key
            )


class SpokenLmAsrMetric(BaseMetric):
    '''SpokenLm Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        acoustic_nll_loss=WeightedMeter,
        text_nll_loss=WeightedMeter,
        # lprobs=WeightedMeter,
        # ac_lprobs=WeightedMeter,
        # txt_lprobs=WeightedMeter,
        acc=WeightedMeter,
        ac_acc=WeightedMeter,
        txt_acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
        token_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in (
                'loss',
                'nll_loss',
                'acc',
            ):
                count = data_dict.get('tgt_size') + data_dict.get('frame_size')
            elif key in ('acoustic_nll_loss', 'ac_lprobs', 'ac_acc'):
                count = data_dict.get('frame_size')
            elif key in ('text_nll_loss', 'txt_lprobs', 'txt_acc'):
                count = data_dict.get('tgt_size')
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
                'rank %d: spokenlm_metric: value is not finite for keys: %r', rank, skipped_key
            )


class SpokenLmAffixMetric(BaseMetric):
    '''SpokenLm Metric

    ::

        Calculation Formula:
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        prefix_nll_loss=WeightedMeter,
        affix_nll_loss=WeightedMeter,
        # lprobs=WeightedMeter,
        # ac_lprobs=WeightedMeter,
        # txt_lprobs=WeightedMeter,
        acc=WeightedMeter,
        prefix_acc=WeightedMeter,
        affix_acc=WeightedMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
        token_num=SumMeter,
        valid_token_ratio=WeightedMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                count = None
            elif key == 'frame_size':
                val = val * world_size
                key = 'fps'
                count = None
            elif key in (
                'loss',
                'nll_loss',
                'acc',
                'prefix_nll_loss',
                'prefix_lprobs',
                'prefix_acc',
                'affix_nll_loss',
                'affix_lprobs',
                'affix_acc',
            ):
                count = data_dict.get('tgt_size')
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
                'rank %d: spokenlm_metric: value is not finite for keys: %r', rank, skipped_key
            )


class SpokenlmMetric(BaseMetric):
    '''Asr Metric

    ::

        Calculation Formula
        average(acc) = sum(acc * target_size) / sum(target_size)
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(nll_loss) = sum(nll_loss * target_size) / sum(target_size)
        average(tps) = sum(target_size) / sum(time)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        nll_loss=WeightedMeter,
        # distiller_loss=WeightedMeter,
        acc=WeightedMeter,
        tps=SpeedMeter,
        valid_token_ratio=WeightedMeter,
        cer=BaseMeter,
        gnorm=BaseMeter,
        utt_num=SumMeter,
    )

    @torch.no_grad()
    def update(self, data_dict):
        '''update'''
        # pylint:disable=too-many-branches
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
                    'time',
                )
                and key not in self._meters
            ):
                continue
            val = data_dict[key]
            if key == 'tgt_size':
                key = 'tps'
                val = val * world_size
                count = None
            elif key in ('loss', 'nll_loss', 'distiller_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('dist')
            elif key in ('valid_token_ratio'):
                count = data_dict.get('data_num')
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


class AuLlmValMetric(BaseMetric):
    '''Asr Metric for CIF validation

    ::

        Calculation Formula:
        average(loss) = sum(loss * target_size) / sum(target_size)
        average(ce_loss) = sum(ce_loss * target_size) / sum(target_size)
        average(ctc_loss) = sum(ctc_loss * target_size) / sum(target_size)
        average(quantity_loss) = sum(quantity_loss * target_size) / sum(target_size)
        avreage(cer) = sum(error_char_num)  / sum(total_char_num)
        average(tps) = sum(target_size) / sum(time)
        avreage(fps) = sum(frame_size) / sum(time)
        dist: number of word
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        loss=WeightedMeter,
        ce_loss=WeightedMeter,
        acc=WeightedMeter,
        cer=BaseMeter,
        total_error_count=SumMeter,
        sub_error_count=SumMeter,
        del_error_count=SumMeter,
        ins_error_count=SumMeter,
        total_count=SumMeter,
        fps=SpeedMeter,
        tps=SpeedMeter,
        utt_num=SumMeter,
    )

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
            elif key in ('loss', 'ce_loss', 'acc'):
                count = data_dict.get('tgt_size')
            elif key in ('cer',):
                count = data_dict.get('total_count')
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
                'rank %d: asr_metric: value is not finite for keys: %r', rank, skipped_key
            )

