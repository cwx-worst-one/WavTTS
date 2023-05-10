'''
time mask module.
'''
import random
import numpy
import torch
from core.extensions import time_mask as time_mask_cuda
from .preprocess import PREPROCESS


def time_mask(spec, time=40, n_mask=2, replace_with_zero=True, inplace=False, max_time_p=0.2):
    """
    freq mask for spec agument
    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        if hasattr(spec, 'clone'):
            cloned = spec.clone()
        else:
            cloned = spec.copy()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, time, size=(n_mask,))
    mask_start_list = []
    mask_length_list = []
    for t in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        t = min(t, int(max_time_p * len_spectro))
        mask_length_list.append(t)
        mask_start_list.append(t_zero)
        mask_end = t_zero + t
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned, mask_start_list, mask_length_list


@PREPROCESS.register_module()
class TimeMask:
    '''time mask process class.'''

    def __init__(
        self,
        time_mask_size=40,
        time_mask_num=2,
        key='fbank',
        mask_key='src_mask',
        replace_with_zero=False,
        inplace=False,
        max_time_p=0.2,
        skip_num=0,
        adap_timemask=False,
        adap_timemask_ratio=0,
    ):
        '''init.'''
        self.time = time_mask_size
        self.n_mask = time_mask_num
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.max_time_p = max_time_p
        self.key = key
        self.mask_key = mask_key
        self.skip_num = skip_num
        self.adap_timemask = adap_timemask
        self.adap_timemask_ratio = adap_timemask_ratio

    def __call__(self, item, **_kwargs):
        '''do time mask.'''
        if item is None or self.key not in item:
            return item
        if self.skip_num > 0:
            self.skip_num -= 1
            return item

        fbank = item[self.key]
        if len(fbank.shape) == 3:
            bsz = fbank.shape[0]
            outs = []
            for idx in range(bsz):
                if self.adap_timemask:
                    self.time = max(int(fbank.shape[1] * float(self.adap_timemask_ratio)), 1)
                out, _, _ = time_mask(
                    fbank[idx],
                    self.time,
                    self.n_mask,
                    self.replace_with_zero,
                    self.inplace,
                    self.max_time_p,
                )
                outs.append(out)
            fbank = torch.stack(outs)
            if self.mask_key in item:
                mask = item[self.mask_key]
                fbank *= mask.unsqueeze(-1)
        else:
            if self.adap_timemask:
                self.time = max(int(fbank.shape[0] * float(self.adap_timemask_ratio)), 1)
            fbank, _, _ = time_mask(
                fbank, self.time, self.n_mask, self.replace_with_zero, self.inplace, self.max_time_p
            )
        item[self.key] = fbank
        return item


@PREPROCESS.register_module()
class DynamicTimeMask:
    '''dynamic time mask process class.'''

    def __init__(
        self,
        time_mask_size=40,
        time_mask_block=80,
        time_mask_num_dither=0,
        replace_with_zero=False,
        inplace=True,
        max_time_p=0.2,
        key='fbank',
        skip_num=0,
    ):
        '''init.'''
        self.time = time_mask_size
        self.block = time_mask_block
        self.dither = time_mask_num_dither
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.max_time_p = max_time_p
        self.key = key
        self.skip_num = skip_num

    def __call__(self, item, **_kwargs):
        '''do time mask.'''
        if item is None or self.key not in item:
            return item
        if self.skip_num > 0:
            self.skip_num -= 1
            return item

        fbank = item[self.key]
        n_mask = (fbank.shape[0] + self.block - 1) // self.block
        if self.dither > 0:
            min_n_mask = max(1, n_mask - self.dither)
            max_n_mask = n_mask + self.dither
            n_mask = random.randint(min_n_mask, max_n_mask)
        fbank, _, _ = time_mask(
            fbank, self.time, n_mask, self.replace_with_zero, self.inplace, self.max_time_p
        )
        item[self.key] = fbank
        return item


@PREPROCESS.register_module()
class BatchTimeMask:
    '''time mask process class.'''

    def __init__(
        self,
        time_mask_size=40,
        time_mask_num=2,
        key='fbank',
        replace_with_zero=False,
        inplace=False,
        max_time_p=0.2,
        skip_num=0,
        randdisurb=0.1,
    ):
        '''init.'''
        self.time = time_mask_size
        self.n_mask = time_mask_num
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.max_time_p = max_time_p
        self.key = key
        self.skip_num = skip_num
        self.randdisurb = randdisurb

    def __call__(self, item, **_kwargs):
        '''do time mask.'''
        if item is None:
            return None
        if self.skip_num > 0:
            self.skip_num -= 1
            return item
        cloned = item[self.key]
        item[self.key] = time_mask_cuda(
            cloned, self.replace_with_zero, self.n_mask, self.time, self.randdisurb
        )
        return item


@PREPROCESS.register_module()
class BatchDynamicTimeMask:
    '''dynamic time mask process class.'''

    def __init__(
        self,
        time_mask_size=40,
        time_mask_block=80,
        time_mask_num_dither=0,
        replace_with_zero=False,
        inplace=True,
        max_time_p=0.2,
        key='fbank',
        skip_num=0,
        randdisurb=0.1,
    ):
        '''init.'''
        self.time = time_mask_size
        self.block = time_mask_block
        self.dither = time_mask_num_dither
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.max_time_p = max_time_p
        self.key = key
        self.skip_num = skip_num
        self.randdisurb = randdisurb

    def __call__(self, item, **_kwargs):
        '''do time mask.'''
        if item is None:
            return None
        if self.skip_num > 0:
            self.skip_num -= 1
            return item

        fbank = item[self.key]
        n_mask = (fbank.shape[1] + self.block - 1) // self.block
        assert fbank.dim == 3
        item[self.key] = time_mask_cuda(
            fbank,
            self.replace_with_zero,
            n_mask,
            self.time,
            self.randdisurb,
            self.max_time_p,
            True,
            self.dither,
            self.block,
        )
        return item


@PREPROCESS.register_module()
class TimeMaskCollate(TimeMask):
    '''time mask for batch transform.'''

    def __call__(self, bucket_list, batch_out):
        '''do freq mask.'''
        if not bucket_list:
            return
        super().__call__(batch_out)
