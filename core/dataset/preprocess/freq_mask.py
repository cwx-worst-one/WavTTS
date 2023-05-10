'''
freq mask.
'''
import random
import numpy
import torch
from core.extensions import freq_mask as freq_mask_cuda
from .preprocess import PREPROCESS


def freq_mask(x, freq=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        if hasattr(x, 'clone'):
            cloned = x.clone()
        else:
            cloned = x.copy()
    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, freq, size=(n_mask,))
    mask_start_list = []
    mask_length_list = []

    for f in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end = f_zero + f

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        mask_length_list.append(f)
        mask_start_list.append(f_zero)

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned, mask_start_list, mask_length_list


@PREPROCESS.register_module()
class FreqMask:
    '''freq mask.'''

    def __init__(
        self,
        freq_mask_size=30,
        freq_mask_num=2,
        replace_with_zero=False,
        inplace=False,
        key='fbank',
        mask_key='src_mask',
        skip_num=0,
    ):
        '''init.'''
        self.freq = freq_mask_size
        self.n_mask = freq_mask_num
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.key = key
        self.mask_key = mask_key
        self.skip_num = skip_num

    def __call__(self, item, **_kwargs):
        '''do freq mask.'''
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
                out, _, _ = freq_mask(
                    fbank[idx], self.freq, self.n_mask, self.replace_with_zero, self.inplace
                )
                outs.append(out)
            fbank = torch.stack(outs)
            if self.mask_key in item:
                mask = item[self.mask_key]
                fbank *= mask.unsqueeze(-1)
        else:
            fbank, _, _ = freq_mask(
                fbank, self.freq, self.n_mask, self.replace_with_zero, self.inplace
            )
        item[self.key] = fbank

        return item


@PREPROCESS.register_module()
class BatchFreqMask:
    '''freq mask.'''

    def __init__(
        self,
        freq_mask_size=30,
        freq_mask_num=2,
        replace_with_zero=False,
        inplace=False,
        key='fbank',
        skip_num=0,
        randdisurb=0.0,
    ):
        '''init.'''
        self.freq = freq_mask_size
        self.n_mask = freq_mask_num
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace
        self.key = key
        self.skip_num = skip_num
        self.randdisurb = randdisurb

    def __call__(self, item, **_kwargs):
        '''do freq mask.'''
        if item is None:
            return None
        if self.skip_num > 0:
            self.skip_num -= 1
            return item

        fbank = item[self.key]
        item[self.key] = freq_mask_cuda(
            fbank, self.replace_with_zero, self.n_mask, self.freq, self.randdisurb
        )
        return item


@PREPROCESS.register_module()
class FreqMaskCollate(FreqMask):
    '''freq mask for batch transform.'''

    def __call__(self, bucket_list, batch_out):
        '''do freq mask.'''
        if not bucket_list:
            return
        super().__call__(batch_out)
