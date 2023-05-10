'''
block dropout module.
'''

import numpy
from .preprocess import PREPROCESS


def block_dropout_fn(
    spec, dropout=0.3, time_p=0.05, freq_p=0.2, replace_with_zero=True, inplace=False
):
    """
    block mask for spec agument
    :param numpy.ndarray spec: (time, freq)
    :param float time_p: block_length / spec_len
    :param float freq_p: block_height / spec_height
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()

    mask_val = 0 if replace_with_zero else cloned.mean()

    spec_len, spec_height = cloned.shape
    block_len, block_height = int(time_p * spec_len), int(freq_p * spec_height)
    block_len = max(block_len, 1)
    time_start = 0

    while time_start < spec_len:
        time_end = min(time_start + block_len, spec_len)
        freq_start = 0
        while freq_start < spec_height:
            freq_end = min(freq_start + block_height, spec_height)
            if numpy.random.binomial(1, dropout):
                cloned[time_start:time_end, freq_start:freq_end] = mask_val
            freq_start = freq_end
        time_start = time_end
    return cloned


@PREPROCESS.register_module()
class BlockDropout:
    '''block dropout class.'''

    def __init__(
        self,
        key='fbank',
        block_dropout=0.3,
        block_time_p=0.05,
        block_freq_p=0.2,
        replace_with_zero=True,
        inplace=False,
    ):
        '''init.'''
        self.key = key
        self.dropout = block_dropout
        self.time_p = block_time_p
        self.freq_p = block_freq_p
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, item, **_kwargs):
        '''do block dropout.'''
        if item is None:
            return None

        fbank = item[self.key]
        item[self.key] = block_dropout_fn(
            fbank, self.dropout, self.time_p, self.freq_p, self.replace_with_zero, self.inplace
        )
        return item
