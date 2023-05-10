'''
time warp.
'''

import random
import numpy
from PIL import Image
from PIL.Image import BICUBIC
from .preprocess import PREPROCESS


def time_warp(x, max_time_warp=80, inplace=False):
    """
    time warp for spec augment
    move random center frame by the random width ~ uniform(-window, window)
    Args:
        x(numpy.ndarray): spectrogram (time, freq)
        max_time_warp(int): maximum time frames to warp
        inplace(bool): overwrite x with the result
    Returns:
        numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp

    t = x.shape[0]
    if t - window <= window:
        return x
    # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
    center = random.randrange(window, t - window)
    # 1 ... t - 1
    warped = random.randrange(center - window, center + window) + 1

    left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
    right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
    if inplace:
        x[:warped] = left
        x[warped:] = right
        return x
    return numpy.concatenate((left, right), 0)


@PREPROCESS.register_module()
class TimeWarp:
    '''
    Time Warp.
    move random center frame by the random width ~ uniform(-window, window)
    '''

    def __init__(self, time_warp_size=80, inplace=False):
        '''init.'''
        self.max_time_warp = time_warp_size
        self.inplace = inplace

    def __call__(self, item, **_kwargs):
        '''
        do time warp.
        Args:
            item(dict): item data.
        return:
            dict: processed item data.
        '''
        if item is None:
            return None

        x = item['fbank']
        item['fbank'] = time_warp(x)

        return item
