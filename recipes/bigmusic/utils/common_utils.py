import itertools
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

__author__ = ["chaonan99"]


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def object_to_dict(obj, classkey=None):
    """https://stackoverflow.com/a/1118038/6225574"""
    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[k] = object_to_dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return object_to_dict(obj._ast())
    elif (
        hasattr(obj, "__iter__")
        and not isinstance(obj, str)
        and not isinstance(obj, bytes)
    ):
        return [object_to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict(
            [
                (key, object_to_dict(value, classkey))
                for key, value in obj.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def colorful_sequence(cs, us):
    """ """
    if len(us) < 1:
        return []
    ic, iu = 0, 0
    res = []
    while ic < len(cs) - 1 and cs[ic + 1] <= us[iu]:
        ic += 1
    while iu < len(us) - 1 and cs[ic] >= us[iu + 1]:
        res.append(-1)
        iu += 1
    while ic < len(cs) - 1 and iu < len(us) - 1:
        longest_i = -1
        longest_len = 0
        while ic < len(cs) - 1 and cs[ic] < us[iu + 1]:
            curr_len = min(cs[ic + 1], us[iu + 1]) - max(cs[ic], us[iu])
            if curr_len > longest_len:
                longest_len = curr_len
                longest_i = ic
            ic += 1
        res.append(longest_i)
        iu += 1
        ic -= 1
    while iu < len(us) - 1:
        res.append(-1)
        iu += 1
    return res


def colorful_segments(
    cs: List[Tuple[float, float]],
    us: List[Tuple[float, float]],
):
    """
    Finds the index of intervals in 'cs' with the longest overlap for each interval in 'us'.

    For each interval in 'us', the function looks through 'cs' to find the interval that has
    the longest overlapping section with it. If no intervals in 'cs' overlap with an interval
    in 'us', the result is -1 for that position.

    Parameters:
    - cs (List[Tuple[float, float]]): A list of intervals (start, end) sorted by their start times.
    - us (List[Tuple[float, float]]): Another list of intervals (start, end) sorted by their start times.

    Returns:
    - List[int]: A list of indices, each corresponding to the interval in 'cs'
      with the longest overlap with the interval in 'us' at the same position. If
      there is no overlapping interval in 'cs', the value is -1.

    Note: It is assumed that the intervals in both 'cs' and 'us' are sorted by their start times.

    Example:
    >>> cs = [(1, 4), (5, 8), (9, 12)]
    >>> us = [(2, 3), (6, 10)]
    >>> colorful_segments(cs, us)
    [0, 1] # Since (1, 4) overlaps with (2, 3) completely and (5, 8) has the longest overlap with (6, 10)
    """
    if len(us) < 1:
        return []
    ic, iu = 0, 0
    res = []
    while ic < len(cs) and iu < len(us):
        while ic < len(cs) and cs[ic][1] <= us[iu][0]:
            ic += 1
        if ic >= len(cs):
            break
        while iu < len(us) and cs[ic][0] >= us[iu][1]:
            res.append(-1)
            iu += 1
        if iu >= len(us):
            break
        longest_i = -1
        longest_len = 0
        fallback_count = 0
        while ic < len(cs) and cs[ic][0] < us[iu][1]:
            curr_len = min(cs[ic][1], us[iu][1]) - max(cs[ic][0], us[iu][0])
            if curr_len > longest_len:
                longest_len = curr_len
                longest_i = ic
            if cs[ic][1] > us[iu][1]:
                fallback_count += 1
            ic += 1
        res.append(longest_i)
        iu += 1
        ic -= fallback_count
    while iu < len(us):
        res.append(-1)
        iu += 1
    return res


class TqdmWrapper:
    """Provides a "verbose" parameter so that you can actually disable
    tqdm by just setting verbose to False without deleting tqdm calls.
    """

    def __init__(self, verbose: bool):
        self.pbar = tqdm()
        self.verbose = verbose

    def update(self):
        if self.verbose:
            self.pbar.update()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if self.verbose:
            self.pbar.set_postfix(ordered_dict, refresh, **kwargs)

    def close(self):
        if self.verbose:
            self.pbar.close()
