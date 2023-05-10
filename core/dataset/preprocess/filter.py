'''
filters implementation.
filter item data by some condition.
'''

import random
import numpy as np
from core.utils.dist_hdfs import dist_hdfs_get
from .preprocess import PREPROCESS


# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001


def check_numerals(strings):
    '''check'''
    for char in strings:
        if u'\u0030' <= char <= u'\u0039':
            return True
    return False


@PREPROCESS.register_module()
class LabelLengthFilter:
    '''label length filter.
    let the data go, if the labels'num is greater then
    frames' num * some ratio, or if its num is outside
    min_len and max_len

    '''

    def __init__(
        self, in_out_ratio, key='label', min_len=0, max_len=float('inf'), strict_mode=True
    ):
        '''get in_out_ratio.
        Args:
            in_out_ratio(int): in out ratio condition.
        '''
        self.key = key
        self.in_out_ratio = in_out_ratio
        self.min_len = min_len
        self.max_len = max_len
        self.strict_mode = strict_mode

    def __call__(self, item_data, **_kwargs):
        '''do the filter.

        Args:
            item_data(dict): must contains fbank and label.

        Returns
            dict: item_data if the item_data meet conditions,
                  None, otherwise.
        '''
        if item_data is None or self.key not in item_data:
            return None

        frames_num = self.get_frames_num(item_data)
        if frames_num is None:
            if self.strict_mode:
                return None
            return item_data

        label_len = len(item_data[self.key])
        threshold = int(frames_num // self.in_out_ratio)
        if label_len > threshold:
            return None
        if label_len < self.min_len or label_len > self.max_len:
            return None

        return item_data

    def get_frames_num(self, item_data):
        '''get frame_num from fbank or length.'''
        fbank = item_data.get('fbank', None)
        if not isinstance(fbank, np.ndarray):
            return item_data.get('length', None)

        if len(fbank.shape) < 1 or fbank.shape[0] <= self.in_out_ratio:
            return None

        return fbank.shape[0]


@PREPROCESS.register_module()
class TagFilter:
    '''filter tag'''

    def __init__(self, select_tag=None, key='tag'):
        '''init.
        Args:
            select_tag(list of string): a list for selecting
        '''
        self.select_tag = select_tag
        self.key = key

    def __call__(self, item_data, **_kwargs):
        '''do selecting
        Args:
            item_data(dict): input data

        Returns:
            dict: item data if selected, None otherwise
        '''
        if item_data is None or (self.select_tag and not self.select(item_data[self.key])):
            return None
        return item_data

    def select(self, in_list):
        '''select
        Args:
            in_list: tag of a item

        Returns:
            bool: True if one of the item's tag is in self.select_tag
                  False otherwise
        '''
        for tag in in_list:
            if tag in self.select_tag:
                return True
        return False


@PREPROCESS.register_module()
class LengthFilter:
    '''filter long and short utterance'''

    def __init__(self, key='length', min_len=0, max_len=2000):
        '''init'''
        self.key = key
        self.max_len = max_len
        self.min_len = min_len

    def __call__(self, item_data, **_kwargs):
        '''call'''
        if item_data is None:
            return None
        data = item_data.get(self.key, None)
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            length = data.shape[0]
        else:
            length = int(data)
        if self.min_len > 0 and length < self.min_len:
            return None
        if self.max_len > 0 and length > self.max_len:
            return None
        return item_data


@PREPROCESS.register_module()
class UttidKeywordsFilter:
    '''filter certain utt if the keywords in uttid'''

    def __init__(self, key='uttid', keywords=None, dropout=1.0):
        '''init.
        Args:
            keywords(list): keywords that needed to be filtered
        '''
        self.key = key
        self.keywords = keywords
        self.dropout = dropout

    def __call__(self, item_data, **_kwargs):
        '''do selecting
        Args:
            item_data(dict): input data

        Returns:
            dict: item data if it's uttid does not contain the keywords else None
        '''
        if item_data is None:
            return None

        if self.keywords is None:
            return item_data

        uttid = item_data[self.key]
        for keyword in self.keywords:
            if keyword in uttid and (self.dropout >= 1.0 or random.random() < self.dropout):
                return None
        return item_data


def load_filter_list(filter_set):
    '''load scp'''
    if str(filter_set).startswith('hdfs'):
        filter_set = load_from_hdfs(filter_set)
        if filter_set is None:
            return set()
    filter_set_ = set()
    try:
        with open(filter_set, 'r', encoding='utf-8') as r:
            uttids = r.readlines()
        for utt in uttids:
            utt = utt.strip()
            if len(utt) == 0:
                continue
            filter_set_.add(utt)
    except Exception:
        pass
    return filter_set_


def load_from_hdfs(path):
    '''load scp from hdfs'''
    write_dir = '/tmp/'
    write_file = 'filter_list.txt'
    write_path = dist_hdfs_get(path, write_dir, write_file)
    return write_path


@PREPROCESS.register_module()
class BadNumeralsFilter:
    '''remove some sentence include Arabic numerals after TN'''

    def __init__(self, key1='uttid', key2='label', filter_set=None):
        '''init'''
        self.key1 = key1
        self.key2 = key2
        self.filter_set = set()
        if isinstance(filter_set, str):
            self.filter_set = load_filter_list(filter_set)
        elif isinstance(filter_set, (set, list, tuple)):
            self.filter_set = set(filter_set)

    def __call__(self, item_data, **_kwargs):
        '''call'''
        if item_data is None:
            return None
        uttid = item_data.get(self.key1, None)
        label = item_data.get(self.key2, None)
        if uttid is None or label is None:
            return item_data
        if isinstance(label, list):
            label = ' '.join(label)
        if uttid.strip() in self.filter_set or check_numerals(label):
            return None
        return item_data


@PREPROCESS.register_module()
class SpecialTokenFilter:
    '''
    remove special token from the speech transcripts.
    '''

    def __init__(self, key='label', filter_list=None):
        '''init.
        Args:
            filter_list(list): list of special tokens to ignore
        '''
        self.key = key
        self.filter_list = filter_list

    def __call__(self, item_data, **_kwargs):
        '''ignore examples for label sequence containing special tokens in filter_list.
        Args:
            item_data(dict): input data.

        Returns:
            dict: item_data with clean label.
            None, otherwise.
        '''
        if item_data is None:
            return None
        label = item_data.get(self.key, None)
        if label is None:
            return None
        if self.filter_list is None:
            return item_data

        if isinstance(label, str):
            label = label.split()
        for token in label:
            if token.lower() in self.filter_list:
                return None
        return item_data


# TODO: we will rename this class name to a more appropriate one in the feature
@PREPROCESS.register_module()
class EerFilter:
    '''filter for eer'''

    def __call__(self, item_data, **kwargs):
        '''call'''
        item_data['utt'] = kwargs.get('keys')
        if item_data['utt'] in ('meta',):
            return None
        return item_data


@PREPROCESS.register_module()
class SilenceRatioFilter:
    '''filter the data whose silence ratio is too low or high'''

    def __init__(self, key='bi_vad_label', min_ratio=0.3, max_ratio=0.7):
        '''init'''
        self.key = key
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, item_data, **_kwargs):
        '''do the filter.

        Args:
            item_data(dict): must contains vad label.

        Returns
            dict: item_data if the item_data meet conditions, None, otherwise.
        '''

        if item_data is None or self.key not in item_data:
            return None

        label = item_data.get(self.key, None)
        frames_num = len(label)
        if frames_num < 1:
            return None

        silence_ratio = np.sum(label == 0) / frames_num
        if silence_ratio < self.min_ratio or silence_ratio > self.max_ratio:
            return None
        return item_data


@PREPROCESS.register_module()
class ValidSpeakersFilter:
    '''valid speakers filter'''

    def __init__(self, key='speakers', embedding_keys=None, use_same_utt_embedding=True):
        '''init'''
        self.key = key
        self.get_embedding_keys(embedding_keys, use_same_utt_embedding)
        self.use_same_utt_embedding = use_same_utt_embedding

    def get_embedding_keys(self, embedding_keys, use_same_utt_embedding):
        """get embedding keys"""
        self.embedding_keys = set()
        for key in embedding_keys:
            if use_same_utt_embedding:
                self.embedding_keys.add(key.rsplit('-', 1)[0])  # utt-spk
            else:
                self.embedding_keys.add(key.rsplit("-", 2)[1])  # spk

    def __call__(self, item_data, **_kwargs):
        '''do the filter.

        Args:
            item_data(dict): must contains speakers label.

        Returns
            dict: item_data if the item_data meet conditions, None, otherwise.
        '''

        if item_data is None or self.key not in item_data:
            return item_data
        if not self.embedding_keys:
            return item_data

        speakers = item_data.get(self.key, None)
        if isinstance(speakers, str):
            speakers = [speakers]
        for speaker in speakers:
            spk = speaker if self.use_same_utt_embedding else speaker.rsplit("-", 1)[1]
            if spk not in self.embedding_keys:
                return None
        return item_data


@PREPROCESS.register_module()
class WavDurationFilter:
    '''filter too long or too short wavs before computing Fbank'''

    def __init__(
        self,
        key='waveform',
        sample_rate=16000,
        frame_length=25,  # second
        min_duration=0.01,  # second
        max_duration=30.0,  # second
    ):
        '''init'''
        self.key = key
        self.min_len = int(sample_rate * min_duration)
        self.max_len = int(sample_rate * max_duration)
        # window_size for FFT
        window_size = int(sample_rate * frame_length * MILLISECONDS_TO_SECONDS)
        self.min_len = max(self.min_len, window_size)

    def __call__(self, item_data, **_kwargs):
        '''do the filter.'''
        if item_data is None or self.key not in item_data:
            return item_data
        num_sample_points = item_data[self.key].shape[1]
        if not self.min_len <= num_sample_points <= self.max_len:
            return None
        return item_data
