import pickle

import pytest
from torch.utils.data import IterableDataset

from samantha.dataio.dataset import MultiIterableDataset


class CountingIterableDataset(IterableDataset):
    def __init__(self, start, stop):
        super().__init__()
        self.start = start
        self.stop = stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self):
        return self.stop - self.start


@pytest.fixture
def data():
    return {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }


@pytest.fixture
def rawdata():
    keys = ["rid1.wav", "rid2.wav", "rid3.wav", "rid4.wav"]
    values = ["rid1", "rid2", "rid3", "rid4"]
    values = [pickle.dumps(v) for v in values]
    return keys, values


@pytest.fixture
def metadata():
    return {
        "rid1": {"label": "speaker1", "wav_file": "rid1.wav"},
        "rid2": {"label": "speaker1", "wav_file": "rid2.wav"},
        "rid3": {"label": "speaker2", "wav_file": "rid3.wav"},
        "rid4": {"label": "speaker2", "wav_file": "rid4.wav"},
    }


def test_sample_multi_iterable_dataset_equal_prob():
    DS_1_START = 0
    DS_1_STOP = 1
    DS_2_START = 1
    DS_2_STOP = 3
    DS_3_START = 3
    DS_3_STOP = 6

    ds_1 = CountingIterableDataset(DS_1_START, DS_1_STOP)
    ds_2 = CountingIterableDataset(DS_2_START, DS_2_STOP)
    ds_3 = CountingIterableDataset(DS_3_START, DS_3_STOP)

    total = DS_1_STOP - DS_1_START + DS_2_STOP - DS_2_START + DS_3_STOP - DS_3_START

    multi_dataset = MultiIterableDataset(
        datasets=[ds_1, ds_2, ds_3], num_samples=total * 4
    )
    multi_dataset_iter = iter(multi_dataset)

    with pytest.raises(StopIteration):
        for _ in range(total * 4 + 1):
            next(multi_dataset_iter)


def test_sample_multi_iterable_dataset_prob():
    DS_0_PROB = 0.8
    DS_1_PROB = 0.1
    DS_2_PROB = 0.1

    DS_0_START = 0
    DS_0_STOP = 1
    DS_1_START = 1
    DS_1_STOP = 3
    DS_2_START = 3
    DS_2_STOP = 6

    ds_0 = CountingIterableDataset(DS_0_START, DS_0_STOP)
    ds_1 = CountingIterableDataset(DS_1_START, DS_1_STOP)
    ds_2 = CountingIterableDataset(DS_2_START, DS_2_STOP)

    total = 20

    multi_dataset = MultiIterableDataset(
        datasets=[ds_0, ds_1, ds_2],
        num_samples=total,
        weights=[DS_0_PROB, DS_1_PROB, DS_2_PROB],
    )
    multi_dataset_iter = iter(multi_dataset)

    with pytest.raises(StopIteration):
        for _ in range(total + 1):
            next(multi_dataset_iter)
