import pickle

import pytest
import torch.utils.data
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


def item_transform(item, **_):
    item = pickle.loads(item)
    return item["audio.npy"], item["meta.json"]


@pytest.mark.disable
def test_mix_read_wds_kv():
    from hyperpyyaml import load_hyperpyyaml

    config_str = """
    wds_dataset: !new:samantha.dataio.webdataset.WebPipeline
        dataset: !new:webdataset.WebDataset
            urls: "pipe: hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/jingsong.gao/wds/audioset_{0000..0109}.tar"
        pipeline:
            - decode
            - to_tuple: "audio.npy meta.json"

    kv_dataset: !new:core.dataset.falcon_dataset.FalconDataset
        path_list:
          - hdfs://harunava/home/byte_speech_sv/jingsong.gao/kv/audio_set
        cfg:
            chunk_size: 20
            io_thread_num: 12
            max_batch_size: 12360
            shuffle: False
        item_transform: !name:tests.unittests.dataio.test_dataset.item_transform

    dataset: !new:samantha.dataio.dataset.MultiIterableDataset
        datasets:
          - !ref <wds_dataset>
          - !ref <kv_dataset>
        num_samples: 1024

    dataloader: !new:torch.utils.data.DataLoader
        dataset: !ref <dataset>
        batch_size: 4
        num_workers: 8
    """
    cfg = load_hyperpyyaml(config_str)
    dataloader = cfg["dataloader"]
    for idx, item in enumerate(dataloader):
        assert len(item[0]) == 4
        assert "tags" in item[1]
        assert "data_source" in item[1]
        assert "music_id" in item[1]
        assert len(item[1]["tags"]) == 4
        assert len(item[1]["data_source"]) == 4
        assert len(item[1]["music_id"]) == 4

def test_draw_num_samples_from_multi_iterable_dataset():
    ds_1 = CountingIterableDataset(0, 1024)
    ds_2 = CountingIterableDataset(1024, 2048)

    num_samples = 1024
    batch_size = 4
    num_workers = 8
    dataset = MultiIterableDataset([ds_1, ds_2], num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, num_workers=num_workers, batch_size=batch_size)

    total_samples, total_batches = 0, 0
    for item in dataloader:
        total_batches += 1
        total_samples += len(item)

    assert total_samples == num_samples
    assert total_batches == num_samples // batch_size
