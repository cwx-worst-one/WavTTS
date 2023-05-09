import os

import pytest
import webdataset as wds
from torch.utils.data import IterableDataset

from samantha.dataio.webdataset import WebPipeline


@pytest.fixture
def webdataset(global_datadir):
    return wds.WebDataset(os.path.join(global_datadir, "webdataset/compressed.tar"))


class DummyIter(IterableDataset):
    def __init__(self, n):
        self.data = [
            {"audio.npy": f"audio_{i:02d}.npy", "label.txt": f"label_{i:02d}.txt"}
            for i in range(n)
        ]

    def __iter__(self):
        return iter(self.data)


def transform_fn(data):
    for sample in data:
        sample["audio"] = sample["audio.npy"][5:-4]
        yield sample


def process_fn(sample):
    sample["label"] = sample["label.txt"][5:-4]
    return sample


def test_web_pipeline_with_wds(global_datadir):
    dataset = wds.WebDataset(os.path.join(global_datadir, "webdataset/compressed.tar"))
    pipeline = WebPipeline(dataset, ["decode"])
    sample = next(iter(pipeline))
    assert sample["__key__"] == "compressed/0001"
    assert sample["txt.gz"] == "hello\n"


def test_web_pipeline_with_iterable_dataset():
    dataset = DummyIter(100)
    pipeline = WebPipeline(
        dataset,
        [
            {"compose": transform_fn},
            {"map": process_fn},
            {"to_tuple": "audio label"},
            {"shuffle": [10]},
            {"batched": {"batchsize": 10}},
        ],
    )
    count = 0
    for batch in pipeline:
        assert len(batch) == 2
        assert len(batch[0]) == 10
        assert len(batch[1]) == 10
        assert batch[0] == batch[1]
        count += 1
    assert count == 10


def test_web_pipeline_with_none_wds_method(webdataset):
    with pytest.raises(ValueError):
        WebPipeline(webdataset, ["invalid_method"])


def test_web_pipeline_with_invalid_stage(webdataset):
    with pytest.raises(ValueError):
        WebPipeline(webdataset, [["decode"]])
