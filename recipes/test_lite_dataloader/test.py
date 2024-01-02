import logging
from typing import List

from hyperpyyaml import load_hyperpyyaml

from samantha.dataio.lite.transform import CollatorBase, ItemTransformBase

logger = logging.getLogger(__name__)


class FakeDataset:
    def __init__(self, item_transform):
        self.item_transform = item_transform

    def __iter__(self):
        for source in ["A", "B", "C", "D"]:
            item = {"source": source}
            yield from self.item_transform(item)


class FakeItemTransformA(ItemTransformBase):
    def __call__(self, item):
        if item is None:
            item = {"FakeItemTransformA": True}
        else:
            item |= {"FakeItemTransformA": True}
        return item


class FakeItemTransformB(ItemTransformBase):
    def __call__(self, item):
        if item is None:
            item = {"FakeItemTransformB": True}
        else:
            item |= {"FakeItemTransformB": True}
        return item


class FakeBatchItemTransformA(CollatorBase):
    def __call__(self, batch):
        for item in batch:
            if item is None:
                item = {"FakeBatchItemTransformA": True}
            else:
                item |= {"FakeBatchItemTransformA": True}
        return batch


if __name__ == "__main__":
    with open("samantha/recipes/test_lite_dataloader/test.yaml", "r") as f:
        cfg = load_hyperpyyaml(f)

    dm = cfg["pl_datamodule"]
    for item in dm.train_dataloader():
        print(item)
