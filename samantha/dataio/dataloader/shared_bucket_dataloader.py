from torch.utils.data import DataLoader
from webdataset.filters import _shuffle

from samantha.utils.webdataset import return_self


class SharedBucketDataLoader(DataLoader):
    def __init__(self, bucket_batcher, shuffle_buffer_size, **kwargs):
        self.__collate_fn = kwargs.pop("collate_fn", return_self)
        kwargs["batch_size"] = None
        super().__init__(**kwargs)

        self.batcher = bucket_batcher
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        data_source = super().__iter__()
        for item in _shuffle(data_source, bufsize=self.shuffle_buffer_size):
            batch = self.batcher.collate_batch(item)
            if batch:
                yield self.__collate_fn(batch)

        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield self.__collate_fn(batch)
