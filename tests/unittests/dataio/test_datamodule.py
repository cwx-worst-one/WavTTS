from samantha.dataio.datamodule import UniDataModule, ProcessorBase


class Processor(ProcessorBase):
    def process_one(self, sample):
        return sample


def test_uni_datamodule():
    path = ["hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/ci/00000.tar"]
    batcher_config = dict(buckets=None, dynamic_batch=False, maximum_bucket_size=10_000, batch_size=4, length_fn=lambda x: 1)
    processor = Processor(batcher_config=batcher_config)
    num_workers = 4 if len(path) > 4 else len(path) # num workers must less than number of urls
    dm = UniDataModule(
        processor=processor,
        train_data_path=path,
        train_num_workers=num_workers,
    )
    keys = set()
    for_loop = 0
    for item in dm.train_dataloader():
        for_loop += 1
        for e in item:
            keys.add(e["__key__"])
    assert for_loop == 1024
    assert len(keys) == 4096


def test_uni_datamodule_with_data_id():
    batcher_config = dict(buckets=None, dynamic_batch=False, maximum_bucket_size=10_000, batch_size=4, length_fn=lambda x: 1)
    processor = Processor(batcher_config=batcher_config)
    dm = UniDataModule(
        processor=processor,
        train_data_id=23,
        train_num_workers=1,
    )
    assert len(dm.hparams.train_data_path) == 1
    keys = set()
    for_loop = 0
    for item in dm.train_dataloader():
        for_loop += 1
        for e in item:
            keys.add(e["__key__"])
    assert for_loop == 1024
    assert len(keys) == 4096
