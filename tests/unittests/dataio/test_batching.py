import torch
from samantha.dataio.batching import BucketBatcher


def generate_data(num_samples):
    for _ in range(num_samples):
        yield torch.randn((torch.randint(1, 1000, (1,))))


def test_collate_data_dynamic():

    maximum_bucket_size = 20_000
    bucket = BucketBatcher(
        buckets=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        dynamic_batch=True,
        maximum_bucket_size=maximum_bucket_size,
        batch_size=None,
        length_fn=lambda x: x.shape[0],
    )

    for data in generate_data(10000):
        batch = bucket.collate_batch(data)
        if batch:
            shapes = [x.shape[0] for x in batch]
            assert len(batch) * max(shapes) <= maximum_bucket_size

    for batch in bucket.collect_last_batch():
        shapes = [x.shape[0] for x in batch]
        assert len(batch) * max(shapes) <= maximum_bucket_size

def test_collate_data_fixed():

    batch_size = 128
    bucket = BucketBatcher(
        buckets=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        dynamic_batch=False,
        maximum_bucket_size=None,
        batch_size=batch_size,
        length_fn=lambda x: x.shape[0],
    )

    for data in generate_data(10000):
        batch = bucket.collate_batch(data)
        if batch:
            assert len(batch) == batch_size

    for batch in bucket.collect_last_batch():
        assert len(batch) <= batch_size
