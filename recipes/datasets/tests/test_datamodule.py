import random

import pytest
import torch
import torchaudio
from tqdm import tqdm

from samantha.dataio.batching import BucketBatcher


def generate_data(num_samples: int, max_seq_len: int):
    for _ in range(num_samples):
        yield {"audio": torch.randn(1, random.randint(1, max_seq_len))}


def test_bucket_batching():
    sample_rate = 24000
    max_data_size = sample_rate * 30  # 30 seconds

    batch_size = 8
    batcher = BucketBatcher(
        buckets=[sample_rate * 10, sample_rate * 20, sample_rate * 30],
        dynamic_batch=False,
        batch_size=batch_size,
        length_fn=lambda x: x["audio"].shape[-1],
    )

    for data in generate_data(num_samples=50, max_seq_len=max_data_size):
        batch = batcher.collate_batch(data)
        if batch:
            assert len(batch) == batch_size

            batch = speech_collate_fn(batch)
            audio = batch["audio"]
            # print((audio == 0).sum() / audio.numel())
            assert audio.shape[0] == batch_size
