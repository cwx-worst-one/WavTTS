import random

import torchaudio
from tqdm import tqdm

from recipes.datasets.libritts import LibriTTSWebDataModule


def test_libritts_datamodule():
    batch_size = 8
    sample_rate = 24000
    pl_datamodule = LibriTTSWebDataModule(
        sample_rate=sample_rate,
        batch_size=batch_size,
        shuffle_buffer_size=100,
        buckets_sec=[2, 5],
        use_bucket_batcher=True,
        # duration=10,
    )
    train_loader = pl_datamodule.train_dataloader()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        assert "audio" in batch
        if batch_idx > 5:
            break

        rand_idx = random.choices(range(batch_size), k=4)
        audio = batch.audio

        # assert audio.shape == (batch_size, 1, pl_datamodule.n_audio_samples)

        audio = audio[rand_idx]
        for a_idx, a in enumerate(audio):
            torchaudio.save(
                f"libritts-{batch_idx}-{a_idx}-test.mp3", a, pl_datamodule.sample_rate
            )


# def test_libritts_bucket_datamodule():
#     batch_size = 8
#     sample_rate = 24000

#     buckets_sec = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
#     buckets_samples = list(map(lambda i: i * sample_rate, buckets_sec))

#     batcher = BucketBatcher(
#         buckets=buckets_samples,
#         dynamic_batch=False,
#         batch_size=batch_size,
#         length_fn=lambda x: x["audio"].shape[-1],
#     )
#     max_bucket_size = batcher.buckets[-1]

#     pl_datamodule = LibriTTSWebDataModule(
#         sample_rate=sample_rate,
#         batch_size=batch_size,
#         shuffle_buffer_size=100,
#         batcher=batcher,
#         collate_fn=speech_collate_fn,
#         num_workers=0,
#     )
#     train_loader = pl_datamodule.train_dataloader()
#     for batch_idx, batch in enumerate(tqdm(train_loader)):
#         assert "audio" in batch

#         if batch_idx > 10:
#             break

#         audio = batch["audio"]
#         assert audio.shape[0] == batch_size
#         assert audio.shape[2] <= max_bucket_size

#         for a_idx, a in enumerate(audio):
#             torchaudio.save(
#                 f"libritts-{batch_idx}-{a_idx}-test.mp3", a, pl_datamodule.sample_rate
#             )
