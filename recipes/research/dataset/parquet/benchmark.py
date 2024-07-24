from time import perf_counter
from typing import Dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import AdvancedProfiler
from tqdm import tqdm
from webdataset.utils import pytorch_worker_info

from recipes.research.dataset.collection import (
    Billboardv2WebDataset,
    EveryNoiseParquetDataset,
    MultiLingualBigASR48LangParquetDataset,
    MusDBTrainAudioFolderDataset,
    PlaylistV5ParquetDataset,
    ShutterStockFeatureParquetDataset,
    ShutterStockParquetDataset,
)
from recipes.research.dataset.data_monitor import DataMonitor
from recipes.research.dataset.monitor.memory import start_memory_monitor
from recipes.research.dataset.parquet.local_indexes import (
    everynoise_data_urls,
    playlist_v5_data_urls,
)
from recipes.research.dataset.parquet_dataset import AudioParquetDataset
from samantha.data.audio.dataset import AudioFolderDataModule
from samantha.data.audio.webdataset import AudioWebDataset
from samantha.models.base import DefaultTrainingBaseModule, TrainingResultBase


class DummyModule(DefaultTrainingBaseModule):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

    def step(self, batch: Dict[str, torch.Tensor], batch_idx: int, return_loss: bool):
        return TrainingResultBase(
            loss={"loss": torch.tensor([0.0], requires_grad=True)}
        )


if __name__ == "__main__":

    devices = 8
    batch_size = 4
    sample_rate = 44100
    channels = 2
    segment_duration = 30
    num_workers = 8

    playlist = PlaylistV5ParquetDataset(
        sample_rate=sample_rate,
        channels=channels,
        segment_duration=segment_duration,
        resampled=True,
        shardshuffle=True,
    )

    shutterstock = ShutterStockParquetDataset(
        sample_rate=sample_rate,
        channels=channels,
        segment_duration=segment_duration,
        resampled=True,
        shardshuffle=True,
    )

    everynoise = EveryNoiseParquetDataset(
        sample_rate=sample_rate,
        channels=channels,
        segment_duration=segment_duration,
        resampled=True,
        shardshuffle=True,
    )

    billboard_v2 = Billboardv2WebDataset(
        sample_rate=sample_rate,
        channels=channels,
        segment_duration=segment_duration,
        resampled=True,
        shardshuffle=True,
    )

    musdb_train = MusDBTrainAudioFolderDataset(
        sample_rate=sample_rate, channels=channels, segment_duration=segment_duration
    )

    # speech = MultiLingualBigASR48LangParquetDataset(
    #     sample_rate=24000,
    #     channels=1,
    #     resampled=True,
    #     shardshuffle=True,
    #     buckets_sec=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    #     min_audio_duration=1.0,
    #     max_audio_duration=30.0,
    #     batch_size=batch_size,
    # )

    # billboard_v2 = AudioWebDataset(
    #     url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/shards/billboard_hot_200-v2_normalised_-16LUFS/*/url2index.txt",
    #     data_type="music_vocal",
    #     sample_rate=44100,
    #     channels=2,
    #     pad=True,
    #     segment_duration=30,
    #     n_segments_per_read=1,
    #     shuffle_buffer_size=0,
    #     resampled=True,
    #     shardshuffle=True,
    # )

    # import pickle
    # item = next(iter(everynoise))
    # item = pickle.loads(pickle.dumps(item))

    # import msgpack
    # item = next(iter(everynoise))
    # item = msgpack.dumps(item.audio.cpu())
    # start_memory_monitor(shutterstock, n_workers=num_workers, n_seconds=1000)

    datamodule = AudioFolderDataModule(
        [playlist, shutterstock, everynoise, billboard_v2, musdb_train],
        [],
        [],
        weights=None,
        batch_size=batch_size,
        shuffle=None,
        num_workers=num_workers,
        prefetch_factor=2,
        batch_drop_duplicates=True,
    )

    # total_steps = 10000
    # tik = perf_counter()
    # rank, world_size, worker, num_workers = pytorch_worker_info(group=None)
    # dataloader = datamodule.train_dataloader()
    # for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"DataLoader: {rank=}/{world_size=}")):
    #     if batch_idx == total_steps:
    #         break

    #     # print(f"{rank=}, {world_size=}", batch.audio.shape)

    # tok = perf_counter()
    # print(f"Benchmark | duration: {(tok - tik)} | it/sec: {total_steps / (tok - tik)}")

    ## validate distributed setting:

    monitor = DataMonitor()
    module = DummyModule()
    module.setup(stage="fit")
    trainer = Trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=1,
        strategy="ddp_find_unused_parameters_false",
        use_distributed_sampler=False,  # NOTE: use this?
        callbacks=[monitor],
    )
    # trainer.profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer.fit(module, datamodule=datamodule)
