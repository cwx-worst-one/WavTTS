import json
import os
import sys
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning import Trainer

from recipes.research.audio_codec.zoo import (
    AudioCodec_7c355ea_64l,
    AudioCodec_f81b3fa_64l,
)
from recipes.research.dataset.collection import (
    EveryNoisev1ParquetDataset,
    PlaylistV5_24khz_ParquetDataset,
    ShutterStockv1ParquetDataset,
    RadioheadParquetDataset
)
from recipes.research.dataset.featextract.audiostats import AudioStats
from recipes.research.mel_codec.mel_rq import MelRQ, MelRQConfig
from samantha.data.audio.dataset import AudioFolderDataModule
from samantha.data.audio.types import AudioDataResult, ShardInfo
from samantha.dataio.parquet.writer import ParquetWriter
from samantha.models.base import LightningModuleBase
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class ProcessModule(LightningModuleBase):

    def __init__(self, model: LightningModuleBase):
        super().__init__(ignore=["model"])
        self.writers = {}
        self.model = model

        self.model = self.model.eval()
        self.model.freeze()

    def get_parquet_writer(self, shard_target_url: str):
        if shard_target_url not in self.writers:
            fs = get_filesystem(shard_target_url)

            # if exists(shard_target_url):
            #     return None

            self.writers[shard_target_url] = ParquetWriter(
                shard_target_url,
                row_group_size=sys.maxsize,
                need_row_group_no=False,
                filesystem=fs,
                verbose=True,
            )
        return self.writers[shard_target_url]

    def predict_step(
        self, batch: AudioDataResult, batch_idx: int, dataloader_idx: int = 0
    ):
        assert len(batch.audio) == 1

        shard_info: ShardInfo = batch.shard_info[0]

        shard_url = Path(shard_info.url)
        shard_target_url = f"{TARGET_DIR}/{shard_url.parent.name}/{shard_url.name}"

        writer: Optional[ParquetWriter] = self.get_parquet_writer(shard_target_url)
        if writer is None:
            logger.info(f"Skipping {shard_target_url}")

        logger.debug(
            f"{shard_target_url, shard_info.row_group, shard_info.last_row_group, shard_info.group_index, shard_info.last_group}"
        )

        with torch.no_grad():
            audio = batch.audio[0:1]
            sample_rate = batch.segment_info[0].sample_rate
            audio = self.model.preprocess_audio(audio, sample_rate)
            feature = self.model.get_mel(audio)  # [B, F, T]


            audio_stats = AudioStats.measure(audio, sample_rate)
            audio_stats = audio_stats.to_dict()

            # feature = self.model.forward(
            #     audio, sample_rate, chunk_seconds=300
            # )

            feature = feature.squeeze(dim=0)

            buffer = BytesIO()
            torch.save(feature.cpu(), buffer)

            item = {"uttid": shard_info.uttid, "feature": buffer.getvalue(), "audiostats": json.dumps(audio_stats)}
            writer.write(item)

        if shard_info.group_index == shard_info.last_group:
            logger.debug(
                f"Writing row group: {shard_info.group_index}/{shard_info.last_group}"
            )
            row_group_size = shard_info.last_group + 1
            writer.write_row_group(row_group_size)

        if (
            shard_info.group_index == shard_info.last_group
            and shard_info.row_group == shard_info.last_row_group
        ):
            logger.info(f"WRITING SHARD: {shard_target_url}")
            writer.close()
            del self.writers[shard_target_url]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # TARGET_DIR = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_Ssstk-pond5_Mnonvocal_T44k_N1608k_AudioCodec_7c355ea_64l/data"
    # TARGET_DIR = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_everynoise_N937k_Lmix_mp3_AudioCodec_7c355ea_64l/data"
    TARGET_DIR = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/playlist_v5_2M_24khz_mel_160m_2048fft_240hl/data"

    # TARGET_DIR = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/music_artist_sft_radiohead_24khz_mel_160m_2048fft_240hl/data"

    sample_rate = 24000
    channels = 1
    # dataset = ShutterStockv1ParquetDataset(
    #     sample_rate=sample_rate,
    #     channels=channels,
    #     segment_duration=None,
    #     resampled=False,
    #     shardshuffle=False,
    #     crop_from_start=False,
    # )

    # dataset = EveryNoisev1ParquetDataset(
    #     sample_rate=sample_rate,
    #     channels=channels,
    #     segment_duration=None,
    #     resampled=False,
    #     shardshuffle=False,
    #     crop_from_start=False,
    # )

    dataset = PlaylistV5_24khz_ParquetDataset(
        sample_rate=sample_rate,
        channels=channels,
        segment_duration=None,
        resampled=False,
        shardshuffle=False,
        crop_from_start=False,   
    )

    # dataset = RadioheadParquetDataset(
    #     sample_rate=sample_rate,
    #     channels=channels,
    #     segment_duration=None,
    #     resampled=False,
    #     shardshuffle=False,
    #     crop_from_start=False,   
    # )

    pl_datamodule = AudioFolderDataModule(
        train_datasets=[dataset],
        validation_datasets=[],
        test_datasets=[],
        weights=None,
        batch_size=1,
        shuffle=None,
        num_workers=args.num_workers,
        batch_drop_duplicates=False,
        prefetch_factor=None,
    )
    dataloader = pl_datamodule.train_dataloader()

    writers = {}

    # model = AudioCodec_f81b3fa_64l()
    # model = AudioCodec_7c355ea_64l()

    model = MelRQ(MelRQConfig(
        sample_rate=24000,
        n_freq_bins=160,
        n_fft=2048,
        hop_length=240,
        f_max=None,
        n_layer=2,
        n_embd=128,
        n_head=2,
    ))

    pl_module = ProcessModule(model)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=int(os.getenv("ARNOLD_WORKER_NUM", 1)),
        precision="32",
        callbacks=[],
        max_epochs=1,
    )
    trainer.predict(pl_module, dataloader)
