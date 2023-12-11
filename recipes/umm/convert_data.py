import os
import multiprocessing
from argparse import ArgumentParser
from typing import List

import numpy as np
from hyperpyyaml import load_hyperpyyaml
from webdataset.shardlists import SimpleShardList
from pytorch_lightning import  Trainer
from recipes.datasets.base import LightningDataModuleBase
from recipes.datasets.mcc.mix import ParquetDataModule
from recipes.umm.transforms.speech import SpeechTransform
from samantha.dataio.data_bucket import data_bucket
from samantha.utils.hdfs_helper import hdfs_getsize
from samantha.utils.hdfs_tools import hdfs_cp, hdfs_glob, hdfs_loadtxt, hdfs_open
from samantha.utils.hparams import DotDict
from samantha.utils.logger import RankedLogger
from samantha.models.base import LightningModuleBase

logger = RankedLogger(__name__)


def split_shard_urls_by_rank(all_urls: List[str], rank: int, world_size: int):
    urls_per_rank = np.array_split(all_urls, world_size)

    if len(urls_per_rank) > rank:
        node_urls = urls_per_rank[rank].tolist()
    else:
        node_urls = []

    logger.info("#######")
    logger.info(f"There are {len(all_urls)} shards in total")
    logger.info(f"This is rank {rank+1} out of {world_size} total ranks.")
    logger.info(
        f"This is rank will process {len(node_urls)} out of {len(all_urls)} shards."
    )
    logger.info("#######")
    return node_urls


def split_parquet_dataset_urls(dataset_split, node_rank: int, world_size: int):
    # TODO: This isn't very pretty.. but it works :)
    # Note that operations are done in-place (i.e., there is no deepcopy)
    for dataset in dataset_split.pipeline[0]._datasets:
        urls = dataset.pipeline[0].data_pipeline.pipeline[0].urls
        split_urls = split_shard_urls_by_rank(
            urls, node_rank, world_size
        )
        dataset.pipeline[0].data_pipeline.pipeline[0].urls = split_urls
    return dataset_split

def split_datamodule_urls(dataset_split, node_rank: int, world_size: int):
    dataset_split.pipeline[0].urls = split_shard_urls_by_rank(
        dataset_split.pipeline[0].urls, node_rank, world_size
    )
    return dataset_split

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--merge_url2index", action="store_true")
    args = parser.parse_args()

    hparams_file = args.conf
    if hparams_file.startswith("hdfs"):
        with hdfs_open(hparams_file, "r") as fin:
            hparams = load_hyperpyyaml(fin)
    else:
        with open(hparams_file, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin)

    node_rank = int(os.getenv("ARNOLD_ID", 0))
    world_size = int(os.getenv("ARNOLD_WORKER_NUM", 1))

    cfg = DotDict(hparams)
    pl_datamodule: LightningDataModuleBase = cfg.pl_datamodule

    root_dir = data_bucket("data/music/converted")
    logger.warning(f"Writing to HDFS directory: {root_dir}")

    # assert (
    #     type(pl_datamodule.train_dataset.pipeline[0]) == SimpleShardList
    # ), "You must set resampled=False"

    # if type(pl_datamodule) == ParquetDataModule:
    #     root_dir = os.path.join(root_dir, "parquet", str(pl_datamodule.data_id))
    #     # TODO: This isn't very pretty.. but it works :)
    #     pl_datamodule.train_dataset = split_parquet_dataset_urls(pl_datamodule.train_dataset, node_rank, world_size)
    # else:
    #     pl_datamodule.train_dataset = split_datamodule_urls(pl_datamodule.train_dataset, node_rank, world_size)
    #     pl_datamodule.validation_dataset = split_datamodule_urls(pl_datamodule.validation_dataset, node_rank, world_size)

    # these are mutable objects, so we don't have to reassign the list:
    for dataset in pl_datamodule.train_dataset.pipeline[0]._datasets:
        dataset.data_pipeline = split_datamodule_urls(dataset.data_pipeline, node_rank, world_size)
    
    # audio_transform: SpeechTransform = cfg.audio_transform

    pl_module: LightningModuleBase = cfg.pl_module
    trainer: Trainer = cfg.trainer

    trainer.strategy.connect(pl_module)
    trainer.strategy.setup(trainer)

    audio_transform: SpeechTransform = pl_module.audio_tokenizer.model.audio_transform
    audio_transform = audio_transform.cpu() # TODO allow for GPU processing

    if not args.merge_url2index:
        # pl_module.preprocess(pl_datamodule, root_dir, rank=node_rank)
        audio_transform.convert_data(pl_datamodule, root_dir, rank=node_rank)
    else:
        # root_path = pl_module.preprocess_save_fp(pl_datamodule, root_dir)
        root_path = audio_transform.save_fp(pl_datamodule, root_dir)
        for split in ["train", "validation", "test"]:
            merged_urls = []
            url_files = os.path.join(root_path, split, "*/url2index.txt")
            url_files = hdfs_glob(url_files)
            for f in url_files:
                urls = hdfs_loadtxt(f)
                
                for url in urls:
                    tar_file, index_file = url.split("\t")

                    if hdfs_getsize(tar_file) == 0:
                        logger.warning(f"Skipping empty tar file: {tar_file}")
                        continue
                    
                    merged_urls.append(url)

            with open("url2index.txt", "w") as f:
                f.write("\n".join(merged_urls))

            merged_url2index_fp = os.path.join(root_path, split, "url2index.txt")
            logger.info(f"Writing merged url2index to: {merged_url2index_fp}")

            hdfs_cp("url2index.txt", merged_url2index_fp, override=True)
            os.remove("url2index.txt")
