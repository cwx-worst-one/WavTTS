import os
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from hyperpyyaml import load_hyperpyyaml

from recipes.umm.requires.model_initializer import ensure_hdfs_ckpt_is_local


__author__ = ["chaonan99"]


def return_self(x):
    return x


class EmptyDataset(Dataset):
    def __len__(self):
        return 0  # No data
    def __getitem__(self, idx):
        raise IndexError("Empty dataset has no items.")  # Prevent access


class EmptyDataModule(pl.LightningModule):
    def predict_dataloader(self):
        return DataLoader(EmptyDataset())


class FirstLevelKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # Only parse the first level of keys
        return {key: None for key, _ in node.value}


def load_hyperpyyaml_partial(yaml_stream, keys=[]):
    """Only load specific first-level keys and skipping others
    """
    raw_yaml = yaml.load(yaml_stream, Loader=FirstLevelKeyLoader)
    overrides = {k.value: None for k in raw_yaml.keys() if k.value not in keys}
    yaml_stream.seek(0)
    return load_hyperpyyaml(yaml_stream, overrides=overrides)


class HdfsFileWrapper(str):
    """The wrapper provide __fspath__ interface, so that it can be opened
    using `with open(file_wrapper, "r")` just like a local file path, but
    handle download (if file_path is an hdfs path) and open at the background.
    """
    def __new__(cls, file_path, local_cache_dir):
        os.makedirs(local_cache_dir, exist_ok=True)
        path = ensure_hdfs_ckpt_is_local(file_path, local_cache_dir)
        instance = super().__new__(cls, path)
        return instance