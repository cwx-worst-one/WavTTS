"""
Dataset function and class from Dolphin.
"""

import multiprocessing as mp

from .hdfs_dataset import HDFSDataset, ValidHDFSDataset
from .mix_dataloader import MixedHDFSDataset, MixedValidHDFSDataset
from .debug_dataset import DebugHDFSDataset
from .balance_dataset import BalancedHDFSDataset
from .dictionary import Dictionary, ScpDictionary
from .preprocess import build_item_augmentation, build_draw_batch_fn, build_device_augmentation
from .get_meta import get_meta
from .data_fetcher import get_paths, DataFetcher, TargetDataFetcher


__all__ = [
    'HDFSDataset',
    'ValidHDFSDataset',
    'DebugHDFSDataset',
    'BalancedHDFSDataset',
    'Dictionary',
    'ScpDictionary',
    'build_item_augmentation',
    'build_draw_batch_fn',
    'build_device_augmentation',
    'get_meta',
    'get_paths',
    'DataFetcher',
    'TargetDataFetcher',
    'MixedHDFSDataset',
    'MixedValidHDFSDataset'
]

mp.set_start_method('spawn', force=True)
