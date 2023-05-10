# pylint: disable=all
# TODO(zhengyijie) del this pylint disable
''' test sid balance dataset. '''
import random
import pickle
import os
import numpy as np
import pytest
from core.utils import Config, distributed_init
from core.dataset import build_item_augmentation, build_draw_batch_fn
from core.dataset import BalancedHDFSDataset, ValidHDFSDataset


@pytest.mark.isolate
def _test_valid_hdfs_dataset():
    '''Test hdfs balance Dataset.'''
    distributed_init()

    path = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_test/test_data/sub0'
    meta = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_test/test_data/meta'
    item_transform = build_item_augmentation(
        [
            dict(type='PickleParser'),
            dict(type='LilcomDecompress', in_key='src', out_key='src'),
            # dict(type='DictTrans', in_key='spk', out_key='spk', dicts=spk2idx),
            dict(type='DictTrans', in_key='spk', out_key='spk', meta=meta, meta_key='spk2idx'),
            dict(type='SplitFeature', splited_len=100, in_key='src', out_key='src'),
        ]
    )
    batch_transforms = build_draw_batch_fn(
        [dict(type='SidCollate', min_len=100, max_len=200, random_clip=False)]
    )
    cfg = Config(
        {
            'max_batch_size': 10,
            'batch_means_tokens': False,
        }
    )

    dataset = ValidHDFSDataset(
        [path], '', cfg, item_transform, batch_transforms, ignored_keys_in_data=['meta']
    )
    dataset.reset()
    batch_data = dataset.next()
    while batch_data is not None:
        batch_data = dataset.next()
        # print('data from ValidHDFSDataset', data['features'].shape)
    print("Finish feature reading.")
    dataset.terminate()
    # os.system('rm -rf {}*'.format(path))


# TODO(liyong): pytest doesn't support multiprocessing well.
def _test_balance_hdfs_dataset():
    '''Test hdfs balance Dataset.'''
    distributed_init()

    # Load data and meta from a hdfs dataset.
    path_list = [
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_test/test_data/sub0'
    ]
    meta = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_test/test_data/meta'
    item_transform = build_item_augmentation(
        [
            dict(type='PickleParser'),
            dict(type='LilcomDecompress', in_key='src', out_key='src'),
            dict(type='DictTrans', in_key='spk', out_key='spk', meta=meta, meta_key='spk2idx'),
            # dict(type='DictTrans', in_key='spk', out_key='spk', dicts=spk2idx),
        ]
    )
    batch_transforms = build_draw_batch_fn([dict(type='SidCollate', min_len=100, max_len=200)])
    cfg = Config(
        {
            'batch_size_per_class': 4,
            'batch_class_num': 4,
        }
    )

    dataset = BalancedHDFSDataset(path_list, cfg, item_transform, batch_transforms)
    dataset.reset()
    for _ in range(1):
        data = dataset.next()
        print('data from BalancedHDFSDataset', data['feature'].shape)
    dataset.terminate()
    # os.system('rm -rf {}*'.format(path))


if __name__ == '__main__':
    _test_valid_hdfs_dataset()
    _test_balance_hdfs_dataset()
