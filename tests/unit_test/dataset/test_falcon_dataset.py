"""
an example you can test FalconDataset
which used new FalconReader, can support read tensorbundle
and tfrecord files,can support global_shuffle and shuffle_in_files
"""
import os
from core.utils import Config, logging, get_logger
from core.dataset import build_item_augmentation, build_draw_batch_fn, get_meta
from torch.utils.data import DataLoader, dataloader
from core.dataset.falcon_dataset import FalconDataset
from core.dataset.mix_dataloader import MixedDataLoader, MixedHDFSDataset


def get_tf_data_path():
    '''
    get tfrecord data path
    '''
    data_root = (
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/data/'
        'training_platform/lingvo/input/LibriSpeech/'
    )
    data_path = (
        ["train-clean-100/tfrecord/tfrecoerd-{}".format(i) for i in range(20)]
        + ["train-clean-360/tfrecord/tfrecoerd-{}".format(i) for i in range(72)]
        + ["train-other-500/tfrecord/tfrecoerd-{}".format(i) for i in range(100)]
    )
    return ['{}{}'.format(data_root, path) for path in data_path]


def get_tb_data_path():
    '''
    get tensorbundle data path
    '''
    data_root = (
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
        'datasets/dolphin/librispeech_wav/'
    )
    data_path = ["{}train_sub{}".format(data_root, i) for i in range(16)]
    return data_path


def get_falcon_dataset():
    ''' '''
    get_logger(log_level='INFO')
    # clear cache
    os.system('rm -rf /tmp/falconreader')
    os.system('rm -rf /dev/shm/falconreader_*')

    # get data_root here
    data_path = get_tb_data_path()
    meta_file = (
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
        'datasets/dolphin/librispeech_wav/meta'
    )
    meta_data = get_meta(meta_file)

    item_transform_cfg = [
        dict(type='PickleParser'),
        dict(type='ProtoParser'),
        dict(type='DecordRaw', key2type=dict(frames='int16', transcript='bytes', uttid='bytes')),
        dict(type='LabelParser', in_key='transcript'),
        dict(type='WavConvert', in_key='frames'),
        dict(type='WavParser'),
        dict(type='SpeedPerturbation', p=0.666, speed_rate_list=[0.95, 1.05]),
        dict(type='KaldiFbank', dither=1.0, dynamic_dither=True),
        dict(type='LabelLengthFilter', in_out_ratio=4),
        dict(type='BPE', use_eos=False),
        dict(type='CMVN', key='fbank'),
        dict(
            type='DynamicTimeMask',
            time_mask_size=20,
            time_mask_block=60,
            replace_with_zero=False,
            inplace=True,
            time_mask_num_dither=1,
        ),
        dict(
            type='FreqMask',
            freq_mask_size=27,
            freq_mask_num=1,
            inplace=True,
            replace_with_zero=False,
        ),
    ]
    batch_transform_cfg = [
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate', args=dict()),
    ]

    item_transform = build_item_augmentation(item_transform_cfg, meta_data)
    batch_transforms = build_draw_batch_fn(batch_transform_cfg, meta_data)
    bucket_schedule = '50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000'
    cfg = Config(
        {
            'chunk_size': 20,
            'bucket_schedule_key': 'fbank',
            'batch_means_tokens': 1,
            'max_batch_size': 12360,
            'shuffle': True,
            'global_shuffle': False,
            'drop_last': False,
            'prefetch_worker_num': 3,
        }
    )
    dataset = FalconDataset(data_path, cfg=cfg, item_transform=item_transform)
    return dataset


def test_hdfs_dataset():
    '''main function'''
    get_logger(log_level='INFO')
    # clear cache
    os.system('rm -rf /tmp/falconreader')
    os.system('rm -rf /dev/shm/falconreader_*')

    # get data_root here
    meta_file = (
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
        'datasets/dolphin/librispeech_wav/meta'
    )
    meta_data = get_meta(meta_file)

    batch_transform_cfg = [
        dict(type='ListCollate', key='uttid'),
        dict(type='FbankCollate', fbank_dim=80),
        dict(type='CharCollate'),
        dict(type='PreCharCollate', args=dict()),
    ]

    batch_transforms = build_draw_batch_fn(batch_transform_cfg, meta_data)
    bucket_schedule = '50,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,2000'
    cfg = Config(
        {
            'chunk_size': 20,
            'bucket_schedule_key': 'fbank',
            'batch_means_tokens': 1,
            'max_batch_size': 12360,
            'shuffle': True,
            'global_shuffle': False,
            'drop_last': False,
            'prefetch_worker_num': 3,
        }
    )
    dataset = get_falcon_dataset()
    dataloader = MixedDataLoader(
        dataset,
        bucket_schedule,
        cfg,
        batch_transforms
    ) 
    iter_num = 20
    for idx, data in enumerate(dataloader):
        if idx > iter_num:
            break
        print(idx, data.keys(), data['src'].shape, data['src'].device)
    logging.error('dataset end')

if __name__ == '__main__':
    test_hdfs_dataset()
