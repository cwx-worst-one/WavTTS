'''
Example Command:
        python3 tests/dev/data_preprocess.py \
        --config configs/asr_task_hdfs_fusion/las_asr_base.py
'''

import os
import sys
from argparse import ArgumentParser
import torch
from transformers import BertTokenizer as BertTokenizer_huggingface
from core.dataset import (
    DebugHDFSDataset,
    get_meta,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.utils import Config, logging, get_logger, distributed_init, dist_hdfs_get
from core.models.pretrained.bert_utils import build_bert_vocab

o_path = os.getcwd()
sys.path.append(o_path)


def _str2bool(bool_str):
    """str to bool"""
    return bool_str.lower() in ("true", "1")


def parse_args():
    '''parse args'''
    parser = ArgumentParser(description='ASR debug dataset config scripts')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--init-port', type=int, default=0, help='init port')
    parser.add_argument(
        '--nccl-block', type=_str2bool, default=True, help='nccl block in dist init'
    )
    return parser.parse_known_args()


def get_data_list(dataset_cfg):
    '''
    get valid_file_list,train_file_list,meta_file_list
    '''
    data_root = dataset_cfg.get("data_root", None)
    train_data_root = dataset_cfg.get("train_data_root", data_root)
    valid_data_root = dataset_cfg.get("valid_data_root", data_root)
    meta_data_root = dataset_cfg.get('meta_data_root', train_data_root)
    train_file_list = dataset_cfg.get("train_file_list", None)
    valid_file_list = dataset_cfg.get("valid_file_list", None)
    meta_file_list = dataset_cfg.get('meta_file', None)
    data_path_separator = dataset_cfg.get('data_path_separator', '')
    dataset_file_nums = []

    # multiple data inputs from the command cfg can be splited by data_path_separator
    if (
        data_path_separator != ''
        and not isinstance(train_data_root, list)
        and not isinstance(meta_file_list, list)
    ):
        train_data_root = train_data_root.split(data_path_separator)
        train_file_list = train_file_list.split(data_path_separator)
        meta_file_list = meta_file_list.split(data_path_separator)

    # get train_file_list and eval_file_list
    if isinstance(train_data_root, str):
        train_data_root = [train_data_root]
        train_file_list = [train_file_list]
    all_train_file_list = []
    assert len(train_data_root) == len(train_file_list)
    for train_root, train_file in zip(train_data_root, train_file_list):
        train_dataset_now = [os.path.join(train_root, p) for p in eval(train_file)]
        all_train_file_list += train_dataset_now
        dataset_file_nums.append(len(train_dataset_now))

    dataset_cfg.dataset_length = dataset_file_nums

    # get valida_file_lists
    if isinstance(valid_data_root, str):
        valid_data_root = [valid_data_root]
        valid_file_list = [valid_file_list]
    all_valid_file_list = []
    assert len(valid_data_root) == len(valid_file_list)
    for valid_root, valid_file in zip(valid_data_root, valid_file_list):
        valid_dataset_now = [os.path.join(valid_root, p) for p in eval(valid_file)]
        all_valid_file_list += valid_dataset_now

    # get meta_file_lists
    if isinstance(meta_data_root, str):
        meta_data_root = [meta_data_root]
    if isinstance(meta_file_list, str) or not meta_file_list:
        meta_file_list = [meta_file_list]
    all_meta_file_list = []
    for meta_root, meta_file in zip(meta_data_root, meta_file_list):
        if meta_file:
            all_meta_file_list.append(os.path.join(meta_root, meta_file))
    return all_train_file_list, all_valid_file_list, all_meta_file_list


# pylint: disable='too-many-branches'
def setup_transform_cfg(config, item_transforms, batch_transforms, meta_data):
    '''setup  transform_cfg'''
    for cfg in item_transforms:
        if cfg.type == 'Textchar2Index':
            bert_vocab_file = config.solution.get('bert_vocab_dict')
            bert_filter_punc = config.solution.get('bert_filter_punc', False)
            bert_char2index = config.solution.get('bert_char2index', True)
            bert_text_lexicon, bert_punc_table = build_bert_vocab(
                bert_vocab_file, bert_char2index, bert_filter_punc
            )
            cfg['vocab_dict'] = bert_text_lexicon
            cfg['punc_table'] = bert_punc_table
            cfg['char2index'] = bert_char2index
        if cfg.type == 'DialogHistToContext':
            reorder_tgt_dict = meta_data['reorder_tgt_dict']
            cfg['vocab'] = reorder_tgt_dict.symbols
        elif cfg.type == 'BertTokenizer':
            bert_vocab_dict = config.solution.get('bert_vocab_dict', '')
            local_vocab_file = None
            if bert_vocab_dict:
                local_vocab_file = dist_hdfs_get(bert_vocab_dict, './tmp/', 'vocab_dict.txt')
            assert local_vocab_file is not None
            bert_tokenizer = BertTokenizer_huggingface(local_vocab_file)
            cfg['bert_tokenizer'] = bert_tokenizer
        if cfg.type == 'W2vPhone2charLabel':
            cfg['tgt_dict'] = meta_data['tgt_dict']
            # setup lexicon
            lexicon_file = os.path.join(
                config.data.data_root[0].replace('hdfs_data', ''), config.data.lexicon
            )
            dist_hdfs_get(lexicon_file)
            lexicon = dict()
            with open(config.data.lexicon, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split()
                    lexicon[line[0]] = line[1:]  # word to phones
            cfg['lexicon'] = lexicon
    for cfg in batch_transforms:
        if cfg.type == 'PreCharCollate':
            cfg['args'] = config.solution
        if cfg.type == 'CharCollate':
            cfg['tgt_dict'] = meta_data['tgt_dict']
        if cfg.type == 'ComputeW2vMask':
            cfg['mask_prob'] = config.solution.mask_prob
            cfg['mask_length'] = config.solution.mask_length
            cfg['use_fbank'] = config.solution.wav2vec_use_fbank
            cfg['conv_feature_layers'] = config.solution.get('conv_feature_layers', '')
            cfg['mask_type'] = config.solution.mask_selection
            cfg['mask_other'] = config.solution.mask_other
            cfg['mask_minlen_type'] = config.solution.mask_minlen_type
            cfg['no_overlap'] = config.solution.no_mask_overlap
            cfg['min_space'] = config.solution.mask_min_space
        if cfg.type == 'ContextMakePairsCollate':
            context_loss_scale = config.solution.get('context_loss_scale', 0)
            if context_loss_scale > 0:
                cfg['context_loss_scale'] = context_loss_scale

    print("out", batch_transforms)


def build_dataset(dataset_cfg):
    '''build dataset.'''
    train_file_list, _valid_file_list, meta_file_list = get_data_list(dataset_cfg)
    meta_data = {}
    if meta_file_list:
        meta_data = get_meta(meta_file_list[0])
    train_item_trans_cfg = dataset_cfg.get("train_item_transform", [])

    draw_batch_cfg = dataset_cfg.get("batch_transform", [])
    draw_train_batch_cfg = dataset_cfg.get('train_batch_transform', draw_batch_cfg)

    device_trans_cfg = dataset_cfg.get('device_transform', [])
    train_device_trans_cfg = dataset_cfg.get('train_device_transform', device_trans_cfg)

    return (
        train_file_list,
        train_item_trans_cfg,
        draw_train_batch_cfg,
        train_device_trans_cfg,
        meta_data,
    )


def build_dataloader(config, dataset_config, is_cuda_available):
    '''build dataloader'''
    (
        train_file_list,
        train_item_trans_cfg,
        draw_train_batch_cfg,
        train_device_trans_cfg,
        meta_data,
    ) = build_dataset(dataset_config)
    setup_transform_cfg(config, train_item_trans_cfg, draw_train_batch_cfg, meta_data)

    train_item_trans = build_item_augmentation(train_item_trans_cfg, meta_data)
    draw_train_batch_fn = build_draw_batch_fn(draw_train_batch_cfg, meta_data)
    if not is_cuda_available:
        train_device_trans = None
    else:
        train_device_trans = build_device_augmentation(train_device_trans_cfg, meta_data)

    dataloader = DebugHDFSDataset(
        path_list=train_file_list,
        bucket_schedule=dataset_config.get('bucket_schedule', ''),
        cfg=dataset_config,
        item_transform=train_item_trans,
        batch_transforms=draw_train_batch_fn,
        device_transforms=train_device_trans,
        split_path_list_by_rank=True,
        prefetch_block_num=200,
        epoch_count=0,
        shuffle=False,
        is_cuda_available=is_cuda_available,
    )
    return dataloader


def main():
    '''main functions'''
    args, unknown = parse_args()
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        # init distributed environment if necessary
        distributed_init(args.init_port, block=args.nccl_block)

    get_logger(log_level='INFO')

    cfg = Config.fromfile(args.config)
    cfg.merge_from_list(unknown)
    logging.info(cfg.filename + ':\n' + cfg.dump())
    dataloader = build_dataloader(cfg, cfg.data, is_cuda_available)
    dataloader.reset()
    for _ in range(2):
        data = dataloader.next()
        logging.info("batch_data keys: %s", data.keys())
    dataloader.terminate()
    logging.error('dataset end.')


if __name__ == '__main__':
    main()
