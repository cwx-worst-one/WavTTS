''' package meta file '''

import os
import sys
import pickle
import numpy as np
from falconpai import FalconWriter
# ScpDictionary is imported from dolphin
# make sure dolphin in PYTHONPATH
from core.dataset import ScpDictionary


def reorder_dict(meta_dir, bpe_file_path, tgt_dict):
    ''' reorder_dict '''
    # this is used for reorder the dict by freq
    # which is critical for addaptive softmax
    out_dict = {}
    with open(bpe_file_path) as in_dict_f:
        for line in in_dict_f:
            kv_items = line.strip().split()
            out_dict[kv_items[0]] = int(kv_items[1])

    sorted_out_dict = dict(sorted(out_dict.items(),
                                  key=lambda item: item[1],
                                  reverse=True))
    with open(meta_dir + '/reorder.dict', 'w') as out_dict_f:
        for k, v in sorted_out_dict.items():
            out_dict_f.write('{} {}\n'.format(k, v))

    reorder_tgt_dict = ScpDictionary.load(meta_dir + '/reorder.dict')
    reorder_dict_map = {}
    for k, v in sorted_out_dict.items():
        ori_idx = tgt_dict.index(k)
        reorder_idx = reorder_tgt_dict.index(k)
        reorder_dict_map[ori_idx] = reorder_idx
    for token in ['<s>', '</s>', '<pad>', '<unk>']:
        reorder_dict_map[tgt_dict.index(token)] = \
                reorder_tgt_dict.index(token)
    return reorder_tgt_dict, reorder_dict_map


def prepare_meta_data(meta_dir):
    meta_data = {}

    # get cmvn
    meta_data['cmvn_mean'] = np.loadtxt(meta_dir + '/cmvn_mean')
    meta_data['cmvn_var'] = np.loadtxt(meta_dir + '/cmvn_var')

    # get bpe
    meta_data['total.code'] = open(meta_dir + '/bpe_code.txt').read()

    # get tgt_dict
    vocab_dict = meta_dir + '/vocab_dict.txt'
    tgt_dict = ScpDictionary.load(vocab_dict)
    reorder_tgt_dict, reorder_dict_map = reorder_dict(meta_dir, vocab_dict, tgt_dict)
    meta_data['tgt_dict'] = tgt_dict
    meta_data['reorder_tgt_dict'] = reorder_tgt_dict
    meta_data['reorder_dict_map'] = reorder_dict_map

    return meta_data


def main():
    if len(sys.argv) < 3:
        print("Usage:\n\t python3 %s local_meta_dir meta_hdfs_path \n" % sys.argv[0])
        print("Expected files in local_meta_dir/:")
        print("\tcmvn_mean & cmvn_var: the vectors of mean and var to be loaded by numpy.loadtxt()")
        print("\tbpe_code.txt & vocab_dict.txt: produced by subword-nmt, e.g: `subword-nmt learn-joint-bpe-and-vocab --input corpus.txt -s 7200 --output bpe_code.txt --write-vocabulary vocab_dict.txt`")
        exit(1)

    meta_dir = sys.argv[1]
    meta_hdfs_path = sys.argv[2]

    meta_data = prepare_meta_data(meta_dir)
    # keys of meta_data: ['cmvn_mean', 'cmvn_var', 'total.code', 'tgt_dict', 'reorder_tgt_dict', 'reorder_dict_map']
    print('cmvn_mean vec:\n', meta_data['cmvn_mean'])
    print('cmvn_var vec:\n', meta_data['cmvn_var'])

    vocab_size = len(meta_data['reorder_tgt_dict'])
    print('vocab size:\n', vocab_size)
    print('vocab size inscreased to minimum multiple of 8:\n', (vocab_size + 7) // 8 * 8)

    writer = FalconWriter(meta_hdfs_path, 102400)
    keys, vals = [], []
    for k, v in meta_data.items():
        # k is a string, such as 'cmvn_mean', 'total.code'
        # v can be any python object, such as dict, numpy array, string or others.
        keys.append(k)
        vals.append(pickle.dumps(v))
    writer.write_many(keys, vals)
    writer.flush()
    print('meta data uploaded to:\n' + meta_hdfs_path)


if __name__ == '__main__':
    main()
