"""
data check

Usage:
`python3 scripts/data_check.py --data_file hdfs://xxxx`
or
`python3 scripts/data_check.py --data_file hdfs://xxxx --check_num data_num`

check_num means data checked num


path check

Usage:
    python3 path_check.py --config config_file
"""

import os
import os.path as osp
import sys
import pickle
import wave
import re
import io
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.io import wavfile
from dataloader import FalconReader, ParseTypes
import multiprocessing as mp

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append("..")
from core.utils import Config, get_logger
from core.dataset import get_meta
from core.dataset.preprocess import ProtoParser, DecordRaw, LabelParser, WavConvert


def convert_waveform_to_bin(waveform, sample_rate):
    """convert waveform data to binary data
    Args:
        waveform(np.ndarray):  The input signal of size (c, n)
        sample_rate(int): The frequency of the signal

    Returns:
        wave_data(bytes): The binary data of raw wav file
    """
    stream = io.BytesIO()
    f_write = wave.open(stream, 'wb')
    f_write.setnchannels(1)
    f_write.setsampwidth(2)
    f_write.setframerate(sample_rate)
    f_write.writeframes(waveform.astype(np.short).tostring())
    f_write.close()
    stream.seek(0)
    wave_data = stream.read()
    return wave_data

def convert_bin_to_waveform(wav):
    """convert binary data to waveform data
    Args:
        wave_data(bytes): The binary data of raw wav file

    Returns:
        sample_rate(int): The frequency of the signal
        waveform(np.ndarray):  The output signal of size (c, n)
    """
    sample_rate, waveform = wavfile.read(io.BytesIO(wav))
    waveform = waveform.reshape(1, -1).astype(np.float32)
    return sample_rate, waveform

def parse_tfrecord(item):
    ''' parse tfrecord item '''
    parser = ProtoParser()
    decoder = DecordRaw(key2type={'frames':'int16', 'transcript':'bytes', 'uttid':'bytes'})
    label_parser = LabelParser(in_key='transcript')
    wav_parser = WavConvert(in_key='frames')
    item = parser(item)
    item = decoder(item)
    item = label_parser(item)
    item = wav_parser(item)
    return item

def data_print(path, data_num=5):
    """
    data print, support wav transform
    """

    reader = FalconReader(path)
    raw_keys = reader.list_keys()
    entry_num = len(raw_keys)
    data_num = min(data_num, entry_num) if data_num > 0 else entry_num
    chunk_idxs = list(range(entry_num))
    chunk_idxs = chunk_idxs[:data_num]
    keys = raw_keys[:data_num]
    print(f"data path: {path}\n")
    print(f"all data nums: {entry_num}\n")
    vals = reader.read_many(chunk_idxs, True)
    new_vals = []
    for key, val in zip(keys, vals):
        print(f"\n\nentry key: {key}\n")
        print("values: \n")
        try:
            val = pickle.loads(val[0])
        except:
            val = parse_tfrecord(val[0])

        new_val = val.copy()
        assert isinstance(val, dict)
        for val_key, item in val.items():
            print("    ", val_key, ": ", type(item))
            # wav -> waveform
            if val_key == "wav":
                sample_rate, waveform = convert_bin_to_waveform(item)
                new_val['sample_rate'] = sample_rate
                new_val['waveform'] = waveform
            # waveform -> wav
            if val_key == 'waveform':
                wav = convert_waveform_to_bin(item, val['sample_rate'])
                new_val['wav'] = wav
        # generate .wav file with wav datas
        # with open('{}.wav'.format(key), 'wb') as f:
        #     f.write(new_val['wav'])
        new_vals.append(new_val)
    return keys, new_vals

def parse_dtype(path, data_num=5):
    """
    parse tfrecord datatype
    """
    reader = FalconReader(path)
    _keys = reader.list_keys()
    vals = reader.read_many(list(range(data_num)), True)
    vals = sum(vals, [])
    for idx, val in enumerate(vals):
        print(f"print data_{idx} key and dtype")
        key2types = ParseTypes(val)
        for data in key2types:
            name = bytes.decode(data[0])
            dtype = bytes.decode(data[1])
            print("data-key: ", name, ", value-type: ", dtype)
        print("\n")

def parse_args():
    '''parse args'''
    parser = ArgumentParser(description='path check scripts')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--data_file', help='read file path')
    parser.add_argument('--check_num', type=int, default=5, help='read file path')
    parser.add_argument('--parse_dtype', help='parse data type')
    parser.add_argument('--meta_file', help='print meta data')
    parser.add_argument('--get_time', help='print meta data')
    return parser.parse_known_args()

def get_data_list(dataset_cfg):
    '''
    get valid_file_list,train_file_list
    '''
    data_root = dataset_cfg.get("data_root", None)
    train_data_root = dataset_cfg.get("train_data_root", data_root)
    valid_data_root = dataset_cfg.get("valid_data_root", data_root)
    meta_data_root = dataset_cfg.get('meta_data_root', train_data_root)
    train_file_list = dataset_cfg.get("train_file_list", None)
    valid_file_list = dataset_cfg.get("valid_file_list", None)
    meta_file_list = dataset_cfg.get('meta_file', None)
    data_path_separator = dataset_cfg.get('data_path_separator', '')

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
    all_train_file_list = list()
    assert len(train_data_root) == len(train_file_list)
    for train_root, train_file in zip(train_data_root, train_file_list):
        train_dataset_now = [osp.join(train_root, p) for p in eval(train_file)]
        all_train_file_list += train_dataset_now

    # get valida_file_lists
    if isinstance(valid_data_root, str):
        valid_data_root = [valid_data_root]
        valid_file_list = [valid_file_list]
    all_valid_file_list = list()
    assert len(valid_data_root) == len(valid_file_list)
    for valid_root, valid_file in zip(valid_data_root, valid_file_list):
        valid_dataset_now = [osp.join(valid_root, p) for p in eval(valid_file)]
        all_valid_file_list += valid_dataset_now

    # get meta_file_lists
    if isinstance(meta_data_root, str):
        meta_data_root = [meta_data_root]
    if isinstance(meta_file_list, str) or not meta_file_list:
        meta_file_list = [meta_file_list]
    all_meta_file_list = list()
    for meta_root, meta_file in zip(meta_data_root, meta_file_list):
        if meta_file:
            all_meta_file_list.append(osp.join(meta_root, meta_file))

    return all_train_file_list, all_valid_file_list, all_meta_file_list

def format_file_list(file_list):
    """Format file list, merge sub0,sub1,sub2 to sub0-2.

    The training list is too long to make the log dirty.
    Format file list by merge prefix.
    """
    file_str = ""
    if len(file_list) == 0:
        return file_str
    if len(file_list) == 1:
        return f"{file_list[0]}"
    # Get all prefix and their postfix collections.
    # NOTE: sub file should endswith "[0-9]+$"
    prefix_dic = {}
    re_idx = re.compile(r"[0-9]+$")
    for f in file_list:
        idx = re_idx.findall(f)
        if len(idx) == 0:
            # Not splitted dataset parts.
            prefix_dic[f] = []
        elif len(idx) == 1:
            # dataset parts.
            idx = idx[0]
            prefix = f[: -len(idx)]
            if prefix not in prefix_dic:
                prefix_dic[prefix] = [int(idx)]
            else:
                prefix_dic[prefix].append(int(idx))
    # Format idx.
    file_str = ""
    for prefix, idx_list in prefix_dic.items():
        idx_str = format_idx(idx_list)
        if idx_str:
            file_str += f"{prefix}{idx_str}\n"
        else:
            file_str += f"{prefix}\n"
    return file_str


####### Split #######
def split_list(input_list, split_num):
    '''
    split list to split num
    '''
    split_size = len(input_list) // split_num
    total_shard_list = [
        input_list[idx * split_size : (idx + 1) * split_size] for idx in range(split_num)
    ]
    remain_size = len(input_list) - split_size * split_num
    if remain_size > 0:
        remain_list = input_list[split_size * split_num :]
        for remain_idx, remain_start_key in enumerate(remain_list):
            total_shard_list[remain_idx].append(remain_start_key)
    return total_shard_list


def check_path(path):
    '''check tensorbundle path'''
    if path.startswith("hdfs://"):
        ret = os.popen(f"hdfs dfs -ls {path}.*")
    else:
        ret = os.popen(f"ls {path}.*")
    lines = ret.readlines()
    file_nums = len(lines)
    data_file = path + '.data'
    shard_num = 0
    for line in lines:
        if data_file in line:
            shard_num = int(line.split('-')[-1])
            break
    if shard_num + 1 == file_nums:
        return True
    return False

def read_check(path, chunk_size=20, parrallel_chunk_size=20):
    ''' read file to check paths '''
    reader = FalconReader(path, chunk_size)
    keys = reader.list_keys()
    entry_num = len(keys)
    chunks = [i*chunk_size for i in range(entry_num // chunk_size)]
    for st in range(0, len(chunks), parrallel_chunk_size):
        vals = reader.read_many(chunks[st: st+parrallel_chunk_size])
        vals = sum(vals, [])
        if len(vals) < parrallel_chunk_size:
            return False
        for val in vals:
            if len(val) <=0:
                return False
    return True

def check_paths(path_list):
    '''check paths'''
    error_list = []
    for path in tqdm(path_list):
        is_ok = check_path(path) and read_check(path)
        if not is_ok:
            error_list.append(path)
    return error_list

def check_paths_with_pool(paths_list, nproc=40):
    pool = mp.Pool(processes=nproc)
    path_lists = split_list(paths_list, nproc)
    rets = []
    for pid in range(nproc):
        pool.apply_async(func=check_paths, args=(path_lists[pid], ), callback=rets.extend)
    pool.close()
    pool.join()
    return rets

def get_wav_time(path_list, chunk_size=30, parrallel_chunk_num=20):
    ''' get wav time'''
    reader = FalconReader(path_list, chunk_size)
    entry_num = reader.get_entry_num(list(range(len(path_list))), False)
    chunk_idxs = [i * chunk_size for i in range(entry_num // chunk_size)]
    times = 0
    for st in range(0, len(chunk_idxs), parrallel_chunk_num):
        chunks = chunk_idxs[st:st+parrallel_chunk_num]
        vals = reader.read_many(chunks)
        for val in vals:
            for v in val:
                item = pickle.loads(v)
                sample_rate, waveform = convert_bin_to_waveform(item['wav'])
                l = waveform.shape[-1] / sample_rate
                times += l
    return times


def path_check(args, unknown):
    '''main functions'''

    cfg = Config.fromfile(args.config)
    cfg.merge_from_list(unknown)
    train_paths, valid_paths, meta_paths = get_data_list(dataset_cfg=cfg.data)
    logger = get_logger(log_level='INFO')
    logger.info("start checking....")

    err_list = check_paths_with_pool(train_paths)
    logger.info(f"train-error-list:: \n{err_list}\n")

    err_list = check_paths_with_pool(valid_paths)
    logger.info(f"valid-error-list:: \n{err_list}\n")

    err_list = check_paths_with_pool(meta_paths)
    logger.info(f"meta-error-list:: \n{err_list}\n")

def print_meta(meta_file):
    ''' print meta data '''
    print("meta file: ", meta_file)
    meta = get_meta(meta_file)
    for key, val in meta.items():
        print(f"{key} ->", '\n', f" {val}", '\n')

def get_time_len(args, unknown):
    ''' '''
    cfg = Config.fromfile(args.config)
    cfg.merge_from_list(unknown)
    train_paths, valid_paths, meta_paths = get_data_list(dataset_cfg=cfg.data)
    times = get_wav_time(train_paths)
    print(times)

def main():
    '''real main func'''
    args, unknown = parse_args()
    if args.data_file:
        data_print(args.data_file, args.check_num)
    elif args.meta_file:
        print_meta(args.meta_file)
    elif args.parse_dtype:
        parse_dtype(args.parse_dtype, args.check_num)
    elif args.get_time:
        get_time_len(args, unknown)
    else:
        path_check(args, unknown)


if __name__ == '__main__':
    main()
