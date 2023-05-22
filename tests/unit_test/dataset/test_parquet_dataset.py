"""
an example you can test ParquetDataset
"""
import os
import subprocess
from core.utils import Config, logging, get_logger
from torch.utils.data import DataLoader
from core.dataset.parquet_dataset import ParquetDataset


def run_command(cmd):
    '''run command, get output'''
    rets = None
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8'
    ) as p:
        rets = p.communicate()[0]
    return rets


def list_tensorbundle_files(cmd, cur_pattern):
    '''list tensorbundle files'''
    cmd = f"{cmd} {cur_pattern}*.index"
    text_wrapper = run_command(cmd)
    ret_file_list = []
    for elem in text_wrapper.strip().split():
        if elem.startswith("hdfs://") or elem.startswith("/mnt"):
            ret_file_list.append(elem[:-6])
    return ret_file_list


def list_tfrecord_files(cmd, cur_pattern):
    '''list tfrecord files'''
    cmd = f"{cmd} {cur_pattern}"
    text_wrapper = run_command(cmd)
    ret_file_list = []
    for elem in text_wrapper.strip().split():
        if elem.startswith("hdfs://") or elem.startswith("/mnt"):
            ret_file_list.append(elem)
    return ret_file_list


# pylint: disable='redefined-outer-name'
def file_pattern(data_roots, file_patterns):
    '''deal file pattern'''
    ret_file_lists = []
    ret_str_flag = False
    if isinstance(data_roots, str):
        data_roots = [data_roots]
        file_patterns = [file_patterns]
        ret_str_flag = True

    for data_root, file_pattern in zip(data_roots, file_patterns):
        if data_root.startswith("hdfs://"):  # hdfs_files
            cmd = 'hdfs dfs -ls'
        elif data_root.startswith("/mnt"):
            cmd = 'ls'
        data_root_length = len(data_root)
        if data_root[-1] != '/':
            data_root_length += 1
        ret_file_list = []
        for cur_file_pattern in eval(file_pattern):
            # only contains *, then will go file pattern logic
            if '*' not in cur_file_pattern:
                ret_file_list.append(cur_file_pattern)
                continue
            cur_pattern = os.path.join(data_root, cur_file_pattern)
            file_list = list_tensorbundle_files(cmd, cur_pattern)
            if len(file_list) <= 0:
                file_list = list_tfrecord_files(cmd, cur_pattern)
            for cur_file in file_list:
                cur_file_name = cur_file[data_root_length:]
                ret_file_list.append(cur_file_name)
        ret_file_lists.append(str(ret_file_list))  # trans to str for support eval
    if ret_str_flag:
        ret_file_lists = ret_file_lists[0]
    return ret_file_lists


def get_parquet_file_list(
    data_roots=[
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xiaohai/'
        'Multi-Modal/data/multi-modal-eval/zhibo_mm/audit_pass/20230401'
    ],
    file_lists=['["part-*-dd58a7d3-630b-44fb-9019-22a0ba62d9b5-c000.snappy.parquet"]'],
):
    '''
    get parquet file list
    '''
    file_lists = file_pattern(data_roots, file_lists)
    path_list = []
    for data_root, file_list in zip(data_roots, file_lists):
        for file_path in eval(file_list):
            path_list.append(os.path.join(data_root, file_path))
    return path_list

def test_hdfs_dataset():
    '''main function'''
    get_logger(log_level='INFO')

    data_path = get_parquet_file_list()
    cfg = Config({'read_batch_size': 20})
    dataset = ParquetDataset(data_path, cfg=cfg)
    iter_num = 20
    dataloader = DataLoader(dataset, num_workers=2, batch_size=None)

    for idx, data in enumerate(dataloader):
        if idx > iter_num:
            break
        print(idx, data.keys())
    logging.error('dataset end.')


if __name__ == '__main__':
    test_hdfs_dataset()
