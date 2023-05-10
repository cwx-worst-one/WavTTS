''' runner utils. '''

import os
import subprocess
import re
import sys
import time
import math
from getpass import getuser
from socket import gethostname
import torch
from core.utils.misc import is_str
from core.utils import logging
from core.utils.misc import infer_text_format


def get_host_info():
    '''get_host_info.'''
    return f'{getuser()}@{gethostname()}'


def get_time_str():
    '''get_time_str.'''
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def get_max_memory():
    '''get max gpu memory.'''
    mem = torch.cuda.max_memory_allocated()
    return mem / (1024 * 1024)


def get_word_boundary(res_str, spike_list, frames_len, downsampling_size, time_delay_frames=0):
    '''get word boundary based on spike'''
    frame_time = 10
    utt_time = frames_len * frame_time
    time_delay_time = time_delay_frames * frame_time
    spike_word_index = res_str.split()
    spike_index_list = [
        ((index_tmp + 1) * downsampling_size - 1) * frame_time for index_tmp in spike_list
    ]
    spike_time_list = []
    average_word_boundary = 300

    list_len = len(spike_index_list)
    if list_len == 0:
        spike_time_list.append(["sil", 0, utt_time])
        return spike_time_list
    if list_len == 1:
        ind_start = spike_index_list[0] - average_word_boundary / 2
        ind_end = spike_index_list[0] + average_word_boundary / 2
        ind_start = max(0, ind_start - time_delay_time)
        ind_end = min(utt_time - 1, ind_end - time_delay_time)
        spike_time_list.append([spike_word_index[0], ind_start, ind_end])
        spike_time_list.append(["sil", ind_end, utt_time])
        return spike_time_list
    prev_bound = (spike_index_list[1] - spike_index_list[0]) / 2
    for i in range(0, list_len):
        if i < list_len - 1:
            next_bound = (spike_index_list[i + 1] - spike_index_list[i]) / 2
        ind_start = spike_index_list[i] - prev_bound
        ind_end = spike_index_list[i] + next_bound
        prev_bound = next_bound
        ind_start = max(0, ind_start - time_delay_time)
        ind_end = min(utt_time - 1, ind_end - time_delay_time)
        spike_time_list.append([spike_word_index[i], ind_start, ind_end])
        if i == list_len - 1:
            spike_time_list.append(["sil", ind_end, utt_time])
    return spike_time_list


def get_oracle_ed_info(hyps, uid, ref_format, tgt_dict, filter_list, formator, ed_calculator, rank):
    '''get oracle info of nbest hyp'''
    min_cer = 100
    best_ed_info = None
    best_align_info = None
    for beam_idx, beam_hyp in enumerate(hyps):
        if isinstance(beam_hyp, tuple):
            beam_hyp = beam_hyp[0]
        hyp_str = tgt_dict.string(beam_hyp)
        res_format = infer_text_format(hyp_str, filter_list, formator)
        ed_info, align_info = ed_calculator.show_alignment(uid, ref_format, res_format)
        cur_cer = (ed_info['ins_err'] + ed_info['del_err'] + ed_info['sub_err']) / max(
            1, ed_info['ref_word_num']
        )
        logging.info(
            'rank %d, uttid %s, beam %d, CER %.2f, %s',
            rank,
            uid,
            beam_idx,
            cur_cer * 100,
            ' '.join(res_format),
        )
        if cur_cer < min_cer:
            min_cer = cur_cer
            best_ed_info = ed_info
            best_align_info = align_info
    return best_ed_info, best_align_info


def get_time(time_name):
    '''It's a decorator which used to calculate the time fuction runs.'''

    def wrapper(func):
        def decorate(*args, **kw):
            runner = args[0]
            start_time = time.time()
            result = func(*args, **kw)
            end_time = time.time()
            if runner.mode == 'train':
                runner.train_log_buffer.update({time_name: end_time - start_time})
            return result

        return decorate

    return wrapper


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


def format_idx(idx_list):
    """Format idx list.

    1. Merge continuous idx: [0,1,2,3] -> 0-3;
    2. Keep repeated idx: [0,0,1,2,3,3,4] -> 0,0-3,3-4;
    3. Mark repeat times: [0,0,1,1,2,2] -> 0-2; REPEAT 2 TIMES;
    """
    # pylint:disable=too-many-branches
    idx_str = ""
    if len(idx_list) == 0:
        return ""
    # De-duplicate idx_list.
    idx_list = sorted(idx_list)
    repeat_times = 1.0
    if len(idx_list) != len(set(idx_list)):
        idx_no_repeat = sorted(list(set(idx_list)))
        repeat_times = len(idx_list) / len(idx_no_repeat)
        if math.ceil(repeat_times) == math.floor(repeat_times):
            idx_repeat = sorted(idx_no_repeat * int(repeat_times))
            if idx_list == idx_repeat:
                repeat_times = int(repeat_times)
                idx_list = idx_no_repeat
    if len(idx_list) == 1:
        idx_str = f"{idx_list[0]}"
        if isinstance(repeat_times, int) and repeat_times > 1:
            idx_str += f"; REPEAT {repeat_times} TIMES"
        return idx_str

    # Generate idx_str.
    pre = idx_list[0]  # Previous idx.
    bottom = idx_list[0]  # min value of latest continuous idx sequence.
    for cur in idx_list[1:]:
        if cur == pre + 1:
            # In a continuous idx sequence.
            pre = cur
        else:
            if pre == bottom:
                idx_str += f"{bottom},"
            elif pre > bottom:
                idx_str += f"{bottom}-{pre},"
            pre = cur
            bottom = cur
    if pre == bottom:
        idx_str += f"{bottom}"
    elif pre > bottom:
        idx_str += f"{bottom}-{pre}"
    if isinstance(repeat_times, int) and repeat_times > 1:
        idx_str += f"; REPEAT {repeat_times} TIMES"
    return idx_str


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
