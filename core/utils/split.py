# pylint: disable=missing-function-docstring, missing-module-docstring
import random


def split_list(input_list, split_num):
    # split_size = (len(input_list) + split_num - 1) // split_num
    split_size = len(input_list) // split_num
    total_shard_list = [
        input_list[idx * split_size : (idx + 1) * split_size] for idx in range(split_num)
    ]
    remain_size = len(input_list) - split_size * split_num
    if remain_size > 0:
        remain_list = input_list[split_size * split_num :]
        # random scatter the remain start key to shard list
        random_shard_idx_list = list(range(split_num))
        random.shuffle(random_shard_idx_list)
        for remain_idx, remain_start_key in enumerate(remain_list):
            total_shard_list[random_shard_idx_list[remain_idx]].append(remain_start_key)
    return total_shard_list


def flatten_list(input_list):
    """get a flatten_list
    example:
        input: [1,[2,3], [4,[5,[6]]],[[7]]]
        output: [1,2,3,4,5,6,7]
    """
    new_list = []
    for elem in input_list:
        if not isinstance(elem, list):
            new_list.append(elem)
        else:
            new_list.extend(flatten_list(elem))
    return new_list
