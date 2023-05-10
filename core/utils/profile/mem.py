'''
memory profiler.
usage:
    invoke the hook to model before train begin:
    ```
        solution = build_solution(cfg)
        invoke_memory_profile(solution, 'MyRNNTSolution', 2)
        # do train
    ```
'''

import gc
import torch
from core.utils.dist_util import get_rank


def readable_size(size):
    '''get readable size.'''
    ends = ['B', 'KB', 'MB', 'GB', 'TB']
    ei = 0
    while abs(size) > 1024:
        ei += 1
        size /= 1024
    return '{:.2f} {}'.format(size, ends[ei])


PYTORCH_MIN_ALLOCATE = 2**9


def get_batch_info(obj, shape=True):
    '''get tensor size.'''
    if isinstance(obj, torch.Tensor):
        if shape:
            return str(obj.shape)
        element_size = obj.element_size()
        fact_numel = obj.storage().size()
        fact_memory_size = fact_numel * element_size
        fact_memory_size += PYTORCH_MIN_ALLOCATE - 1
        fact_memory_size = fact_memory_size // PYTORCH_MIN_ALLOCATE
        fact_memory_size *= PYTORCH_MIN_ALLOCATE
        return readable_size(fact_memory_size)
    if isinstance(obj, (tuple, list)):
        ret = []
        for ob in obj:
            if isinstance(ob, str):
                continue
            ret.append(get_batch_info(ob))
        return ret
    if isinstance(obj, dict):
        ret = {}
        for k, v in obj.items():
            if isinstance(v, str):
                continue
            ret[k] = get_batch_info(v)
        return ret
    return 'Unkown Type'


def get_tensors(record):
    '''
    to get tensor's contents
    Args:
       record: a list which contains two-turple like (tensors,torch.numl(tensor))
    Return:
       a string of tensors
    '''
    tensors = ""
    for i in range(min(len(record), 3)):
        tensors += " " + record[i][0]
    return tensors


def invoke_mem_profile_record(name, stage, depth):
    '''
    invoke memory profile record.
    It's usefull for manual.
    '''
    # gc.collect()
    # torch.cuda.empty_cache()
    mem_size = torch.cuda.memory_allocated()
    if MemProfileHook.pre_mem_size is None:
        diff = 0
    else:
        diff = mem_size - MemProfileHook.pre_mem_size
    objects = gc.get_objects()
    add_tensors = []
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor) and obj.is_cuda]
    for tensor in tensors:
        max_size = len(MemProfileHook.id_set)
        MemProfileHook.id_set.add(id(tensor))
        if max_size != len(MemProfileHook.id_set):
            add_tensors.append((str(tensor.shape), torch.numel(tensor)))
    for i in range(0, min(3, len(add_tensors))):
        max_elememt = add_tensors[i][1]
        for j in range(i + 1, len(add_tensors)):
            if max_elememt < add_tensors[i][1]:
                add_tensors[i], add_tensors[j] = add_tensors[j], add_tensors[i]
    record = (name, stage, mem_size, depth, diff, add_tensors)
    MemProfileHook.id_set.clear()
    for tensor in tensors:
        MemProfileHook.id_set.add(id(tensor))
    MemProfileHook.records.append(record)
    MemProfileHook.pre_mem_size = mem_size


class MemProfileHook:
    '''
    module hook for memory profile.
    '''

    def __init__(self, name, stage, depth, is_root=False):
        '''init.'''
        self.name = name
        self.stage = stage
        self.depth = depth
        self.is_root = is_root

    def __call__(self, module, inputs, outputs=None):
        '''
        callable function for memory profile.
        Args:
            module(torch.nn.Module): related module.
            inputs(any): module function's input.
            outputs(any): module function's outputs.
        Return:
            None: this is a none sense return value.
        '''
        if self.is_root:
            analysis_records(MemProfileHook.records)
            MemProfileHook.records.clear()
            MemProfileHook.times += 1
            MemProfileHook.input_info = get_batch_info(inputs)
        invoke_mem_profile_record(self.name, self.stage, self.depth)


def analysis_records(records):
    '''analysis records.'''
    if len(records) == 0:
        return
    sorted_records = sorted(records, key=lambda x: x[4])
    out = []
    out.append('=' * 10 + 'Cuda Memory Profile {}:'.format(MemProfileHook.times) + '=' * 10)
    out.append('Input Batch Data info: ' + str(MemProfileHook.input_info))
    out.append('=' * 10 + 'MAX Cuda Memory Cost' + '=' * 10)
    for i in range(5):
        idx = len(sorted_records) - 1 - i
        record = sorted_records[idx]
        if record[4] <= 0:
            break
        prefix = ' ' * 4 * (record[3] - 1)
        tensors = get_tensors(record[5])
        out.append(
            prefix
            + '{} {}: current mem {}, later cost {} {}'.format(
                record[0], record[1], readable_size(record[2]), readable_size(record[4]), tensors
            )
        )
    out.append('=' * 10 + 'MAX Cuda Memory free' + '=' * 10)
    for i in range(3):
        idx = i
        record = sorted_records[idx]
        if record[4] >= 0:
            break
        prefix = ' ' * 4 * (record[3] - 1)
        tensors = get_tensors(record[5])
        out.append(
            prefix
            + '{} {}: current mem {}, later cost {} {}'.format(
                record[0], record[1], readable_size(record[2]), readable_size(record[4]), tensors
            )
        )
    out.append('=' * 10 + 'Records list' + '=' * 10)
    for record in records:
        prefix = ' ' * 4 * (record[3] - 1)
        tensors = get_tensors(record[5])
        out.append(
            prefix
            + '{} {}: current mem {}, later cost {} {}'.format(
                record[0], record[1], readable_size(record[2]), readable_size(record[4]), tensors
            )
        )
    out.append('=' * 10 + 'End Cuda Memory Profile {}:'.format(MemProfileHook.times) + '=' * 10)
    file_name = 'mem_profile_rank_{}_times_{}.txt'.format(get_rank(), MemProfileHook.times)
    out = [line + '\n' for line in out]
    with open(file_name, 'w', encoding='utf-8') as f:
        f.writelines(out)


def invoke_memory_profile(model, name='Solution', max_depth=4, current_depth=1):
    '''
    invoke memory profiler into a model.
    Args:
        model(torch.nn.Module): model to profile.
        name(str): model name.
        max_depth(int): max module depth to profile.
        current_depth(int): current depth.
    Return:
        torch.nn.Module: the input model.
    '''
    if current_depth == 1:
        MemProfileHook.times = 0
        MemProfileHook.pre_mem_size = None
        MemProfileHook.input_info = None
        MemProfileHook.records = []
        MemProfileHook.id_set = set()
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if current_depth >= max_depth:
        return model
    model.register_forward_pre_hook(
        MemProfileHook(name, 'pre forward', current_depth, current_depth == 1)
    )
    model.register_forward_hook(MemProfileHook(name, 'after forward', current_depth))
    model.register_backward_hook(MemProfileHook(name, 'after backward', current_depth))

    for child_name, module in model.named_children():
        invoke_memory_profile(module, child_name, max_depth, current_depth + 1)

    return model
