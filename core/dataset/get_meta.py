'''
get dataset meta data.
'''

import multiprocessing as mp
import pickle
import time
from dataloader import FalconReader
from core.utils.dist_util import get_local_rank, get_local_size


class MetaGetter:
    '''
    get meta data in child process.
    if a FalconReader instantiated in main process,
    dataloader worker(child process) will stuck.
    because hdfs's dynamic library(used by FalconReader) is not safe for fork.
    '''

    def __init__(self, path):
        '''init.'''
        self.path = "".join(path.split())
        self.mp_manager = mp.Manager()
        self.queue = self.mp_manager.Queue(1)
        self.proc = mp.Process(target=self._read, args=(path, self.queue))
        self.proc.daemon = True
        self.proc.start()

    @staticmethod
    def _read(path, queue, retry=10):
        '''do read, it's in child process.'''
        chunk_size = 1
        io_retry = retry
        while retry >= 0:
            retry -= 1
            try:
                reader = FalconReader(
                    [path],  # path_list
                    1,  # fd_cache_size
                    1,  # io_thread_nums
                    io_retry,  # io_retry
                    'MetaGetter',  # unused
                    get_local_size(),  # GPU num in one worker
                    get_local_rank(),  # GPU idx in one worker
                    chunk_size,
                )

                ret = dict()
                keys = reader.list_keys([0], False)
                entry_nums = len(keys)
                chunk_idxs = [i * chunk_size for i in range(entry_nums // chunk_size)]
                vals = reader.read_many(chunk_idxs, True)
                for chunk_idx, val in enumerate(vals):
                    for idx, v in enumerate(val):
                        k = keys[chunk_idxs[chunk_idx] + idx]
                        ret[k] = pickle.loads(v)
                del keys, vals, chunk_idxs
                break
            except Exception:
                pass
            try:
                del reader
            except Exception:
                pass
            time.sleep(2)

        queue.put(ret, block=True, timeout=None)

    def get(self, retry=4):
        '''get value from child process.'''
        while retry > 0:
            try:
                ret = self.queue.get(block=True, timeout=60)
                return ret
            except Exception:
                retry -= 1

        raise RuntimeError('get meta data failed: ' + self.path)

    def close(self):
        '''close the getter, do child process join.'''
        self.proc.join()


def get_meta(path):
    '''
    read meta data from an arnold dataloader file.
    Args:
        path(str): file path, can be on local or hdfs.

    Returns:
        dict: a dict that contains all info in meta file.
    '''
    getter = MetaGetter(path)
    ret = getter.get()
    getter.close()
    return ret
