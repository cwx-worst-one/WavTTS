import os
import pickle
import sys
import time
from collections import defaultdict
from time import perf_counter

import psutil
import torch
from tabulate import tabulate
from torch.utils.data import Dataset


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res["rss"] += mmap.rss
        res["pss"] += mmap.pss
        res["uss"] += mmap.private_clean + mmap.private_dirty
        res["shared"] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith("/"):  # looks like a file path
            res["shared_file"] += mmap.shared_clean + mmap.shared_dirty
    return res


class MemoryMonitor:
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ("", "K", "M", "G"):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)


def read_sample(x):
    return pickle.dumps(x)
    # A function that is supposed to read object x, incrementing its refcount.
    # This mimics what a real dataloader would do.
    if sys.version_info >= (3, 10, 6):
        # Before this version, pickle does not increment refcount. This is a bug that's
        # fixed in https://github.com/python/cpython/pull/92931.
        return pickle.dumps(x)
    else:
        import msgpack

        return msgpack.dumps(x)


def mock_worker(_, dataset: Dataset):
    while True:
        for sample in dataset:
            time.sleep(0.000001)
            result = read_sample(sample)


def start_processes(dataset: Dataset, n_workers: int):
    return torch.multiprocessing.start_processes(
        mock_worker,
        (dataset,),
        nprocs=n_workers,
        join=False,
        daemon=True,
        start_method="fork",
    )


def start_memory_monitor(dataset: Dataset, n_workers: int, n_seconds: int):
    monitor = MemoryMonitor()
    print(monitor.table())

    ctx = start_processes(dataset, n_workers)
    for pid in ctx.pids():
        monitor.add_pid(pid)

    try:
        for _ in range(n_seconds):
            print(monitor.table())
            time.sleep(1)
    except Exception as e:
        print(e)
    finally:
        ctx.join()
