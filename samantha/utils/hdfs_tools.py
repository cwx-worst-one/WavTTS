#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import subprocess
from contextlib import contextmanager
from typing import IO, Any, AnyStr, List

import torch

HDFS_BIN = "hdfs"


class hdfs_open:
    def __init__(self, hdfs_path: str, mode: str = "r"):
        self.hdfs = False
        if hdfs_path.startswith("hdfs://"):
            self.hdfs = True
            if mode == "r":
                self.pipe = subprocess.Popen(
                    "{} dfs -text {}".format(HDFS_BIN, hdfs_path),
                    shell=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            elif mode == "wa":
                self.pipe = subprocess.Popen(
                    "{} dfs -appendToFile - {}".format(HDFS_BIN, hdfs_path),
                    shell=True,
                    stdin=subprocess.PIPE,
                    text=True,
                )
            elif mode == "w":
                self.pipe = subprocess.Popen(
                    "{} dfs -put -f - {}".format(HDFS_BIN, hdfs_path),
                    shell=True,
                    stdin=subprocess.PIPE,
                    text=True,
                )
            else:
                raise RuntimeError("unsupported io mode: {}".format(mode))
        else:
            self.pipe = open(hdfs_path, mode, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, x, y, z):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration()

    def write(self, s: AnyStr) -> int:
        if self.hdfs:
            return self.pipe.stdin.write(s)
        else:
            return self.pipe.write(s)

    def writelines(self, lines: List[AnyStr]) -> None:
        if self.hdfs:
            self.pipe.stdin.writelines(lines)
        else:
            self.pipe.writelines(lines)

    def read(self, n: int = -1) -> AnyStr:
        if self.hdfs:
            return self.pipe.stdout.read(n)
        else:
            return self.pipe.read(n)

    def readline(self, limit: int = -1) -> AnyStr:
        if self.hdfs:
            return self.pipe.stdout.readline(limit)
        else:
            return self.pipe.readline(limit)

    def readlines(self, hint: int = -1) -> List[AnyStr]:
        if self.hdfs:
            return self.pipe.stdout.readlines(hint)
        else:
            return self.pipe.readlines(hint)

    def close(self) -> None:
        if self.hdfs:
            if self.pipe.stdin:
                self.pipe.stdin.close()
            if self.pipe.stdout:
                self.pipe.stdout.close()
            self.pipe.wait()
        else:
            self.pipe.close()

    def flush(self) -> None:
        if self.hdfs:
            self.pipe.stdin.flush()
        else:
            self.pipe.flush()


def hdfs_mkdir(hdfs_path: str):
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -mkdir -p {}".format(HDFS_BIN, hdfs_path), shell=True)
    else:
        if not os.path.exists(hdfs_path):
            os.makedirs(hdfs_path)


def hdfs_put(local_path: str, hdfs_path: str, force: bool = False):
    assert not local_path.startswith("hdfs://"), local_path
    assert hdfs_path.startswith("hdfs://"), hdfs_path
    opt_ = "-f" if force else ""
    subprocess.call(
        "{} dfs -put {} {} {}".format(HDFS_BIN, opt_, local_path, hdfs_path), shell=True
    )


def hdfs_get(hdfs_path: str, local_path: str):
    assert not local_path.startswith("hdfs://"), local_path
    assert hdfs_path.startswith("hdfs://"), hdfs_path
    subprocess.call(
        "{} dfs -get {} {}".format(HDFS_BIN, hdfs_path, local_path), shell=True
    )


def hdfs_rm(hdfs_path: str):
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -rm -r {}".format(HDFS_BIN, hdfs_path), shell=True)
    else:
        os.rmdir(hdfs_path)


def hdfs_cp(source_path: str, target_path: str, override: bool = False):
    src_hdfs = source_path.startswith("hdfs://")
    tgt_hdfs = target_path.startswith("hdfs://")
    if src_hdfs and tgt_hdfs:
        if override:
            subprocess.call(
                "{} dfs -cp -f {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
        else:
            subprocess.call(
                "{} dfs -cp {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
    elif src_hdfs:
        if override:
            if os.path.exists(target_path):
                subprocess.call("rm -rf {}".format(target_path), shell=True)
            subprocess.call(
                "{} dfs -get {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
        else:
            subprocess.call(
                "{} dfs -get {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
    elif tgt_hdfs:
        if override:
            subprocess.call(
                "{} dfs -put -f {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
        else:
            subprocess.call(
                "{} dfs -put {} {}".format(HDFS_BIN, source_path, target_path),
                shell=True,
            )
    else:
        if override:
            subprocess.call(
                "cp -r -f {} {}".format(source_path, target_path), shell=True
            )
        else:
            subprocess.call("cp -r {} {}".format(source_path, target_path), shell=True)


def hdfs_exists(hdfs_path: str):
    if not hdfs_path.startswith("hdfs://"):
        return os.path.exists(hdfs_path)
    info = (
        subprocess.Popen(
            "{} dfs -stat %n {}".format(HDFS_BIN, hdfs_path),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.read()
        .decode("utf-8")
        .splitlines()
    )
    if len(info) == 0:
        return False
    elif info[0] == os.path.split(hdfs_path)[-1]:
        return True
    else:
        print("error in {}".format(hdfs_path))
        exit(1)


def hdfs_loadtxt(hdfs_path: str):
    return hdfs_open(hdfs_path, "r").read().splitlines()


def hdfs_glob(hdfs_path: str):
    if hdfs_path.startswith("hdfs://"):
        info = (
            subprocess.Popen(
                "{} dfs -ls {}".format(HDFS_BIN, hdfs_path),
                shell=True,
                stdout=subprocess.PIPE,
            )
            .stdout.read()
            .decode("utf-8")
            .splitlines()
        )

        filelist = []
        for line in info:
            items = line.split()
            if len(items) == 8:
                filelist.append(items[-1])
    else:
        filelist = os.listdir(hdfs_path)
    return filelist


@contextmanager
def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HDFS_BIN, hdfs_path),
            shell=True,
            stdout=subprocess.PIPE,
        )
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HDFS_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HDFS_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hdfs_torch_load(filepath: str, jit: bool = False, **kwargs):
    if jit:
        torch_load_fn = torch.jit.load
    else:
        torch_load_fn = torch.load

    if not filepath.startswith("hdfs://"):
        return torch_load_fn(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch_load_fn(accessor, **kwargs)
        del accessor
        return state_dict


def hdfs_torch_save(obj, filepath: str, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)
