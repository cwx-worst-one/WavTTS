import subprocess
import os
import shutil
import tempfile
import json

if "TRASH_DIR" not in os.environ or not os.environ["TRASH_DIR"]:
    os.environ["TRASH_DIR"] = "."

def remote_load(*idx):
    """
    Decorator of load function.

    Args:
        idx (list): index of load filename in args.
    """
    def wrap(func):
        nonlocal idx
        remain_idx = idx[:-1]
        idx = idx[-1]
        def decorate_func(*args, **kwargs):
            nonlocal idx
            args = list(args)
            fn = args[idx]
            if is_hdfs(fn):
                with tempfile.TemporaryDirectory(
                        prefix="tmp-", dir=os.environ["TRASH_DIR"]) as temp_dir:
                    local_fn = os.path.join(temp_dir, os.path.basename(fn))
                    hdfs_copy(fn, local_fn, overwrite=True)
                    args[idx] = local_fn
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        if remain_idx:
            return remote_load(*remain_idx)(decorate_func)
        else:
            return decorate_func
    return wrap

def is_hdfs(path):
    """
    Check if path is hdfs.

    Args:
        path (str): path to check.

    Return:
        (bool)
    """
    return path.startswith("hdfs://") or path.startswith("webhdfs://")

def hdfs_copy(src, dst, overwrite=False):
    """
    Copy file/directory from source to destination.

    Args:
        src (str): path of source.
        dst (str): path of destination.
        overwrite (bool): whether to overwrite if destination exists.
    """
    if src == dst:
        return

    if os.path.exists(dst):
        if overwrite:
            os.remove(dst)
        else:
            raise ValueError(f"{dst} exists, abort! (or set overwrite=True)")
    subprocess.check_call(
        f"hdfs dfs -get {src} {dst}",
        shell=True
    )

@remote_load(0)
def load_json(fn, encoding='utf-8'):
    """
    Load dictionary from file.

    Args:
        fn (str): file name.
        encoding (str): encoding type.

    Return:
        (dict): text, list of str.
    """
    with open(fn, 'r', encoding=encoding) as fid:
        return json.load(fid)
