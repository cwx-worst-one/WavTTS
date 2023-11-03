import contextlib
import os

import torch
import torch.distributed as dist

import samantha.utils.hdfs_helper as hh


def is_local_zero():
    local_rank = os.getenv("LOCAL_RANK", None)
    return local_rank is None or local_rank == "0"


@contextlib.contextmanager
def local_zero_first():
    if not dist.is_initialized():
        yield
    else:
        if not is_local_zero():
            dist.barrier()
        yield
        if is_local_zero():
            dist.barrier()


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_stage3(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage3

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = Stage3.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage3": model}


def init_unified_decoder(hpath, local_rank, cache_dir=None):
    from recipes.mir.modules.lit_module import UnifiedDecoder

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = UnifiedDecoder.load_from_checkpoint(local_path).to(device).eval()
        return {"unified_decoder": model}
