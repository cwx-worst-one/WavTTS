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


def init_pretrained_bestrq(hpath, local_rank, cache_dir=None):
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
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        return {"pretrained_bestrq_state": state_dict}


def init_pretrained(hpath, local_rank, cache_dir=None):
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
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        return {"state_dict": state_dict}


def init_bestrq_mel_ctc_vq(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import BestRQMelCTC

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
        model = BestRQMelCTC.load_from_checkpoint(local_path).to(device).eval()
        return {"BestRQMelCTCVQ": model}


def init_mulan(hpath, local_rank, cache_dir=None):
    from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
        create_mulan_model,
        mulan_inference,
    )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }
        else:
            local_path = hpath
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }


def init_stage2(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage2

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
        model = Stage2.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage2": model}


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
