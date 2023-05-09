import os

import numpy as np
import torch
from transformers import AutoModel

import samantha.utils.hdfs_helper as hh

from ..utils.dist import local_zero_first
from .w2v.ssl_frontend import SSLFrontend


def value(func: str):
    exec_data = {}
    exec(f"ret = {func}", globals(), exec_data)
    return exec_data["ret"]


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_mulan(hpath, local_rank, cache_dir=None, version="149"):
    if version in ["149"]:
        from .mulan.mulan_infer_149 import (
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    elif version in ["247", "191"]:
        from .mulan.mulan_infer_247 import (
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    elif version in ["115"]:
        from .mulan.mulan_infer_115 import (
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    elif version in ["g4"]:
        from .mulan.mulan_infer_g4 import (
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    else:
        raise KeyError(f"Not a valid mulan version. {version}")
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
                "mulan_rvq_fn": mulan_rvq_indexs,
            }
    else:
        local_path = hpath
        with local_zero_first():
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
                "mulan_rvq_fn": mulan_rvq_indexs,
            }


def init_mulan_centers(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            mulan_centers = np.load(local_path)
            mulan_centers = torch.from_numpy(mulan_centers).float().to(device)
            return {"mulan_centers": mulan_centers}
    else:
        with local_zero_first():
            mulan_centers = np.load(hpath)
            mulan_centers = torch.from_numpy(mulan_centers).float().to(device)
            return {"mulan_centers": mulan_centers}


def init_mert(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    # loading our model weights
    model = (
        AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir="./inference_test"
        )
        .eval()
        .to(device)
    )

    local_path = f"{cache_dir}/{os.path.basename(hpath)}"
    if hpath.startswith("hdfs://"):
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            return {
                "mert_model": model,
                "mert_centroids": torch.from_numpy(np.load(local_path)).to(device),
            }
    else:
        with local_zero_first():
            return {
                "mert_model": model,
                "mert_centroids": torch.from_numpy(np.load(hpath)).to(device),
            }


def init_w2v(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    h_semantic, h_centroids = (
        f"{hpath}/semantic.jit.pt",
        f"{hpath}/centroids_epoch_10.npy",
    )

    l_semantic, l_centroids = (
        f"{cache_dir}/semantic.jit.pt",
        f"{cache_dir}/centroids_epoch_10.npy",
    )
    if hpath.startswith("hdfs://"):
        with local_zero_first():
            if not os.path.exists(l_centroids):
                if not hh.get(h_centroids, l_centroids):
                    raise ConnectionError(f"Cannot retrieve file from {h_centroids}.")

            if not os.path.exists(l_semantic):
                if not hh.get(h_semantic, l_semantic):
                    raise ConnectionError(f"Cannot retrieve file from {h_semantic}.")
        return {
            "ssl_frontend": SSLFrontend(),
            "semantic": load_torch_script_module(l_semantic, device),
            "centroids": torch.from_numpy(np.load(l_centroids)).to(device),
        }
    else:
        with local_zero_first():
            return {
                "ssl_frontend": SSLFrontend(),
                "semantic": load_torch_script_module(h_semantic, device),
                "centroids": torch.from_numpy(np.load(h_centroids)).to(device),
            }


def init_sound_stream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss{local_rank}.pt"
    local_path = f"{cache_dir}/ss{local_rank}.pt"

    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_path):
            if not hh.get(h_ss, local_path):
                raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        return {"ss": load_torch_script_module(local_path, device)}
    else:
        return {"ss": load_torch_script_module(h_ss, device)}


def init_sound_stream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"

    if not os.path.exists(local_path):
        if not hh.get(h_ss, local_path):
            raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
    return {"ss_dec": load_torch_script_module(local_path, device)}


def init_coarse_2ar(hpath, local_rank, cache_dir=None):
    from ..lit_modules.v2.lit_coarse_2ar import CoarseModule

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    local_path = f"{cache_dir}/{os.path.basename(hpath)}"

    if not os.path.exists(local_path):
        if not hh.get(hpath, local_path):
            raise ConnectionError(f"Cannot retrieve file from {hpath}.")
    model = CoarseModule.load_from_checkpoint(local_path, device=device)
    model.eval()
    return {"coarse": model.to(device)}


def init_fine_dummy(hpath, local_rank, cache_dir=None):
    from ..lit_modules.v2.lit_coarse_2ar import CoarseModule

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    local_path = f"{cache_dir}/{os.path.basename(hpath)}"

    if not os.path.exists(local_path):
        if not hh.get(hpath, local_path):
            raise ConnectionError(f"Cannot retrieve file from {hpath}.")
    model = CoarseModule.load_from_checkpoint(local_path, device=device)
    model.eval()
    return {"fine": model.to(device)}
