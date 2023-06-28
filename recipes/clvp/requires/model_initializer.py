import os

import numpy as np
import torch

import samantha.utils.hdfs_helper as hh

from samantha.utils.distributed import rank_zero_first


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_sound_stream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_encoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_encoder_{local_rank}.pt"

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


def init_mhubert(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    h_semantic, h_centroids = (
        f"{hpath}/semantic.jit.pt",
        f"{hpath}/centroids_epoch_100.npy",
    )

    l_semantic, l_centroids = (
        f"{cache_dir}/semantic.jit.pt",
        f"{cache_dir}/centroids_epoch_100.npy",
    )
    from .mhubert.hubert import mhubert_infer
    if hpath.startswith("hdfs://"):
        with rank_zero_first(is_global=False):
            if not os.path.exists(l_centroids):
                if not hh.get(h_centroids, l_centroids):
                    raise ConnectionError(f"Cannot retrieve file from {h_centroids}.")

            if not os.path.exists(l_semantic):
                if not hh.get(h_semantic, l_semantic):
                    raise ConnectionError(f"Cannot retrieve file from {h_semantic}.")
        return {
            "semantic": load_torch_script_module(l_semantic, device),
            "centroids": torch.from_numpy(np.load(l_centroids)).to(device),
            "semantic_infer_fn": mhubert_infer,
        }
    else:
        with rank_zero_first(is_global=False):
            return {
                "semantic": load_torch_script_module(h_semantic, device),
                "centroids": torch.from_numpy(np.load(h_centroids)).to(device),
                "semantic_infer_fn": mhubert_infer,
            }
    