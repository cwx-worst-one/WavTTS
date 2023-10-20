import os

import numpy as np
import torch

import samantha.utils.hdfs_helper as hh


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_sami_ser(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    sami_ser = {}

    # wav2vec2
    h_wav2vec2 = f"{hpath}/wav2vec2_{local_rank}.pt"
    local_wav2vec2_path = f"{cache_dir}/wav2vec2_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_wav2vec2_path):
            if not hh.get(h_wav2vec2, local_wav2vec2_path):
                raise ConnectionError(f"Cannot retrieve file from {h_wav2vec2}.")
        sami_ser["wav2vec2"] = load_torch_script_module(local_wav2vec2_path, device)
    else:
        sami_ser["wav2vec2"] = load_torch_script_module(h_wav2vec2, device)

    # output_deg
    h_output_deg = f"{hpath}/output_deg_{local_rank}.pt"
    local_output_deg_path = f"{cache_dir}/output_deg_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_output_deg_path):
            if not hh.get(h_output_deg, local_output_deg_path):
                raise ConnectionError(f"Cannot retrieve file from {h_output_deg}.")
        sami_ser["output_deg"] = load_torch_script_module(local_output_deg_path, device)
    else:
        sami_ser["output_deg"] = load_torch_script_module(h_output_deg, device)


    # output_mlp
    h_output_mlp = f"{hpath}/output_mlp_{local_rank}.pt"
    local_output_mlp_path = f"{cache_dir}/output_mlp_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_output_mlp_path):
            if not hh.get(h_output_mlp, local_output_mlp_path):
                raise ConnectionError(f"Cannot retrieve file from {h_output_mlp}.")
        sami_ser["output_mlp"] = load_torch_script_module(local_output_mlp_path, device)
    else:
        sami_ser["output_mlp"] = load_torch_script_module(h_output_mlp, device)
    
    return sami_ser