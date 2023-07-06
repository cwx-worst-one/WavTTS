import numpy as np
import torch
import os
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
    mulan_rvq_indexs
)

def mulan_inference_wrapper(mulan_model, mulan_centers, text, music, device):
    if text is not None:
        mulan_emb = mulan_inference(mulan_model, text=text, device=device)
    elif music is not None:
        mulan_emb = mulan_inference(mulan_model, music=music, device=device)

    mulan_ids, ds = mulan_rvq_indexs(mulan_emb, mulan_centers)

    return mulan_ids

def init_mulan_centers(trainer, path, device, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    local_path = f"{cache_dir}/{os.path.basename(path)}"

    if path.startswith("hdfs://") or path.startswith("/home"):
        if trainer.local_rank == 0:
            if not os.path.exists(local_path):
                try:
                    os.system(f"hdfs dfs -get {path} {cache_dir}")
                except Exception:
                    raise ConnectionError(f"Cannot retrieve file from {path}.")
    trainer.strategy.barrier()

    mulan_centers = np.load(local_path)
    mulan_centers = torch.from_numpy(mulan_centers).float().to(device)

    return {
        "mulan_centers": mulan_centers
    }

def init_mulan(trainer, path, device, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    local_path = f"{cache_dir}/{os.path.basename(path)}"

    if path.startswith("hdfs://") or path.startswith("/home"):
        if trainer.local_rank == 0:
            if not os.path.exists(local_path):
                try:
                    os.system(f"hdfs dfs -get {path} {cache_dir}")
                except Exception:
                    raise ConnectionError(f"Cannot retrieve file from {path}.")
    trainer.strategy.barrier()

    return {
        "mulan_model": create_mulan_model(local_path, device).to(device).eval()
    }