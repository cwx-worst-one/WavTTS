import os
import torch
import numpy as np

from recipes.diffusion.models.semantic_model.model import SSLFrontend

def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location='cpu').to(device).eval()
    return module

def init_wav2vec(trainer, path, device, cache_dir=None):
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
        "ssl_frontend": SSLFrontend(),
        "semantic": load_torch_script_module(local_path, device),
    }

def init_semantic_centers(trainer, path, device, cache_dir=None):
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
    semantic_centers = np.load(local_path)
    semantic_centers = torch.from_numpy(semantic_centers).float().to(device)

    return {"semantic_centers": semantic_centers}
    

def w2v_bert_tokenization(frontend, w2v_model, wavs, centers, device):
    with torch.no_grad():
        b, t = wavs.size()
        with torch.autocast(device_type="cuda", enabled=False):
            feats, feat_mask = frontend(
                wavs, torch.LongTensor([t]).repeat([b]).to(device)
            )
        w2v_embeds, _ = w2v_model(feats, feat_mask)
        # kmeans
        b, t, d = w2v_embeds.shape
        dataset = w2v_embeds.view([b * t, d])
        num_points = dataset.size(0)
        # 5e8 should vary depending on the free memory on the GPU
        # Ideally, automatically ;)
        chunk_size = int(5e8)
        codes = torch.zeros(num_points, dtype=torch.long, device=device)
        centers_t = torch.transpose(centers, 0, 1)  # [1024, 1024]
        centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
        inertia = 0
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(begin + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers_t)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
            inertia += distances[range(distances.shape[0]), min_ind].sum()
        codes = codes.view([b, t])
        return codes