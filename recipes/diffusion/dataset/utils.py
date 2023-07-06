import hashlib
import torch


def fix_hash(x):
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**8


def collate_fn(batch):
    keys = ["music_id", "audio"]
    out_batch = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            out_batch[k].append(b[k])

    for k, v in out_batch.items():
        if k == "audio":
            out_batch[k] = torch.cat(v)

    return out_batch

def worker_init_fn(worker_id):
    np.random.seed()
    random.seed()
