import torch
import pathlib


def save_batch(batch, uttid_keyname="index", suffix="", save_dir="./"):
    save_dir = pathlib.Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    assert uttid_keyname in batch
    for bidx, uttid in enumerate(batch[uttid_keyname]):
        data = {}
        for k, v in batch.items():
            if v is None or k == uttid_keyname or isinstance(v, (int, float, str, dict)):
                continue
            data[k] = v[bidx]
        save_filename = [uttid]
        if suffix:
            save_filename.append(suffix)
        save_filename.append("pt")
        save_filename = save_dir / ".".join(save_filename)
        torch.save(data, save_filename)

    return
