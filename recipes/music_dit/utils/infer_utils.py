from scipy.io.wavfile import write
import random
import numpy as np
import torch
import os
import samantha.utils.hdfs_helper as hh

def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32767
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def to_device(tensors, device):
    if isinstance(tensors, list):
        tensors_to_device = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                tensors_to_device.append(tensor.to(device))
            else:
                tensors_to_device.append(tensor)
    elif isinstance(tensors, dict):
        tensors_to_device = dict()
        for k, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                tensors_to_device[k] = tensor.to(device)
            else:
                tensors_to_device[k] = tensor
    return tensors_to_device

def load_torch_script(model_path, rank, cache_dir):
    device = f"cuda:{rank}"
    if hh.ishdfs(model_path):
        # model_path = model_path % rank
        os.makedirs(cache_dir, exist_ok=True)
        fn = os.path.basename(model_path)
        local_path = os.path.join(cache_dir, fn)
        if not os.path.exists(local_path):
            success = hh.get(model_path, local_path)
            if not success:
                raise ConnectionError(
                    f"failed to retrieve {model_path} to {local_path}"
                )
        return torch.jit.load(local_path).to(device).eval()
    else:
        return torch.jit.load(model_path).to(device).eval()
