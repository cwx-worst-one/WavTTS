from functools import partial
import os
import torch
import contextlib
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

def create_umm(ckpt_path, rank):
    from recipes.umm.requires.model_initializer import init_stage3
    model = init_stage3(ckpt_path, rank, "./")["Stage3"].eval()
    return model

def create_bigvgan_mel(rank):
    bigvgan_mel = partial(mel_spectrogram, 
            n_fft=1024, 
            num_mels=100, 
            sampling_rate=24000, 
            hop_size=256, 
            win_size=1024, 
            fmin=0, fmax=12000)
    return bigvgan_mel

def create_melvq_encoder(torchscript_path, rank, cache_dir="/opt/tiger"):
    hpath = os.path.join(torchscript_path, f"ss_encoder_{rank}.pt")
    local_path = os.path.join(cache_dir, f"ss_encoder_{rank}.pt")
    if not os.path.exists(local_path):
        if not hh.get(hpath, local_path):
            raise ConnectionError(f"Cannot retrieve file from {hpath}.")

    model = torch.jit.load(local_path, map_location=f"cuda:{rank}").eval()
    return model

def create_umm_codebook(ckpt_path, rank, cache_dir="/opt/tiger"):
    local_path = os.path.join(cache_dir, os.path.basename(ckpt_path))
    with local_zero_first():
        if not os.path.exists(local_path):
            if not hh.get(ckpt_path, local_path):
                raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            
    weight = torch.load(local_path).to(f"cuda:{rank}")
    return weight

