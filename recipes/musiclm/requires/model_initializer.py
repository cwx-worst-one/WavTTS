import os

import numpy as np
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchaudio_augmentations import Compose
from transformers import AutoModel, Wav2Vec2FeatureExtractor

import samantha.utils.hdfs_helper as hh
from recipes.best_rq.modules.lit_datamodule import NormalizeFeature
try:
    from recipes.best_rq.modules.lit_module import BestRQ
except:
    BestRQ = None
from recipes.musiclm.models.compat.semantic_model import SSLFrontend
from functools import partial, lru_cache
from ..utils.dist import local_zero_first


def value(func: str):
    exec_data = {}
    exec(f"ret = {func}", globals(), exec_data)
    return exec_data["ret"]


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module

def init_mulan(hpath, local_rank, cache_dir=None, version="149", prefix=""):
    return _init_cached_mulan(hpath, local_rank, cache_dir, version, prefix)

@lru_cache() # cache initialization so we don't load multiple mulan's (e.g. during inference with semantic and reranker modules)
def _init_cached_mulan(hpath, local_rank, cache_dir=None, version="149", prefix=""):
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
    elif version in ["filmgen"]:
        from .mulan.mulan_infer_filmgen import (
            create_mulan_model,
            mulan_inference,
        )
        mulan_rvq_indexs = None
    elif version in ["Chinese"]:
        from .mulan.mulan_infer_chinese import(
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    elif version in ["sstk"]:
        from .mulan.mulan_infer_sstk import(
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
    elif version in ["sstkmae", "sstkmae_v2", "sstkmae_v3"]:
        from .mulan.mulan_infer_sstk_mae import(
            create_mulan_model,
            mulan_inference,
            mulan_rvq_indexs,
        )
        if version == "sstkmae":
            create_mulan_model = partial(create_mulan_model, version="v1")
        elif version == "sstkmae_v2":
            create_mulan_model = partial(create_mulan_model, version="v2")
            mulan_inference = partial(mulan_inference, normalize_text=True)
        else:
            create_mulan_model = partial(create_mulan_model, version="v2")
            mulan_inference = partial(mulan_inference, normalize_text=False)

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
                f"{prefix}mulan": create_mulan_model(local_path, device=device),
                f"{prefix}mulan_infer_fn": mulan_inference,
                f"{prefix}mulan_rvq_fn": mulan_rvq_indexs,
            }
    else:
        local_path = hpath
        with local_zero_first():
            return {
                f"{prefix}mulan": create_mulan_model(local_path, device=device),
                f"{prefix}mulan_infer_fn": mulan_inference,
                f"{prefix}mulan_rvq_fn": mulan_rvq_indexs,
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
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    return {"semantic": model, "processor": processor}


def init_wav2vec(hpath, local_rank, cache_dir=None):
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
            "ssl_frontend": SSLFrontend(),
            "semantic": load_torch_script_module(local_path, device),
        }
    else:
        with local_zero_first():
            return {
                "ssl_frontend": SSLFrontend(),
                "semantic": load_torch_script_module(hpath, device),
            }


def init_best_rq(hpath, local_rank, cache_dir=None):
    try:
        from recipes.best_rq.modules.lit_module import BestRQ
    except:
        BestRQ = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            model = BestRQ.load_from_checkpoint(local_path).eval().to(device)
            return {"semantic": model}
    else:
        with local_zero_first():
            model = BestRQ.load_from_checkpoint(hpath).eval().to(device)
            return {"semantic": model}

def init_best_rq_minz(hpath, local_rank, cache_dir=None):
    from recipes.best_rq.models.chromatic_t5_rq import BEST_RQ_SCRIPT
    device = torch.device(f"cuda:{local_rank}")
    model = BEST_RQ_SCRIPT(device)
    model = model.to(device)
    model = model.half()
    model.eval()

    return {
        "semantic": model
    }

# def init_best_rq_minz(hpath, local_rank, cache_dir=None):
#     from recipes.best_rq.models.chromatic_t5_rq import BEST_RQ

#     device = torch.device(f"cuda:{local_rank}")
#     model = BEST_RQ(
#         codebook_dim=16,
#         codebook_size=8192,
#         hop_length=240,
#         n_mels=128,
#         conv_dim=512,
#         encoder_dim=1024,
#         encoder_depth=24,
#         mask_hop=0.4,
#         mask_prob=0.5,
#         is_flash=True,
#         global_mean=16.4,
#         global_std=14.7,
#         is_torchscript=True,
#     )
#     S = torch.load(
#         "/mnt/bn/audio-diffusion/pretrained_models/best_rq/chromatic_80k.pt"
#     )["state_dict"]
#     SS = {k[6:]: v for k, v in S.items()}
#     model.load_state_dict(SS, strict=False)

#     model = model.to(device)
#     model = model.half()
#     model.eval()
#     return {"semantic": model}
def init_semantic_centers(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            semantic_centers = np.load(local_path)
            semantic_centers = torch.from_numpy(semantic_centers).float().to(device)
            return {"semantic_centers": semantic_centers}
    else:
        with local_zero_first():
            semantic_centers = np.load(hpath)
            semantic_centers = torch.from_numpy(semantic_centers).float().to(device)
            return {"semantic_centers": semantic_centers}


def init_semantic_cmvn(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            semantic_cmvn = np.load(local_path)
            semantic_cmvn = torch.from_numpy(semantic_cmvn).float().to(device)
            return {"semantic_cmvn": semantic_cmvn}
    else:
        with local_zero_first():
            semantic_cmvn = np.load(hpath)
            semantic_cmvn = torch.from_numpy(semantic_cmvn).float().to(device)
            return {"semantic_cmvn": semantic_cmvn}


def init_soundstream(hpath, local_rank, cache_dir=None):
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


def init_soundstream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"

    if not os.path.exists(local_path):
        if not hh.get(h_ss, local_path):
            raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
    return {"ss_dec": load_torch_script_module(local_path, device)}

def init_t5_encoder(hpath, local_rank, cache_dir=None):
    from transformers import T5Tokenizer, T5Model, T5EncoderModel
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    # loading our model weights
    model = (
        T5EncoderModel.from_pretrained(
            "t5-small", cache_dir=cache_dir
        )
        .eval()
        .to(device)
    )
    return {"t5": model}
