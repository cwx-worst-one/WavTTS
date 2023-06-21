import os

import numpy as np
import torch
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

import samantha.utils.hdfs_helper as hh

from ..utils.dist import local_zero_first
from recipes.musiclm.models.compat.semantic_model import SSLFrontend
from recipes.best_rq.modules.lit_module import BestRq
from recipes.best_rq.modules.lit_datamodule import NormalizeFeature
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio_augmentations import Compose


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
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
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
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    # with local_zero_first():
    #     if not os.path.exists(
    #         f"{cache_dir}/mel_mean_1000.pt"
    #     ):
    #         if not hh.get(
    #             "hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/mel_mean_1000.pt",
    #             f"{cache_dir}/mel_mean_1000.pt"
    #         ):
    #             raise ConnectionError(f"Cannot retrieve mel_mean_1000.pt.")
    #     if not os.path.exists(
    #         f"{cache_dir}/mel_std_1000.pt"
    #     ):
    #         if not hh.get(
    #             "hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/mel_std_1000.pt",
    #             f"{cache_dir}/mel_std_1000.pt"
    #         ):
    #             raise ConnectionError(f"Cannot retrieve mel_std_1000.pt.")
    # mean = torch.load(f"{cache_dir}/mel_mean_1000.pt").to(device)
    # std = torch.load(f"{cache_dir}/mel_std_1000.pt").to(device)
    mean = torch.tensor(3.287).to(device)
    std = torch.tensor(20.043).to(device)
    feature_fn = Compose(
        [
            MelSpectrogram(
                sample_rate=24000,
                n_fft=2048,
                hop_length=240,
                n_mels=128,
            ).eval().to(device),
            AmplitudeToDB().eval().to(device),
            NormalizeFeature(mean, std).eval().to(device)
        ]
    )
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        with local_zero_first():
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            model = BestRq.load_from_checkpoint(local_path).eval().to(device)
            return {
                "feature_fn": feature_fn,
                "semantic": model,
            }
    else:
        with local_zero_first():
            model = BestRq.load_from_checkpoint(hpath).eval().to(device)
            return {
                "feature_fn": feature_fn,
                "semantic": model,
            }


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
