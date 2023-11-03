import contextlib
import os

import torch
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


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_pretrained_bestrq(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        return {"pretrained_bestrq_state": state_dict}


def init_pretrained(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        return {"state_dict": state_dict}


def init_rmvpe(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))
        return {"state_dict": state_dict}


def init_bestrq_mel_ctc_vq(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import BestRQMelCTC

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = BestRQMelCTC.load_from_checkpoint(local_path).to(device).eval()
        return {"BestRQMelCTCVQ": model}


def init_mulan(hpath, local_rank, cache_dir=None):
    from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
        create_mulan_model,
        mulan_inference,
    )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }
        else:
            local_path = hpath
            return {
                "mulan": create_mulan_model(local_path, device=device),
                "mulan_infer_fn": mulan_inference,
            }


def init_stage1(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage1

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = Stage1.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage1": model}


def init_stage2(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage2

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = Stage2.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage2": model}


def init_stage3(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import Stage3

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                hh.get(hpath, local_path)
                print("FIXME: hh.get always returns non-zero exit code")
                # if not hh.get(hpath, local_path):
                #     raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = Stage3.load_from_checkpoint(local_path).to(device).eval()
        return {"Stage3": model}


def init_mkii(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import MKIIVQ

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = MKIIVQ.load_from_checkpoint(local_path).to(device).eval()
        return {"mkii": model}


def init_soundstream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        h_ss = f"{hpath}/ss{local_rank}.pt"
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/ss{local_rank}.pt"
            if not os.path.exists(local_path):
                if not hh.get(h_ss, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        else:
            local_path = h_ss
    return {"ss": load_torch_script_module(local_path, device)}


def init_soundstream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"
            if not os.path.exists(local_path):
                if not hh.get(h_ss, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        else:
            local_path = h_ss
    return {"ss_dec": load_torch_script_module(local_path, device)}


def init_sami_tts_api(fe_version, fe_task, cache_dir="/opt/tiger/sami_tts_api/models"):
    from sami_tts_api.sail import download_model

    model_name = f"{fe_task}__{fe_version}.model"
    local_path = f"{cache_dir}/{model_name}"
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    with local_zero_first():
        if not os.path.exists(local_path):
            fe = download_model(fe_task, cache_dir, fe_version)
            assert fe == local_path
    return local_path

    # model_name = f"{fe_task}__{fe_version}.model"
    # hdfs_path = f"hdfs:///home/byte_speech_sv/zongyu.yin/sami_tts_api/models/{model_name}"
    # local_path = f"{cache_dir}/{model_name}"
    # with local_zero_first():
    #     if not os.path.exists(local_path):
    #         if not hh.get(hdfs_path, local_path):
    #             raise ConnectionError(f"Cannot retrieve file from {hdfs_path}.")
    # return local_path


def init_unified_decoder(hpath, local_rank, cache_dir=None):
    from recipes.umm.modules.lit_module import UnifiedDecoder

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        model = UnifiedDecoder.load_from_checkpoint(local_path).to(device).eval()
        return {"unified_decoder": model}
