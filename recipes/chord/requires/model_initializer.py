import contextlib
import os
import torch
import torch.distributed as dist
import kenlm

import samantha.utils.hdfs_helper as hh
from recipes.chord.pl_modules.pl_module import LitChord
from recipes.beat.models.perceiver import BeatPerceiverModelStage
from recipes.chord.models.classifier import ChordPerceiverClassifierStage
from samantha.core import BaseModel


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


def init_chord(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    if hpath.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(hpath)}"
        if not os.path.exists(local_path):
            if not hh.get(hpath, local_path):
                raise ConnectionError(f"Cannot retrieve file from {hpath}.")
    else:
        local_path = hpath

    with local_zero_first():
        # TODO: streamline this loading process
        perceiver_model_stage = BeatPerceiverModelStage(
            sample_rate=16000,
            sample_len=12.0,
            hop_len=500,
            n_layers=6,
            spec_dim=256,
            temporal_dim=256,
            temporal_heads=8,
            resnet_pools=[[2, 2], [2, 1]],
            n_fft=2048,
            semitone_scale=1,
            freq_pool_size=4,
            time_pool_size=2,
            num_fct=8,
        )
        chord_classifier_stage = ChordPerceiverClassifierStage(
            n_channel=2048,
            chord_pool=[2],
        )
        model = BaseModel(
            input_names=["audio", "aug_hop_size"],
            output_names=["chord_root", "chord_triad"],
            stages=[perceiver_model_stage, chord_classifier_stage],
        )
        chord_model = LitChord.load_from_checkpoint(
            local_path,
            model=model,
            lr=0.001,
            scheduler_patience=10,
            scheduler_decay_factor=0.8,
            hop_length=500,
            sample_rate=16000,
            sample_len=12.0,
            chord_pool=[2],
            resnet_pools=[[2, 2], [2, 1]],
            strict=False,
        ).to(device).eval()
        return {"chord": chord_model}


def init_chord_lms(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    chord_lms = {}
    for key, path in hpath.items():
        if path is None:
            chord_lms[key] = None
            continue
        if path.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(path)}"
            if not os.path.exists(local_path):
                if not hh.get(path, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {path}.")
        else:
            local_path = path
        chord_lms[key] = kenlm.Model(local_path)
    print(f"Loaded chord LMs: {chord_lms.keys()}")

    with local_zero_first():
        return {"chord_lms": chord_lms}


# For backward compat
def init_chord_lm(hpath, local_rank, cache_dir=None):
    return init_chord_lms({"default": hpath}, local_rank, cache_dir)