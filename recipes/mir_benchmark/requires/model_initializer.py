import contextlib
import os
import torch
import torch.distributed as dist

import samantha.utils.hdfs_helper as hh
from recipes.mir_benchmark.models.frontend import Frontend
from recipes.icassp.models.musicfm_25Hz import MusicFM25Hz
from recipes.mir_benchmark.models.beat_probing import BeatProbingStage
from recipes.mir_benchmark.pl_modules.beat_finetune_pl import LitFinetuneBeat
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


def init_beat(hpath, local_rank, cache_dir, model_batch_size=32):
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
        frontend = Frontend(
            model=MusicFM25Hz(
                encoder_depth=12,
                is_flash=False,
                stat_path="/mnt/bn/audio-diffusion/pretrained_models/musicfm/playlist_classic_stats.json",
                model_path="/mnt/bn/audio-diffusion/pretrained_models/musicfm/icassp/musicfm_25hz_playlist_330m_520k.pt",
            ),
            layer_ix=12,
            is_update=True,
        )
        backend = BeatProbingStage(n_channel=1024, resample=300)
        model = BaseModel(
            input_names=["audio"],
            output_names=["output"],
            stages=[frontend, backend],
        )
        beat_model = LitFinetuneBeat(
            model=model,
            sample_len=6.0,
            pretrained_path=local_path,
        ).to(device).eval()
        return {"beat": beat_model}