import contextlib
import os
import torch
import torch.distributed as dist

import samantha.utils.hdfs_helper as hh
from recipes.structure.pl_modules.pl_module import LitStructure
from recipes.structure.models.get_models import get_specTNT
from recipes.beat.models.perceiver import BeatPerceiverModelStage
from recipes.structure.models.classifier import StructureClassifierStage
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


# hpath="hdfs://harunava/home/byte_speech_sv/amy/log/structure/new_galaxyark/checkpoints/epoch=185-step=93000.ckpt"
def init_structure(hpath, local_rank, cache_dir=None, n_top_bound=14):
    """ Note: the current structure model is VERY sensitive to `n_top_bound`,
    which is used by the post-processor. The default value here was tuned for
    30s input; for longer input (e.g., full songs), it likely needs to be lower.
    """
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
        model = get_specTNT(
            sample_len=36.0,
            sample_rate=16000,
            resnet_pool_sizes=[[2, 3], [2, 1], [1, 2]],
            n_fft=1024,
            depth=5,
            temporal_dim=128,
            temporal_heads=8,
            spec_dim=96,
            spec_heads=4,
            n_boundary=2,
            n_function=7,
        )
        structure_model = LitStructure(
            model,
            sample_len=36.0,
            sample_hop=9,
            n_top_bound=n_top_bound,
            pretrain_path=local_path,
        ).to(device).eval()
        return {"structure": structure_model}


# hpath="hdfs://harunava/home/byte_speech_sv/amy/log/icassp/structure_perceiver3_other_resso/checkpoints/epoch=203-step=102000.ckpt"
def init_structure_90M(hpath, local_rank, cache_dir=None, n_top_bound=18):
    """ Note: the current structure model is VERY sensitive to `n_top_bound`,
    which is used by the post-processor. The default value here was tuned for
    30s input; for longer input (e.g., full songs), it likely needs to be lower.
    """
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
            sample_len=24.0,
            hop_len=512,
            n_layers=6,
            spec_dim=256,
            temporal_dim=256,
            temporal_heads=8,
            resnet_pools=[[2, 3], [2, 1], [1, 2]],
            n_fft=1024,
            semitone_scale=1,
            freq_pool_size=4,
            time_pool_size=6,
            num_fct=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
        structure_classifier_stage = StructureClassifierStage(
            n_channel=2048,
            n_boundary=1,
            n_function=7,
        )
        model = BaseModel(
            input_names=["audio", "aug_hop_size"],
            output_names=["boundary_pred", "function_pred"],
            stages=[perceiver_model_stage, structure_classifier_stage],
        )
        structure_model = LitStructure(
            model,
            sample_len=24.0,
            sample_hop=3,
            n_top_bound=n_top_bound,
            pretrain_path=local_path,
        ).to(device).eval()
        return {"structure": structure_model}
