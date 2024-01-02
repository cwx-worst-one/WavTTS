import contextlib
import os
import torch
import torch.distributed as dist
import samantha.utils.hdfs_helper as hh


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


def _ensure_ckpt_is_local(target_path, cache_dir):
    """If the ckpt path is on HDFS then download it to a local cache, otherwise use the filepath directly."""
    if target_path.startswith("hdfs://"):
        local_path = f"{cache_dir}/{os.path.basename(target_path)}"
        if not os.path.exists(local_path):
            hh.get(target_path, local_path)
            assert os.path.exists(
                local_path
            ), f"Could not retrieve file from {target_path}."
        return local_path
    else:
        return target_path


def init_soundstorm(hpath, local_rank, cache_dir=None):
    from recipes.soundstorm.lightning.soundstorm import SoundStorm

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    with local_zero_first():
        local_path = _ensure_ckpt_is_local(hpath, cache_dir)
        model = SoundStorm.load_from_checkpoint(local_path).to(device).eval()
        return {"soundstorm": model}


@torch.no_grad()
def run_soundstorm(requires, samples, params):
    soundstorm_model = requires["soundstorm"]

    iterations = params.get("iterations")
    score_strategies = params.get("score_strategies")
    temperatures = params.get("temperatures")
    max_seq_len = samples.shape[-1] * soundstorm_model.semantic_to_audio_rate

    sampled_audio_tokens, _ = soundstorm_model.iterative_decoding(
        samples,
        max_seq_len=max_seq_len,
        iterations=iterations,
        score_strategies=score_strategies,
        temperatures=temperatures,
    )
    sampled_audio = soundstorm_model.audio_model.decode(sampled_audio_tokens)
    if len(sampled_audio.shape) == 3:
        sampled_audio = sampled_audio.squeeze(1)
    return sampled_audio