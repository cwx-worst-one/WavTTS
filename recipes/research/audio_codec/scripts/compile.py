import os
from argparse import ArgumentParser
from pprint import pprint
from typing import Optional

import torch

from recipes.research.audio_codec import AudioCodec
from samantha.dataio.data_bucket import data_bucket
from samantha.utils.hdfs_helper import hdfs_ls


def get_commit_checkpoints(log_name: str, commit_hash: str):
    path = data_bucket(f"logs/{log_name}/{commit_hash}/*/*/checkpoints")
    ckpts = hdfs_ls(path)
    ckpts = list(
        filter(
            lambda f: ".ckpt" in f and "_COPYING_" not in f and "last" not in f, ckpts
        )
    )
    ckpts = sorted(
        ckpts,
        key=lambda k: int(k.split("/")[-1].replace("step=", "").replace(".ckpt", "")),
    )
    return ckpts


def get_latest_model_from_commit(log_name: str, commit_hash: str):
    ckpts = get_commit_checkpoints(log_name, commit_hash)
    latest_ckpt = ckpts[-1]
    print(f"latest checkpoint: {latest_ckpt}")
    return latest_ckpt


def get_commit_log_folder(commit_hash: str, date: str = "*", time: str = "*"):
    path = data_bucket(f"logs/*/*/{commit_hash}/{date}/{time}")
    folders = hdfs_ls(path)
    if len(folders):
        folder = "/".join(folders[0].split("/")[:-1])
        return folder
    return None


def get_audio_codec(
    commit_hash: str, ckpt_path: Optional[str] = None, strict: bool = True
):
    if ckpt_path is None:
        ckpt_path = get_latest_model_from_commit("audio_codec/default", commit_hash)

    audio_codec = AudioCodec.load_from_checkpoint(ckpt_path, strict=strict)
    audio_codec = audio_codec.eval()
    print(audio_codec.summarize())
    pprint(audio_codec.config)
    audio_codec.commit_hash = commit_hash
    audio_codec.commit_step = os.path.basename(ckpt_path)
    return audio_codec


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--commit_hash", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    args = parser.parse_args()

    audio_codec = get_audio_codec(args.commit_hash, args.ckpt_path)
    audio_codec = audio_codec.to("cuda")

    x = torch.randn(
        1,
        audio_codec.config.n_channels,
        audio_codec.config.sample_rate * 10,
        device=audio_codec.device,
    )

    with torch.no_grad():
        z = audio_codec.get_z(x, audio_codec.config.sample_rate)
        x_pred = audio_codec.decode_z(z)

    audio_codec = audio_codec.to_torchscript()
    audio_codec.save(
        f"uac-{audio_codec.commit_hash}-{audio_codec.commit_step}-latent={audio_codec.latent_dim}.pt"
    )
