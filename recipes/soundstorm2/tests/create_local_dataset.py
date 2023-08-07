import os

import torch
import webdataset as wds
from pyarrow.fs import FileSystem
from tqdm import tqdm
from webdataset import TarWriter

from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from recipes.soundstorm.lightning.soundstorm import SoundStorm
from samantha.utils.hdfs_helper import hdfs_ls


@torch.no_grad()
def transfer_shard(soundstorm, input_url: str, output_uri: str) -> None:
    output_fs = FileSystem.from_uri(output_uri)[0]

    n_samples = 720000
    transform = MusicLMTransforms(
        n_samples, audio_key="acc.npy", sample_range_key="acc_sample_range.npy"
    )

    n_transforms_per_item = 5

    with output_fs.open_output_stream(output_uri) as outstream:
        writer = TarWriter(outstream)
        audios = []
        for item in tqdm(
            wds.WebDataset(f"pipe:hdfs dfs -cat {input_url}").decode(),
            desc="Reading input shard...",
        ):
            song_id = item["__key__"]

            for t_idx in range(n_transforms_per_item):
                t_item = next(iter(transform(item)))
                audios.append(
                    {"audio.npy": t_item["audio.npy"], "__key__": f"{song_id}-{t_idx}"}
                )

        for item in tqdm(audios, "Adding to target shard..."):
            sample = {}

            sample["__key__"] = item["__key__"]
            audio = item["audio.npy"].to(soundstorm.device)

            semantic_tokens = soundstorm.semantic_model(audio)
            audio_tokens = soundstorm.audio_model(audio)

            sample["semantic_tokens.npy"] = semantic_tokens.cpu().numpy()
            sample["audio_tokens.npy"] = audio_tokens.cpu().numpy()
            sample["audio.npy"] = audio.cpu().numpy()
            writer.write(sample)
        writer.close()


if __name__ == "__main__":
    num_samples = 1000

    inputs = hdfs_ls(
        "hdfs://harunava/home/byte_speech_sv/data/karaoke_for_singsong/shards-*.tar"
    )

    outputs = [
        f"hdfs://harunava/home/byte_speech_sv/data/karaoke_for_soundstorm/{os.path.basename(shard)}"  # noqa
        for shard in inputs
    ]

    soundstorm = SoundStorm(24000, 8, 1, 1, 1, False, False, None, None)
    soundstorm = soundstorm.to("cuda")

    for i, o in tqdm(zip(inputs, outputs), total=len(inputs)):
        if i == o:
            print(f"Skipping {i}, same as {o}")

        transfer_shard(soundstorm, i, o)
