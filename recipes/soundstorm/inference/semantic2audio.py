import sys
from time import perf_counter
from typing import List, Optional

import torch
import torchaudio
import webdataset as wds

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from recipes.soundstorm.lightning.soundstorm import SoundStorm, SoundStormInference


def load_model(
    soundstorm_path: str, semantic_ckpt_path: str, device: str, **kwargs
) -> SoundStorm:
    return SoundStormInference(soundstorm_path, semantic_ckpt_path, **kwargs).to(device)


def karaoke_validation_dataset(sample_rate: int, n_audio_samples: int, batch_size: int):
    transforms = MusicLMTransforms(
        n_samples=n_audio_samples,
        audio_key="acc.npy",
        sample_range_key="acc_sample_range.npy",
    )

    preprocessor = WebDatasetBufferPreprocessor(
        sample_rate=sample_rate, transforms=transforms
    )

    dataset = wds.WebDataset(
        urls="pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/karaoke_for_singsong/shards-0131.tar"
    )
    dataset = (
        dataset.decode()
        .compose(preprocessor.train_buffer_preprocessor)
        .to_tuple("audio")
        .batched(batch_size)
    )
    return dataset


if __name__ == "__main__":
    device = "cuda"
    ckpt_path = sys.argv[1]
    sample_rate = 24000
    n_audio_samples = sample_rate * 10
    batch_size = 1
    n_examples = 10
    warmup = 2

    dataset = karaoke_validation_dataset(sample_rate, n_audio_samples, batch_size)
    soundstorm = load_model(ckpt_path, device, n_audio_samples=n_audio_samples)

    rtfs = []
    duration_sec = n_audio_samples / sample_rate
    for idx, batch in enumerate(dataset):
        if idx == n_examples:
            break

        batch = tuple(map(lambda a: a.to(device), batch))

        semantic_tokens, audio_tokens, audio = soundstorm.prepare_inputs(batch)

        tik = perf_counter()
        sampled_t = None
        guidance_scale = None
        iterations = [32, 32, 32, 32, 8, 8, 8, 8, 8, 8, 8, 8]
        score_strategies = [
            "random",
            "random",
            "random",
            "random",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
        ]
        temperatures = [1.0] * len(iterations)
        semantic_temperature = 0.9
        sampled_audio = soundstorm.generate(
            soundstorm,
            semantic_tokens,
            max_seq_len=audio_tokens.shape[2],
            iterations=iterations,
            score_strategies=score_strategies,
            guidance_scale=guidance_scale,
            sampled_t=sampled_t,
            temperatures=temperatures,
            semantic_temperature=semantic_temperature,
        )

        tok = perf_counter()
        rtf = (tok - tik) / duration_sec

        if idx >= warmup:
            rtfs.append(rtf)
            print(f"RTF: {rtf}")

        # torchaudio.save(f"{idx}-input.mp3", audio[0].cpu(), 24000)
        torchaudio.save(f"{idx}-sampled.mp3", sampled_audio[0].cpu(), 24000)
        # for quant_idx in range(0, 12):
        #     token_subset = torch.cat((pred_tokens[:, :quant_idx], audio_tokens[:, quant_idx:]), dim=1)

        #     with torch.no_grad():
        #         subset_sampled_audio = soundstorm.audio_model.decode(token_subset)
        #     torchaudio.save(f"{idx}-{quant_idx}-sampled.mp3", subset_sampled_audio[0].cpu(), 24000)

    print(f"Mean RTF: {sum(rtfs) / len(rtfs)}")
