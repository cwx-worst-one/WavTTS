from functools import partial

import torch
import webdataset as wds
from tqdm import tqdm
from webdataset import TarWriter

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.musiclm import MusicLMTransforms
from recipes.soundstorm.lightning.soundstorm import SoundStorm
from samantha.utils.datastructures import select_keys


def example_webdataset(sample_rate: int, n_audio_samples: int, batch_size: int = 1):
    transforms = MusicLMTransforms(
        n_samples=n_audio_samples,
        audio_key="acc.npy",
        sample_range_key="acc_sample_range.npy",
    )

    preprocessor = WebDatasetBufferPreprocessor(
        sample_rate=sample_rate, transforms=transforms
    )

    dataset = wds.WebDataset(
        urls="pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/karaoke_for_singsong/shards-{0000..0130}.tar"  # noqa
    )
    dataset = (
        dataset.select(
            partial(
                select_keys,
                keys=[
                    "acc.npy",
                    "acc_sample_range.npy",
                    "vocal.npy",
                    "vocal_sample_range..npy",
                ],
            )
        )
        .decode()
        .compose(preprocessor.train_buffer_preprocessor)
        .to_tuple("audio.npy")
        .batched(batch_size)
    )
    return dataset


def create_test_dataset(num_samples: int):
    dataset = example_webdataset(
        sample_rate=24000, n_audio_samples=24000 * 30, batch_size=32
    )

    soundstorm = SoundStorm(24000, 8, 1, 1, 1, False, False, None, None)
    soundstorm = soundstorm.to("cuda")
    fn = "test_dataset.tar"
    writer = TarWriter(fn)
    idx = 0
    pbar = tqdm(total=num_samples)
    for batch in dataset:
        if idx > num_samples:
            break

        sample = {}
        audio = batch[0].to("cuda")

        with torch.no_grad():
            semantic_tokens = soundstorm.semantic_model(audio)
            audio_tokens = soundstorm.audio_model(audio)

        for st, at, a in zip(semantic_tokens, audio_tokens, audio):
            sample["__key__"] = str(idx)
            sample["semantic_tokens.npy"] = st.cpu().numpy()
            sample["audio_tokens.npy"] = at.cpu().numpy()
            sample["audio.npy"] = a.cpu().numpy()
            writer.write(sample)
            idx += 1
            pbar.update(1)

    writer.close()
    print(f"Written {idx} samples to {fn}")


if __name__ == "__main__":
    num_samples = 1000
    create_test_dataset(num_samples)
