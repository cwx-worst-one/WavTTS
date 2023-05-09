from functools import partial

import torchaudio
import webdataset as wds

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.stems import StemsTransforms
from samantha.dataio.webdataset import select_keys


def test_stem_transform():
    sample_rate = 24000
    n_samples = sample_rate * 10
    batch_size = 10
    transforms = StemsTransforms(
        n_samples=n_samples,
        audio_key="acc.npy",
        sample_range_key="acc_sample_range.npy",
        cond_audio_key="vocal.npy",
        cond_sample_range_key="vocal_sample_range..npy",
        min_volume_threshold=0.1,
    )

    karaoke_preprocessor = WebDatasetBufferPreprocessor(
        sample_rate=sample_rate, transforms=transforms
    )

    karaoke_train_dataset = wds.WebDataset(
        urls="pipe:hdfs dfs -cat hdfs://haruna/home/byte_speech_sv/data/karaoke_for_singsong/shards-{0000..0130}.tar"  # noqa
    )
    karaoke_train_dataset = (
        karaoke_train_dataset.select(
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
        .shuffle(10)
        .decode()
        .compose(karaoke_preprocessor.train_buffer_preprocessor)
        .to_tuple("audio.npy", "cond_audio.npy")
        .batched(batch_size)
    )

    for batch in karaoke_train_dataset:
        audio, cond_audio = batch

        for idx, (acc, vocal) in enumerate(zip(audio, cond_audio)):
            mix = (acc + vocal) / 2
            torchaudio.save(f"mix-{idx}.mp3", mix, sample_rate)

        break
