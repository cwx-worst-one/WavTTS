from functools import partial

import webdataset as wds
from tqdm import tqdm

from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from recipes.musiclm.transforms.stems import StemsTransforms
from sami_ai.dataio.webdataset import select_keys


def test_stem_transform():
    sample_rate = 24000
    n_samples = sample_rate * 10
    batch_size = 10
    transforms = StemsTransforms(
        n_samples=n_samples,
        source_audio_key="vocal.npy",
        source_sample_range_key="vocal_sample_range..npy",
        target_audio_key="acc.npy",
        target_sample_range_key="acc_sample_range.npy",
        min_db=-25.0,
        relative_db=5.0,
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
        .shuffle(100)
        .decode()
        .compose(karaoke_preprocessor.train_buffer_preprocessor)
        .to_tuple("source_audio.npy", "target_audio.npy")
        .batched(batch_size)
    )

    for batch in tqdm(karaoke_train_dataset):
        source_audio, target_audio = batch

        # for idx, (source, target) in enumerate(zip(source_audio, target_audio)):
        #     mix = (source + target) / 2
        #     torchaudio.save(f"mix-{idx}.mp3", mix, sample_rate)

        # break
