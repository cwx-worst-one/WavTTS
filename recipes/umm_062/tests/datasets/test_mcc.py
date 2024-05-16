import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchaudio
from tqdm import tqdm

from recipes.datasets.mcc.mix import (
    MCCInstrumentalDataset,
    MCCVocalDataset,
    MixWebDataModule,
    SodaDataModule,
    MixZhWebDataModule,
    VocalWebDataModule,
    collate_audio_text,
)
from recipes.umm_062.transforms.speech import SpeechTransform
from samantha.transforms.audio import batch_plot_spectrogram

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def interfere_audio(wav_batch):
    interfered_batch = wav_batch.clone()
    b, t = interfered_batch.size()
    primary_indices = np.random.binomial(size=b, n=1, p=0.5).astype(bool)
    for primary_i, to_mix in tqdm(enumerate(primary_indices)):
        if to_mix:
            r = np.random.uniform(-5, 5)
            secondary_i = np.random.randint(b)
            sampled_duration = np.random.randint(1, math.floor(t / 2))
            primary_start = np.random.randint(0, t - sampled_duration)
            secondary_start = np.random.randint(0, t - sampled_duration)
            primary_clip = interfered_batch[
                primary_i, primary_start : primary_start + sampled_duration
            ]
            secondary_clip = interfered_batch[
                secondary_i, secondary_start : secondary_start + sampled_duration
            ]
            scale = np.sqrt(
                ((primary_clip**2).sum() / t)
                / (((secondary_clip**2).sum() / t) * (10 ** (r / 10)))
            )
            interfered_batch[
                primary_i, primary_start : primary_start + sampled_duration
            ] = torch.stack([primary_clip, (scale * secondary_clip)], dim=0).mean(dim=0)
            print(
                f"[{primary_i}] {primary_start / 24000:.2f} - {(primary_start + sampled_duration) / 24000:.2f} | {scale:.2f}"
            )
    print("===============================")
    return interfered_batch


@pytest.mark.skip()
def test_mcc_datasets():
    batch_size = 100
    sample_rate = 24000
    mcc_instrumental = MCCInstrumentalDataset(use_pipe=True)
    mcc_vocal = MCCVocalDataset(use_pipe=True)

    for dataset in [mcc_instrumental, mcc_vocal]:
        dataset_iter = iter(dataset)
        for i in range(batch_size):
            batch = next(dataset_iter)
            audio = batch["audio"]
            text = batch.get("text", None)
            if text is not None:
                print(text)
            torchaudio.save(f"{dataset.name}-item-{i}.wav", audio, sample_rate)


def test_mix_datamodule():
    n_mels = 128
    sample_rate = 24000
    pl_datamodule = SodaDataModule(
        sample_rate=sample_rate,
        batch_size=sample_rate * 10 * 30,
        shuffle_buffer_size=10,
        num_workers=2,
        region="CN",
        weights=[1, 1, 0],
        tokenizer=None,
        collate_fn=collate_audio_text,
    )
    train_loader = pl_datamodule.train_dataloader()
    train_loader = iter(train_loader)
    for i in range(10):
        batch = next(train_loader)
        audio = batch["audio"]
        interfered_audio = interfere_audio(audio.squeeze(1)).unsqueeze(1)
        batch_size = audio.size(0)
        for a_idx, (a, i_a) in enumerate(zip(audio, interfered_audio)):
            torchaudio.save(
                f"./test_out/mix-batch-{i}-item-{a_idx}.mp3",
                a,
                sample_rate,
                format="mp3",
            )
            torchaudio.save(
                f"./test_out/mix-batch-{i}-item-{a_idx}_interfered.mp3",
                i_a,
                sample_rate,
                format="mp3",
            )

        for n_fft in [2048]:
            for win_length in [n_fft]:
                for hop_length in [sample_rate // 100]:
                    model_input_transform = SpeechTransform(
                        sample_rate=sample_rate,
                        n_mels=n_mels,
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        f_min=0,
                        f_max=sample_rate // 2,
                    )
                    text = batch["text"]
                    title = f"n_mels: {n_mels} n_fft: {n_fft} win_length: {win_length} hop_length: {hop_length}"
                    mel = model_input_transform(audio, normalize=False).transpose(1, 2)
                    fig, ax = plt.subplots(batch_size, 1, figsize=(30, 20))
                    batch_plot_spectrogram(
                        mel, plot_log=False, mel=True, title=text, ax=ax
                    )
                    ax[0].set_title(title)
                    plt.tight_layout()
                    plt.savefig(
                        f"./test_out/mix-batch-{i}-mel-{n_mels}-{n_fft}-{win_length}-{hop_length}.pdf"
                    )
                    assert mel.shape[0] == batch_size
                    assert mel.shape[1] == n_mels


test_mix_datamodule()
