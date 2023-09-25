import matplotlib.pyplot as plt
import pytest
import torchaudio

from recipes.datasets.mcc.mix import (
    MCCInstrumentalDataset,
    MCCVocalDataset,
    MixWebDataModule,
    MixZhWebDataModule,
    VocalWebDataModule,
    collate_audio_text,
)
from recipes.umm.transforms.speech import SpeechTransform
from samantha.transforms.audio import batch_plot_spectrogram

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

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
    pl_datamodule = MixZhWebDataModule(
        sample_rate=sample_rate,
        batch_size=sample_rate * 5 * 30,
        shuffle_buffer_size=10,
        num_workers=2,
        region="CN",
        weights=[1, 1, 0],
        tokenizer=None,
        collate_fn=collate_audio_text,
    )
    train_loader = pl_datamodule.train_dataloader()
    train_loader = iter(train_loader)
    for i in range(3):
        batch = next(train_loader)
        audio = batch["audio"]
        batch_size = audio.size(0)
        for a_idx, a in enumerate(audio):
            torchaudio.save(
                f"./test_out/mix-batch-{i}-item-{a_idx}.mp3", a, sample_rate, format="mp3"
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
