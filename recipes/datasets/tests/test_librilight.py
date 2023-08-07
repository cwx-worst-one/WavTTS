import pytest
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

from recipes.umm.datasets.librilight import LibriLightWebDataModule, SAMPLE_RATE
from recipes.umm.transforms.speech import SpeechTransform
from samantha.transforms.audio import batch_plot_spectrogram
from pytorch_lightning import seed_everything

@pytest.mark.skip()
def test_librilight_datamodule():
    batch_size = 8
    pl_datamodule = LibriLightWebDataModule(
        split="medium",
        batch_size=batch_size,
        shuffle_buffer_size=100,
    )

    train_loader = pl_datamodule.train_dataloader()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        assert "audio" in batch
        if batch_idx > 5:
            break

        audio = batch["audio"]
        assert audio.shape[0] == batch_size
        assert audio.shape[1] == 1

        for a_idx, a in enumerate(audio):
            torchaudio.save(f"librilight-batch-{batch_idx}-item-{a_idx}-test.mp3", a, pl_datamodule.sample_rate)

@pytest.mark.parametrize("bucket_size", [2, 5, 10])
def test_mel_spectrogram(bucket_size):
    batch_size = 16
    n_mels = 80

    pl_datamodule = LibriLightWebDataModule(
        split="medium",
        batch_size=batch_size,
        shuffle_buffer_size=100,
        num_workers=8,
        buckets_sec = [
            bucket_size,
        ],
    )
    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    audio = batch["audio"]
    for a_idx, a in enumerate(audio):
        torchaudio.save(f"librilight-bucketsize-{bucket_size}-item-{a_idx}.mp3", a, pl_datamodule.sample_rate)

    for n_fft in [512, 1024, 2048]:
        for win_length in [400]:
            for hop_length in [160]:
                model_input_transform = SpeechTransform(
                    sample_rate=SAMPLE_RATE,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    f_min=0,
                    f_max=SAMPLE_RATE//2
                )

                title = f"n_mels: {n_mels} n_fft: {n_fft} win_length: {win_length} hop_length: {hop_length}"
                mel = model_input_transform(audio)
                fig, ax = plt.subplots(batch_size, 1, dpi=1200)
                batch_plot_spectrogram(mel, plot_log=False, mel=True, ax=ax)
                ax[0].set_title(title)
                plt.tight_layout()
                plt.savefig(f"librilight-bucketsize-{bucket_size}-mel-{n_mels}-{n_fft}-{win_length}-{hop_length}.pdf")
                assert mel.shape[0] == batch_size
                assert mel.shape[1] == 1
                assert mel.shape[2] == n_mels


# Testing spectrogram parameters from https://cdn.openai.com/papers/whisper.pdf
@pytest.mark.skip()
def test_librilight_datamodule_mel_transform():
    n_mels = 80
    n_fft = 512
    win_length = 400 # 25ms at 16kHz
    hop_length = 160 # 10ms at 16kHz

    batch_size = 8

    title = f"n_mels: {n_mels} n_fft: {n_fft} win_length: {win_length} hop_length: {hop_length}"

    pl_datamodule = LibriLightWebDataModule(
        split="medium",
        batch_size=batch_size,
        shuffle_buffer_size=100,
        num_workers=8,
        buckets_sec = [
            10,
        ],
    )

    model_input_transform = SpeechTransform(
        sample_rate=SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=0,
        f_max=SAMPLE_RATE//2
    )
    train_loader = pl_datamodule.train_dataloader()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        assert "audio" in batch
        if batch_idx > 5:
            break
        
        audio = batch["audio"]
        mel = model_input_transform(audio)
        fig, ax = plt.subplots(batch_size, 1, figsize=(20,10))
        batch_plot_spectrogram(mel, plot_log=False, mel=True, ax=ax)
        ax[0].set_title(title)
        plt.tight_layout()
        plt.savefig(f"librilight-batch-{batch_idx}-mel-{n_mels}-{n_fft}-{win_length}-{hop_length}.pdf")
        assert mel.shape[0] == batch_size
        assert mel.shape[1] == 1
        assert mel.shape[2] == n_mels
        # for a_idx, a in enumerate(audio):
        #     torchaudio.save(f"librilight-batch-{batch_idx}-{n_mels}-{n_fft}-{win_length}-{hop_length}-item-{a_idx}.mp3", a, pl_datamodule.sample_rate)

