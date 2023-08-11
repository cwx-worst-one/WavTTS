import sys
import torch
import torchaudio

from recipes.datasets.librilight import LibriLightWebDataModule
from recipes.datasets.libritts import LibriTTSWebDataModule
from recipes.soundstorm2.lightning.soundstorm import SoundStorm
from recipes.soundstorm2.lightning.soundstream import SoundStreamSpeech24k
from samantha.utils.hdfs_helper import get, hdfs_ls

if __name__ == "__main__":
    CKPT_PATH = sys.argv[1]

    sample_rate = 16000
    pl_datamodule = LibriTTSWebDataModule(
        sample_rate=sample_rate,
        batch_size=8,
        shuffle_buffer_size=100,
    )

    # pl_datamodule = LibriLightWebDataModule(
    #     sample_rate=16000,
    #     split="large",
    #     batch_size=8,
    #     shuffle_buffer_size=100,
    # )



    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    device = "cuda"

    audio_model = SoundStreamSpeech24k().to(device)
    soundstorm = SoundStorm.load_from_checkpoint(CKPT_PATH, audio_model=audio_model).to(device)

    # ground truth
    with torch.no_grad():
        audio_tokens = soundstorm.audio_model(batch["audio"].to(device))


    iterations = [48, 32, 24, 16, 8, 4, 2, 2, 1, 1, 1, 1]
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
    guidance_scale = None
    temperatures = [1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4]
    sampled_t = torch.randint(50, 100, (1,), device=soundstorm.device)

    sampled_audio_tokens, _ = soundstorm.iterative_decoding(
        max_seq_len=audio_tokens.shape[2],
        iterations=iterations,
        score_strategies=score_strategies,
        guidance_scale=guidance_scale,
        temperatures=temperatures,
        sampled_t=sampled_t,
        seed_tokens=audio_tokens,
        prefix_tokens=None,
        semantic_tokens=None,
    )

    with torch.no_grad():
        dec_audio = soundstorm.audio_model.decode(audio_tokens)
        sampled_audio = soundstorm.audio_model.decode(sampled_audio_tokens)

    torchaudio.save("ground_truth.mp3", dec_audio[0].cpu(), pl_datamodule.sample_rate)
    torchaudio.save("sampled.mp3", sampled_audio[0].cpu(), pl_datamodule.sample_rate)
