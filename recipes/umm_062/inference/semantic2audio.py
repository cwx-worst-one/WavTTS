import torch
import torchaudio

from recipes.soundstorm2.lightning.bestrq import BestRQMelCTCModel
from recipes.soundstorm2.lightning.soundstorm import SoundStorm
from recipes.soundstorm2.lightning.soundstream import SoundStreamSpeech24k
from recipes.umm_062.modules.semantic_module import TextSemanticStage

if __name__ == "__main__":
    device = "cuda"
    semantic_stage_ckpt_path = "step=030000-loss=0.0000-val_loss_0=0.0000.ckpt"
    soundstorm_ckpt_path = "last.ckpt"
    semantic_temperature = 0.4

    # Text - Semantic
    semantic_stage = TextSemanticStage.load_from_checkpoint(
        semantic_stage_ckpt_path
    ).to(device)

    # SoundStorm
    audio_model = SoundStreamSpeech24k()
    semantic_model = BestRQMelCTCModel()
    soundstorm = SoundStorm.load_from_checkpoint(
        soundstorm_ckpt_path, audio_model=audio_model, semantic_model=semantic_model
    ).to(device)

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

    text = ["hello there"]
    semantic_tokens = semantic_stage.generate(
        text, max_seq_len=250, temperature=semantic_temperature
    )

    max_seq_len = semantic_tokens.shape[1] * 2
    audio_tokens, _ = soundstorm.iterative_decoding(
        max_seq_len=max_seq_len,
        iterations=iterations,
        score_strategies=score_strategies,
        semantic_tokens=semantic_tokens,
        guidance_scale=guidance_scale,
        temperatures=temperatures,
        sampled_t=None,
    )

    with torch.no_grad():
        audio = soundstorm.audio_model.decode(audio_tokens)

    torchaudio.save("test.mp3", audio[0].cpu(), 24000)
