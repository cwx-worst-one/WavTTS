import os

import torchaudio
from tqdm import tqdm

from recipes.audio_lm.utils.utils import slugify
from recipes.musiclm.utils.prompts import gather_text_prompts
from recipes.soundstorm.lightning.soundstorm import SoundStormInference

if __name__ == "__main__":
    device = "cuda"
    sample_rate = 24000
    n_audio_samples = sample_rate * 10
    batch_size = 1

    default_ckpt = "/mnt/bn/audio-diffusion/logs/soundstorm/mcc40m-normalized/soundstorm/mcc40m/checkpoints/step=125000-loss=0.0000.ckpt"
    default_semantic_ckpt = "/mnt/bn/audio-diffusion/ckpts/musiclm/semantic_flash_llama/mcc40m/step=040000-tr_loss=2.3858.ckpt"

    prompts, categories = gather_text_prompts(
        "sami", "Acoustic guitar. Funky jazz.", batch_size=batch_size
    )
    iterations = [64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 2]
    temperatures = [1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4]
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
    soundstorm = SoundStormInference(default_ckpt, default_semantic_ckpt)
    soundstorm = soundstorm.to("cuda")

    # ckpt2 = "/mnt/bn/audio-diffusion/logs/soundstorm/2829482/output/soundstorm/baseline/checkpoints/epoch=0-step=105000.ckpt"
    # soundstorm2 = SoundStormInference(ckpt2, default_semantic_ckpt)
    # soundstorm2 = soundstorm2.to("cuda")
    # del soundstorm2.semantic_model
    # del soundstorm2.requires

    semantic_temperature = 0.9
    guidance_scale = None
    category = "tiktok_top100"
    # for prompt, category in tqdm(zip(prompts, categories)):
    prompts = [
        "pop music",
        "rock music",
        "jazz music",
        "hip hop rap music",
        "electronic music",
        "country music",
        "alternative indie music",
        "r&b soul music",
        "new age music",
    ]
    for prompt in prompts:
        prompt = [prompt] * batch_size
        mulan_tokens = soundstorm.sample_mulan_tokens(prompt)
        semantic_tokens = soundstorm.sample_semantic_tokens(
            mulan_tokens, duration=10, temperature=semantic_temperature
        )

        audio, _ = soundstorm.semantic2audio(
            semantic_tokens,
            500,
            iterations,
            score_strategies,
            guidance_scale,
            None,
            temperatures,
        )
        d = slugify(category)
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, slugify(prompt))

        for idx, a in enumerate(audio):
            torchaudio.save(f"{fn}-0-{idx}.mp3", a.cpu(), sample_rate)
            # torchaudio.save(f"{fn}-1-{idx}.mp3", a2.cpu(), sample_rate)
