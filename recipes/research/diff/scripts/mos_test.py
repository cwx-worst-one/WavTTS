import json
import os

import pandas as pd
from tqdm import tqdm

from recipes.research.audio_codec.zoo import AudioCodec_7c355ea_64l
from recipes.research.diff.prod import load_diffusion_model, sample
from samantha.data.av_audio import audio_write


def get_text_prompts(prompts_fp: str):
    prompts = pd.read_csv(prompts_fp)
    return prompts["text_prompt"].tolist()

if __name__ == "__main__":

    commit_hash = "5c5f7aa"  # 1B
    prompts_fp = "/mnt/bn/audio-diffusion/peng/sstk_random_30_prompts.csv"
    OUT_DIR = f"/mnt/bn/janne-research-xl/mos_tests/v2_diffusion_stereo_{commit_hash}"

    os.makedirs(OUT_DIR)

    t = 200
    cfg_weight = 3.0
    schedule_tau = 1.0
    solver = "dpmpp-3m-sde"
    
    model = load_diffusion_model(commit_hash, device="cuda")
    model = model.load_audio_codec(AudioCodec_7c355ea_64l(), device="cuda")

    text_prompts = get_text_prompts(prompts_fp)

    for idx, text in tqdm(enumerate(text_prompts)):

        details = {
            "idx": idx,
            "text": text,
            "commit_hash": model.commit_hash,
            "commit_step": model.commit_step,
            "cfg_weight": cfg_weight,
            "schedule_tau": schedule_tau,
            "solver": solver,
        }

        fn =  f"{idx}_{text[:150]}_t={t}_cfg={cfg_weight}_tau={schedule_tau}_solver={solver}"
        fn = fn.replace("/", "-")

        fp = os.path.join(OUT_DIR, fn)

        pred_audio, sample_rate = sample(
            model,
            text,
            seconds_start=0,
            seconds_total=60,
            t=t,
            cfg_weight=cfg_weight,
            schedule_tau=schedule_tau,
            solver=solver
        )

        audio_write(
            pred_audio[0].cpu(),
            sample_rate,
            stem_name=fp + ".mp3",
            format="mp3",
            mp3_rate=320,
            add_suffix=False
        )

        with open(fp + ".json", "w") as f:
            json.dump(details, f)
