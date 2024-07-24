import os
from time import perf_counter
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from recipes.research.audio_codec.scripts.compile import get_latest_model_from_commit
from recipes.research.diff import DiffInstrumental
from samantha.data.base import BaseAudioTransform
from samantha.utils.logger import RankedLogger

logger = RankedLogger()


def get_instrumental_diff(commit_hash: str, ckpt_path: Optional[str] = None):
    if ckpt_path is None:
        ckpt_path = get_latest_model_from_commit("diff/default", commit_hash)

    diff = DiffInstrumental.load_from_checkpoint(ckpt_path)
    diff = diff.eval().to("cuda")
    diff.setup()
    logger.info(diff.summarize())
    logger.info(diff.config)
    diff.commit_hash = commit_hash
    diff.commit_step = os.path.basename(ckpt_path)
    return diff


def generate_audio(
    audio_prompt: Optional[Tuple[int, np.ndarray]],
    text_prompt: str,
    negative_text_prompt: str,
    timesteps: int,
    cfg_weight: float,
    schedule_tau: float,
):
    if negative_text_prompt == "":
        batch_negative_text_prompt = None
    else:
        batch_negative_text_prompt = [negative_text_prompt]

    if audio_prompt is not None:
        sr, audio_prompt = audio_prompt
        audio_prompt = base_audio_transform(audio_prompt)

    logger.info(
        f"Inference parameters:\nPrompt: {text_prompt}\nNegative prompt: {negative_text_prompt}\nTimesteps: {timesteps}\nCFG weight: {cfg_weight}\nTau: {schedule_tau}"
    )
    with torch.no_grad():
        tik = perf_counter()

        if audio_prompt is None:
            batch_text = [text_prompt]
            pred_noise = model.sample_from_text(
                batch_text,
                t=timesteps,
                cfg_weight=cfg_weight,
                schedule_tau=schedule_tau,
                negative_text=batch_negative_text_prompt,
            )
        else:
            batch_audio = audio_prompt[None, :, :].to(model.device)
            pred_noise = model.sample_from_audio(
                batch_audio,
                sr,
                t=timesteps,
                cfg_weight=cfg_weight,
                schedule_tau=schedule_tau,
            )
        pred_audio = model.decode_audio(pred_noise.permute(0, 2, 1))
        tok = perf_counter()

    return_audios = []
    for a in pred_audio:
        return_audios.append((model.config.sample_rate, a.T.cpu().numpy()))

    rtf = (tok - tik) / model.config.max_duration
    return *return_audios, rtf


if __name__ == "__main__":
    # pip3 install httpx==0.23.3 pydantic==2.6
    batch_size = 1

    base_audio_transform = BaseAudioTransform()

    # model = get_instrumental_diff("7a291ad")
    model = get_instrumental_diff("611c8b2")

    text_prompt = gr.Textbox(label="Text prompt", placeholder="Enter a text prompt")
    negative_text_prompt = gr.Textbox(
        label="Negative text prompt", placeholder="Enter a negative text prompt"
    )
    audio_prompt = gr.Audio(label="Audio prompt")

    timesteps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Timesteps")

    cfg_weight = gr.Slider(
        minimum=0.0, maximum=12.0, value=2.5, step=0.1, label="CFG weight"
    )

    schedule_tau = gr.Slider(
        minimum=0.1, maximum=2.0, value=0.3, step=0.05, label="Tau (cosine schedule)"
    )

    audio_outputs = [gr.Audio(label="Generated") for _ in range(batch_size)]
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            audio_prompt,
            text_prompt,
            negative_text_prompt,
            timesteps,
            cfg_weight,
            schedule_tau,
        ],
        outputs=[*audio_outputs, gr.Textbox(label="RTF")],
        title="Direct Diffusion",
        description="Instrumental music generation using diffusion",
    )
    # demo.queue(concurrency_count=4)
    demo.launch(server_name="0.0.0.0", share=True)
