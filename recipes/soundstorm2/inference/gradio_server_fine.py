import os
import re
from time import perf_counter

import gradio as gr
import numpy as np
import torch
from julius.resample import resample_frac

from recipes.soundstorm.inference.semantic2audio import load_model2
from samantha.utils.hdfs_helper import get, hdfs_ls

soundstorm = None
batch_size = 1
sample_rate = 24000
n_audio_samples = 240000
device = "cuda"


def numpy_int16_to_torch_fp32(audio) -> torch.Tensor:
    return torch.from_numpy(audio.astype(np.float32, order="C") / 32767.0).T


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


def generate_audio(
    audio,
    ckpt_path: str,
    n_seconds: int,
    iterations: str,
    score_strategies: str,
    temperatures: str,
):
    global soundstorm
    sr, audio = audio
    audio = numpy_int16_to_torch_fp32(audio)
    audio = audio.to(soundstorm.device)

    audio = audio[:1, :n_audio_samples]

    audio = audio.unsqueeze(dim=0)

    audio = resample_frac(audio, sr, sample_rate)

    with torch.no_grad():
        audio_tokens = soundstorm.soundstorm.audio_model(audio)

    coarse_audio_tokens = audio_tokens[:, :4]

    with torch.no_grad():
        tmp_tokens = torch.cat(
            (
                coarse_audio_tokens,
                torch.zeros(
                    coarse_audio_tokens.shape[0],
                    8,
                    coarse_audio_tokens.shape[2],
                    dtype=coarse_audio_tokens.dtype,
                    device=coarse_audio_tokens.device,
                ),
            ),
            dim=1,
        )
        coarse_audio = soundstorm.soundstorm.audio_model.decode(tmp_tokens)

    n_seconds = int(n_seconds)

    if getattr(soundstorm, "ckpt_path") != ckpt_path:
        soundstorm._init_soundstorm(ckpt_path)
        soundstorm = soundstorm.to(device)
        soundstorm.ckpt_path = ckpt_path

    iterations = list(map(int, iterations.split(",")))
    score_strategies = score_strategies.split(",")
    temperatures = list(map(float, temperatures.split(",")))

    sampled_t = None

    sampled_audio, elapsed = soundstorm.generate_fine(
        coarse_audio_tokens,
        iterations=iterations,
        score_strategies=score_strategies,
        sampled_t=sampled_t,
        temperatures=temperatures,
    )
    # tok = perf_counter()
    rtf = elapsed / duration_sec

    return_audios = []
    return_coarse_audios = []
    for a, coarse_a in zip(sampled_audio, coarse_audio):
        return_audios.append((sample_rate, a.T.cpu().numpy()))
        return_coarse_audios.append((sample_rate, coarse_a.T.cpu().numpy()))

    return *return_audios, *return_coarse_audios, rtf


if __name__ == "__main__":
    duration_sec = n_audio_samples / sample_rate

    audio = gr.Audio(label="Audio to reconstruct")

    checkpoints = [
        "/mnt/bn/audio-diffusion/logs/soundstorm/3042805/soundstorm_fine/playlist/checkpoints/step=058000-loss=0.0000.ckpt"
    ]

    ckpt_dropdown = gr.Dropdown(
        choices=checkpoints, value=checkpoints[0], label="SoundStorm Checkpoint"
    )

    soundstorm = load_model2(checkpoints[0], None, device)
    soundstorm.ckpt_path = checkpoints[0]

    score_strategies = "random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit"
    n_seconds = gr.Textbox(value=10, label="Number of seconds", visible=True)
    iterations = gr.Textbox(
        # value="48,24,12,8,8,8,4,4,2,2,2,2", # 0.57 RTF
        # value="64,32,16,8,8,8,4,4,2,2,2,2", # 0.69 RTF
        # value="96,48,24,12,8,8,4,4,2,2,2,2",    # 0.96 RTF
        value="128,64,32,16,8,8,4,4,2,2,2,2",  # 1.23 RTF
        label="Iterations",
    )
    score_strategies = gr.Textbox(value=score_strategies, label="Score strategies")
    temperatures = gr.Textbox(
        # value="1.0,1.0,1.0,1.0,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95",
        # value="1.0,1.0,0.95,0.95,0.9,0.9,0.8,0.8,0.4,0.4,0.4,0.4",
        # value="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        value="0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95",
        label="Temperatures",
    )

    audio_outputs = [gr.Audio(label="Generated") for _ in range(batch_size)]
    coarse_outputs = [gr.Audio(label="Coarse audio") for _ in range(batch_size)]
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            audio,
            ckpt_dropdown,
            n_seconds,
            iterations,
            score_strategies,
            temperatures,
        ],
        outputs=[
            *audio_outputs,
            *coarse_outputs,
            gr.Textbox(label="RTF (SoundStorm+SoundStream"),
        ],
        title="SoundStorm",
        description="Demo for efficient, non-autoregressive audio generation.",
    )
    demo.queue(concurrency_count=4)
    demo.launch(share=True)
