import os
import re
from time import perf_counter

import gradio as gr
import torch

from recipes.soundstorm.inference.semantic2audio import load_model
from samantha.utils.hdfs_helper import get, hdfs_ls

arnold_task_id = 558104
soundstorm = None
batch_size = 5
sample_rate = 24000
n_audio_samples = 240000
device = "cuda"


def download_model(ckpt_idx: int):
    ckpt_path = checkpoints[ckpt_idx]
    local_fp = os.path.basename(ckpt_path)
    if not os.path.exists(local_fp):
        get(ckpt_path, local_fp)
    return local_fp


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


def generate_audio(
    text: str,
    semantic_tokens: str,
    ckpt_path: str,
    semantic_ckpt_path: str,
    n_seconds: int,
    iterations: str,
    score_strategies: str,
    guidance_scale: float,
    temperatures: str,
    semantic_temperature: float,
):
    global soundstorm

    n_seconds = int(n_seconds)
    max_seq_len = n_seconds * soundstorm.soundstorm.audio_model.frame_rate

    if (
        not hasattr(soundstorm, "ckpt_path")
        or soundstorm.ckpt_path != ckpt_path
        or soundstorm.semantic_ckpt_path != semantic_ckpt_path
    ):
        soundstorm = load_model(default_ckpt, default_semantic_ckpt, device)

    if semantic_tokens != "":
        semantic_tokens = torch.tensor(eval(semantic_tokens), device=soundstorm.device)[
            None, :
        ].repeat(batch_size, 1)

    iterations = list(map(int, iterations.split(",")))
    score_strategies = score_strategies.split(",")
    temperatures = list(map(float, temperatures.split(",")))
    tik = perf_counter()
    sampled_t = None

    if semantic_tokens != "":
        sampled_audio, _ = soundstorm.semantic2audio(
            semantic_tokens,
            max_seq_len=max_seq_len,
            iterations=iterations,
            score_strategies=score_strategies,
            sampled_t=sampled_t,
            temperatures=temperatures,
        )
    else:
        if guidance_scale is not None:
            print("[TODO] SETTING guidance_scale = None")
            guidance_scale = None

        sampled_audio, semantic_tokens = soundstorm.generate(
            text,
            batch_size,
            max_seq_len=max_seq_len,
            iterations=iterations,
            score_strategies=score_strategies,
            guidance_scale=guidance_scale,
            sampled_t=sampled_t,
            temperatures=temperatures,
            semantic_temperature=semantic_temperature,
        )

    tok = perf_counter()
    rtf = (tok - tik) / duration_sec

    return_audios = []
    for a in sampled_audio:
        return_audios.append((sample_rate, torch_fp32_to_numpy_int16(a)))
    return *return_audios, f"{list(semantic_tokens[0].cpu().numpy())}", rtf


def ckpt_label_formatter(ckpt):
    trial_id = re.search(r"trials/(\d+?)/", ckpt).group(1)
    ckpt_fp = os.path.basename(ckpt)
    return f"{ckpt_fp} (Arnold trial {trial_id})"


from glob import glob

if __name__ == "__main__":
    duration_sec = n_audio_samples / sample_rate

    # hdfs_ckpt_dir = f"hdfs://harunava/home/byte_arnold_va/data/lab/audio/soundstorm/tasks/{arnold_task_id}/trials"
    # checkpoints = list(
    #     filter(lambda f: ".ckpt" in f and "step=" in f, hdfs_ls(f"-R {hdfs_ckpt_dir}"))
    # )
    checkpoints = list(
        glob("/mnt/bn/audio-diffusion/logs/soundstorm/**/*.ckpt", recursive=True)
    )
    checkpoints = list(filter(lambda f: ".ckpt" in f and "step=" in f, checkpoints))
    checkpoints.reverse()

    semantic_checkpoints = list(
        glob(
            "/mnt/bn/audio-diffusion/ckpts/musiclm/semantic_flash_llama/**/*.ckpt",
            recursive=True,
        )
    )

    text = gr.Textbox(label="Text prompt", placeholder="Enter a text prompt")

    semantic_token_sequence = gr.Textbox(
        label="Semantic token sequence (Optional)",
        placeholder="Enter a sequence of semantic tokens as a list, so [712, 333, 236, ...]",
    )

    # choices = list(map(ckpt_label_formatter, checkpoints))
    # default_ckpt = "/mnt/bn/audio-diffusion/logs/soundstorm/2829482/output/soundstorm/baseline/checkpoints/epoch=0-step=105000.ckpt"
    default_ckpt = "/mnt/bn/audio-diffusion/logs/soundstorm/mcc40m-normalized-cfg/soundstorm/mcc40m/checkpoints/step=035000-loss=0.0000.ckpt"
    ckpt_dropdown = gr.Dropdown(
        choices=checkpoints, value=default_ckpt, label="SoundStorm Checkpoint"
    )

    default_semantic_ckpt = "/mnt/bn/audio-diffusion/ckpts/musiclm/semantic_flash_llama/mcc40m/step=040000-tr_loss=2.3858.ckpt"
    semantic_ckpt_dropdown = gr.Dropdown(
        choices=semantic_checkpoints,
        value=default_semantic_ckpt,
        label="Semantic Checkpoint",
    )

    soundstorm = load_model(default_ckpt, default_semantic_ckpt, device)
    soundstorm.ckpt_path = default_ckpt
    soundstorm.semantic_ckpt_path = default_semantic_ckpt

    iterations = "32,32,16,16,8,8,4,4,2,2,1,1"
    score_strategies = "random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit"
    n_seconds = gr.Textbox(value=10, label="Number of seconds", visible=True)
    iterations = gr.Textbox(value=iterations, label="Iterations")
    score_strategies = gr.Textbox(value=score_strategies, label="Score strategies")
    temperatures = gr.Textbox(
        value="1.0,1.0,0.95,0.95,0.9,0.9,0.8,0.8,0.4,0.4,0.4,0.4", label="Temperatures"
    )
    semantic_temperature = gr.Slider(
        minimum=0,
        maximum=2.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label="Semantic temperature",
        visible=True,
    )

    guidance_scale = gr.Slider(
        minimum=0,
        maximum=6.0,
        value=0,
        step=0.1,
        interactive=True,
        label="Guidance scale",
    )

    audio_outputs = [gr.Audio(label="Generated") for _ in range(batch_size)]
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            text,
            semantic_token_sequence,
            ckpt_dropdown,
            semantic_ckpt_dropdown,
            n_seconds,
            iterations,
            score_strategies,
            guidance_scale,
            temperatures,
            semantic_temperature,
        ],
        outputs=[
            *audio_outputs,
            gr.Textbox(label="Semantic tokens"),
            gr.Textbox(label="RTF (SoundStorm+SoundStream"),
        ],
        title="SoundStorm",
        description="Demo for efficient, non-autoregressive audio generation.",
    )
    demo.queue(concurrency_count=4)
    demo.launch(share=True)
