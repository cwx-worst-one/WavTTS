import os
import re
from time import perf_counter

import gradio as gr
import torch

from recipes.soundstorm.inference.semantic2audio import load_model2
from samantha.utils.hdfs_helper import get, hdfs_ls

arnold_task_id = 558104
soundstorm = None
batch_size = 1
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
    temperatures: str,
    semantic_temperature: float,
):
    global soundstorm

    n_seconds = int(n_seconds)
    max_seq_len = n_seconds * soundstorm.soundstorm.audio_model.frame_rate

    if getattr(soundstorm, "ckpt_path") != ckpt_path:
        soundstorm._init_soundstorm(ckpt_path)
        soundstorm = soundstorm.to(device)
        soundstorm.ckpt_path = ckpt_path

    if getattr(soundstorm, "semantic_ckpt_path") != semantic_ckpt_path:
        soundstorm._init_semantic(semantic_ckpt_path)
        soundstorm = soundstorm.to(device)
        soundstorm.semantic_ckpt_path = semantic_ckpt_path

    if semantic_tokens != "":
        semantic_tokens = torch.tensor(
            eval(semantic_tokens), device=soundstorm.device
        )[None, :].repeat(batch_size, 1)

    iterations = list(map(int, iterations.split(",")))
    score_strategies = score_strategies.split(",")
    temperatures = list(map(float, temperatures.split(",")))
    #tik = perf_counter()
    sampled_t = None

    if semantic_tokens != "":
        sampled_audio, _, elapsed = soundstorm.semantic2audio(
            semantic_tokens,
            max_seq_len=max_seq_len,
            iterations=iterations,
            score_strategies=score_strategies,
            sampled_t=sampled_t,
            temperatures=temperatures,
        )
    else:
        sampled_audio, semantic_tokens, elapsed = soundstorm.generate(
            text,
            batch_size,
            max_seq_len=max_seq_len,
            iterations=iterations,
            score_strategies=score_strategies,
            sampled_t=sampled_t,
            temperatures=temperatures,
            semantic_temperature=semantic_temperature,
        )

    #tok = perf_counter()
    rtf = elapsed / duration_sec

    return_audios = []
    for a in sampled_audio:
        return_audios.append((sample_rate, a.T.cpu().numpy()))

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
    #checkpoints = list(glob("/mnt/bn/audio-diffusion/logs/soundstorm/**/*.ckpt", recursive=True))
    #checkpoints = list(filter(lambda f: ".ckpt" in f and "step=" in f, checkpoints))
    #checkpoints.reverse()
    checkpoints = [
        "/mnt/bn/audio-diffusion/logs/soundstorm/3042805/soundstorm_fine/playlist/checkpoints/step=058000-loss=0.0000.ckpt",
        "/mnt/bn/audio-diffusion/ducle/logs/soundstorm_mha/playlist/checkpoints/step=080000-loss=0.0000.ckpt",
        "/mnt/bn/audio-diffusion/ducle/logs/soundstorm_mha/mcc40m_conservative_filtered/checkpoints/step=075000-loss=0.0000.ckpt",
        "/mnt/bn/audio-diffusion/ducle/logs/soundstorm_mha/mcc40m_conservative_filtered/checkpoints/step=100000-loss=0.0000.ckpt",
    ]

    #semantic_checkpoints = list(glob("/mnt/bn/audio-diffusion/ckpts/musiclm/semantic_flash_llama/**/*.ckpt", recursive=True))
    semantic_checkpoints = [
        "/mnt/bn/audio-diffusion/ducle/logs/semantic_flash_llama/mcc40m_filtered_705m/checkpoints/step=081000-tr_loss=2.4451-val_loss_0=2.8561.ckpt",
    ]

    text = gr.Textbox(
        label="Text prompt",
        placeholder="Enter a text prompt",
    )

    semantic_token_sequence = gr.Textbox(
        label="Semantic token sequence (Optional)",
        placeholder="Enter a sequence of semantic tokens as a list, so [712, 333, 236, ...]",
    )

    # choices = list(map(ckpt_label_formatter, checkpoints))
    # default_ckpt = "/mnt/bn/audio-diffusion/logs/soundstorm/2829482/output/soundstorm/baseline/checkpoints/epoch=0-step=105000.ckpt"
    ckpt_dropdown = gr.Dropdown(
        choices=checkpoints, value=checkpoints[0], label="SoundStorm Checkpoint"
    )

    #default_semantic_ckpt = "/mnt/bn/audio-diffusion/ckpts/musiclm/semantic_flash_llama/mcc40m/step=040000-tr_loss=2.3858.ckpt"
    semantic_ckpt_dropdown = gr.Dropdown(
        choices=semantic_checkpoints, value=semantic_checkpoints[0], label="Semantic Checkpoint"
    )

    soundstorm = load_model2(checkpoints[0], semantic_checkpoints[0], device)
    soundstorm.ckpt_path = checkpoints[0]
    soundstorm.semantic_ckpt_path = semantic_checkpoints[0]

    score_strategies = "random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit"
    n_seconds = gr.Textbox(value=10, label="Number of seconds", visible=True)
    iterations = gr.Textbox(
        #value="48,24,12,8,8,8,4,4,2,2,2,2", # 0.57 RTF
        #value="64,32,16,8,8,8,4,4,2,2,2,2", # 0.69 RTF
        #value="96,48,24,12,8,8,4,4,2,2,2,2",    # 0.96 RTF
        value="128,64,32,16,8,8,4,4,2,2,2,2",   # 1.23 RTF
        label="Iterations",
    )
    score_strategies = gr.Textbox(value=score_strategies, label="Score strategies")
    temperatures = gr.Textbox(
        #value="1.0,1.0,1.0,1.0,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95",
        #value="1.0,1.0,0.95,0.95,0.9,0.9,0.8,0.8,0.4,0.4,0.4,0.4",
        #value="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        value="0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95",
        label="Temperatures",
    )
    semantic_temperature = gr.Slider(
        minimum=0,
        maximum=2.0,
        value=1.0,
        step=0.1,
        interactive=True,
        label="Semantic temperature",
        visible=True,
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
