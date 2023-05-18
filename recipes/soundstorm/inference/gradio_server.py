import os
import re
from time import perf_counter

import gradio as gr
import torch

from recipes.soundstorm.inference.semantic2audio import (
    generate,
    karaoke_validation_dataset,
    load_model,
)
from samantha.utils.hdfs_helper import get, hdfs_ls

arnold_task_id = 558104
soundstorm = None
dataset = None
batch_size = 1
sample_rate = 24000
n_audio_samples = 240000
device = "cuda"


def download_dataset(sample_rate: int, n_audio_samples: int, batch_size: int):
    return iter(karaoke_validation_dataset(sample_rate, n_audio_samples, batch_size))


def download_model(ckpt_idx: int):
    ckpt_path = checkpoints[ckpt_idx]
    local_fp = os.path.basename(ckpt_path)
    if not os.path.exists(local_fp):
        get(ckpt_path, local_fp)
    return local_fp


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


def generate_audio(
    semantic_token_sequence: str,
    ckpt_idx: str,
    n_seconds: int,
    iterations: str,
    score_strategies: str,
):
    global dataset
    global soundstorm

    n_seconds = int(n_seconds)
    n_audio_samples = n_seconds * sample_rate
    max_seq_len = n_seconds * soundstorm.audio_model.frame_rate
    if n_audio_samples != soundstorm.hparams.n_audio_samples:
        dataset = download_dataset(sample_rate, n_audio_samples, batch_size)
        soundstorm.hparams.n_audio_samples = n_audio_samples

    local_fp = download_model(ckpt_idx)
    if not hasattr(soundstorm, "local_fp") or soundstorm.local_fp != local_fp:
        soundstorm = load_model(local_fp, device, n_audio_samples=n_audio_samples)

    batch = next(dataset)
    batch = tuple(map(lambda a: a.to(device), batch))

    semantic_tokens, audio_tokens, ground_truth_audio = soundstorm.prepare_inputs(batch)
    if semantic_token_sequence != "":
        semantic_tokens = torch.tensor(
            eval(semantic_token_sequence), device=soundstorm.device
        )[None, :].repeat(batch_size, 1)

    iterations = list(map(int, iterations.split(",")))
    score_strategies = score_strategies.split(",")
    tik = perf_counter()
    sampled_t = None
    temperature = 1.0

    sampled_audio = generate(
        soundstorm,
        semantic_tokens,
        max_seq_len=max_seq_len,
        iterations=iterations,
        score_strategies=score_strategies,
        sampled_t=sampled_t,
        temperature=temperature,
    )

    tok = perf_counter()
    rtf = (tok - tik) / duration_sec

    sampled_audio = torch_fp32_to_numpy_int16(sampled_audio)

    if semantic_token_sequence == "":
        gt_audio_return = (sample_rate, torch_fp32_to_numpy_int16(ground_truth_audio))
    else:
        gt_audio_return = None
    return (sample_rate, sampled_audio), gt_audio_return, rtf


def ckpt_label_formatter(ckpt):
    trial_id = re.search(r"trials/(\d+?)/", ckpt).group(1)
    ckpt_fp = os.path.basename(ckpt)
    return f"{ckpt_fp} (Arnold trial {trial_id})"


if __name__ == "__main__":
    duration_sec = n_audio_samples / sample_rate

    hdfs_ckpt_dir = f"hdfs://harunava/home/byte_arnold_va/data/lab/audio/soundstorm/tasks/{arnold_task_id}/trials"
    checkpoints = list(
        filter(lambda f: ".ckpt" in f and "step=" in f, hdfs_ls(f"-R {hdfs_ckpt_dir}"))
    )
    checkpoints.reverse()

    semantic_token_sequence = gr.Textbox(
        label="Semantic token sequence (Optional)",
        placeholder="Enter a sequence of semantic tokens as a list, so [712, 333, 236, ...]",
    )

    choices = list(map(ckpt_label_formatter, checkpoints))
    ckpt_dropdown = gr.Dropdown(
        choices=choices, value=choices[0], type="index", label="Checkpoint"
    )

    dataset = download_dataset(sample_rate, n_audio_samples, batch_size)
    print(f"Downloading default model: {choices[0]}...")
    local_fp = download_model(ckpt_idx=0)

    print("Loading SoundStorm...")
    soundstorm = load_model(
        local_fp,
        device,
        n_audio_samples=n_audio_samples,
        masking_scheme=None,
        fine_quantizer_embedding_dropout=False,
        strict=False,
    )  # TODO
    soundstorm.local_fp = local_fp

    iterations = "32,32,32,32,8,8,8,8,8,8,8,8"
    score_strategies = "random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit"
    n_seconds = gr.Textbox(value=10, label="Number of seconds", visible=False)
    iterations = gr.Textbox(value=iterations, label="Iterations")
    score_strategies = gr.Textbox(value=score_strategies, label="Score strategies")
    # temperature = gr.Slider(
    #     minimum=0,
    #     maximum=1.0,
    #     value=1.0,
    #     step=0.1,
    #     interactive=True,
    #     label="Temperature",
    #     visible=True,
    # )
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            semantic_token_sequence,
            ckpt_dropdown,
            n_seconds,
            iterations,
            score_strategies,
        ],
        outputs=[
            gr.Audio(label="Generated"),
            gr.Audio(label="Ground truth"),
            gr.Textbox(label="RTF (SoundStorm+SoundStream"),
        ],
        title="SoundStorm",
        description="Demo for efficient, non-autoregressive audio generation.",
    )
    demo.queue(concurrency_count=4)
    demo.launch(share=True)
