import os
import re
from time import perf_counter

import gradio as gr
import torch

from recipes.datasets.libritts import LibriTTSWebDataModule
from recipes.soundstorm2.lightning.soundstorm import SoundStorm
from recipes.soundstorm2.lightning.soundstream import SoundStreamSpeech24k
from samantha.utils.hdfs_helper import get, hdfs_ls

soundstorm = None
batch_size = 1
sample_rate = 24000
device = "cuda"


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


def generate_audio(
    n_seconds: int, iterations: str, score_strategies: str, temperatures: str
):
    global soundstorm

    n_seconds = int(n_seconds)
    max_seq_len = n_seconds * soundstorm.soundstorm.audio_model.frame_rate

    iterations = list(map(int, iterations.split(",")))
    score_strategies = score_strategies.split(",")
    temperatures = list(map(float, temperatures.split(",")))
    # tik = perf_counter()
    sampled_t = None

    with torch.no_grad():
        audio_tokens = soundstorm.audio_model(batch["audio"])

    tik = perf_counter()
    pred_audio_tokens, _ = soundstorm.iterative_decoding(
        max_seq_len=max_seq_len,
        iterations=iterations,
        score_strategies=score_strategies,
        semantic_tokens=None,
        guidance_scale=None,
        temperatures=temperatures,
        sampled_t=50,
        seed_tokens=audio_tokens,
        prefix_tokens=None,
    )

    with torch.no_grad():
        sampled_audio = soundstorm.audio_model.decode(pred_audio_tokens)

    tok = perf_counter()

    elapsed = tok - tik
    rtf = elapsed / n_seconds

    return_audios = []
    for a in sampled_audio:
        return_audios.append((sample_rate, a.T.cpu().numpy()))

    return *return_audios, rtf


if __name__ == "__main__":

    pl_datamodule = LibriTTSWebDataModule(
        sample_rate=24000, batch_size=1, shuffle_buffer_size=100
    )
    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    audio_model = SoundStreamSpeech24k()
    soundstorm = SoundStorm.load_from_checkpoint(
        "epoch=0-step=154000.ckpt", audio_model=audio_model
    )

    score_strategies = "random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit"
    n_seconds = gr.Textbox(value=10, label="Number of seconds", visible=True)
    iterations = gr.Textbox(
        value="48,24,12,8,8,8,4,4,2,2,2,2",  # 0.57 RTF
        # value="64,32,16,8,8,8,4,4,2,2,2,2", # 0.69 RTF
        # value="96,48,24,12,8,8,4,4,2,2,2,2",    # 0.96 RTF
        # value="128,64,32,16,8,8,4,4,2,2,2,2",   # 1.23 RTF
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
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[n_seconds, iterations, score_strategies, temperatures],
        outputs=[*audio_outputs, gr.Textbox(label="RTF (SoundStorm+SoundStream")],
        title="SoundStorm",
        description="Demo for efficient, non-autoregressive audio generation.",
    )
    demo.queue(concurrency_count=4)
    demo.launch(server_name="0.0.0.0")
