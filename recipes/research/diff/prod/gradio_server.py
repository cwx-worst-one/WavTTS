import torch
import gradio as gr

from recipes.research.diff.prod import load_diffusion_model, sample
from recipes.research.prompt_gpt.prod import load_promptgpt_model, sample_prompt_gpt
from samantha.data.av_audio import audio_write

batch_size = 1


def generate_audio(
    prompt: str,
    negative_prompt: str,
    seconds_total: int,
    t: int,
    cfg_weight: float,
    schedule_tau: float,
    solver: str,
    batch_size: int,
):
    prompt = f"high quality {prompt}"

    if negative_prompt == "":
        negative_prompt = None

    pred_audio, sample_rate = sample(
        diffusion_model,
        prompt,
        seconds_start=0,
        seconds_total=seconds_total,
        t=t,
        cfg_weight=cfg_weight,
        schedule_tau=schedule_tau,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        solver=solver,
        device=device,
    )

    pred_audio = pred_audio.cpu()
    audio_bytes = [
        audio_write(a, sample_rate, format="mp3", mp3_rate=320) for a in pred_audio
    ]
    return [prompt] + audio_bytes


def extend_prompt(text_prompt: str, temperature: float):
    pred_text = sample_prompt_gpt(
        prompt_gpt_model, text_prompt, temperature=temperature
    )
    return pred_text


if __name__ == "__main__":
    # pip3 install httpx==0.23.0
    device = "cuda:0"

    # commit_hash = "55ff351"  # v1
    commit_hash = "5c5f7aa"  # 1B
    # commit_hash = "7fa92d7"  # 1B, finetune EveryNoise
    # commit_hash = "ac11aa0"  # 2B
    # commit_hash = "a827e62"  # 1B, logit_normal
    diffusion_model = load_diffusion_model(commit_hash, device=device, cache=True, ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/diff/default/5c5f7aa/2024-07-09/05-08-40/checkpoints/step=500000.ckpt")
    print(diffusion_model.summarize())

    # PromptGPT
    prompt_gpt_model = load_promptgpt_model("55f2f18", device=device, cache=True)

    # Create the Gradio app
    with gr.Blocks(
        title="Research Model",
        theme="soft",
        analytics_enabled=False,
        css="footer {visibility: hidden}",
    ) as app:
        with gr.Row() as row:
            gr.Markdown("# Research Model")

        # PromptGPT
        with gr.Row() as row:
            with gr.Column() as col:
                text_prompt = gr.Textbox(label="Enter your prompt")
                prompt_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8)
                extend_button = gr.Button("Extend Prompt")

            with gr.Column() as col:
                extended_prompt = gr.Textbox(label="Extended prompt")

            extend_button.click(
                fn=extend_prompt,
                inputs=[text_prompt, prompt_temperature],
                outputs=extended_prompt,
            )

            if extend_button.value != "":            
                text_prompt.value = extended_prompt.value


        # Diffusion:
        with gr.Row() as row:
            with gr.Column() as col:

                negative_prompt = gr.Textbox(label="Negative prompt", value="bad quality")

                t = gr.Slider(
                    minimum=1, maximum=500, value=50, step=1, label="Noise steps"
                )

                cfg_weight = gr.Slider(
                    minimum=1.0, maximum=12.0, value=2.5, step=0.1, label="CFG"
                )

                schedule_tau = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Schedule slope (tau)",
                )

                solver = gr.Dropdown(
                    diffusion_model.solvers, value="dpmpp-3m-sde", label="Solver"
                )

                batch_size = gr.Slider(
                    minimum=1, maximum=8, value=2, step=1, label="Batch size"
                )

                duration = gr.Slider(
                    minimum=5, maximum=60, value=60, step=1, label="Duration"
                )
                generate_audio_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column() as col:
                result = [gr.Textbox(label="Prompt")] + [gr.Audio() for _ in range(2)]

            generate_audio_button.click(
                fn=generate_audio,
                inputs=[
                    text_prompt,
                    negative_prompt,
                    duration,
                    t,
                    cfg_weight,
                    schedule_tau,
                    solver,
                    batch_size,
                ],
                outputs=result,
            )

    app.launch(server_name="0.0.0.0", server_port=6006, share=True, debug=True)
