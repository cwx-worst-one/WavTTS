import os

import gradio as gr
import torch
from engine_pl import Engine
from hyperpyyaml import load_hyperpyyaml
from transformers import BertTokenizer

from samantha.utils.hparams import DotDict

root_dir = "/mlx/users/zongyu.yin/playground/samantha"
configs = {
    "mulan_ckpt": "/mnt/bn/audio-diffusion/pretrained_models/"
    "mulan/young-mulan-shortform-step=028000-median_rank_1=149-kaggle_merged.pt",
    "semantic_ckpt": "/mnt/bn/audio-diffusion/logs/377594/trials/2326226/"
    "musiclm_text2semantic/karaoke_acc/checkpoints/epoch=0-step=12500.ckpt",
    "semantic_yaml": os.path.join(
        root_dir, "recipes/audio_diffusion/conf/250223-musiclm/text2semantic_mix.yaml"
    ),
    "coarse_ckpt": "/mnt/bn/audio-diffusion/logs/377594/trials/2326275/"
    "musiclm_coarse/karaoke_acc/checkpoints/epoch=0-step=44000.ckpt",
    "coarse_yaml": os.path.join(
        root_dir, "recipes/audio_diffusion/conf/250223-musiclm/coarse_mix.yaml"
    ),
    "fine_ckpt": "/mnt/bn/audio-diffusion/logs/373893/trials/2310762/"
    "musiclm_fine/resso_mss_uncond/checkpoints/epoch=0-step=154000.ckpt",
    "fine_yaml": os.path.join(
        root_dir,
        "recipes/audio_diffusion/conf/"
        + "080223-coarse-fine/musiclm_fine_v2_karaoke_acc.yaml",
    ),
    "diffusion_ckpt": "/mnt/bn/audio-diffusion/logs/281298/trials/2303785/"
    "diffusion/DAG_700M_seq_mulan/checkpoints/last.ckpt",
    "diffusion_yaml": os.path.join(
        root_dir,
        "recipes/audio_diffusion/conf/"
        + "latent_diffusion_reboot/DAG_700M_sequence_mulan.yaml",
    ),
}


def tokenize(text, tokenizer):
    token_outputs = tokenizer(text, return_tensors="pt")
    token_outputs["input_ids"] = token_outputs["input_ids"].to(device)
    token_outputs["token_type_ids"] = token_outputs["token_type_ids"].to(device)
    token_outputs["attention_mask"] = token_outputs["attention_mask"].to(device)
    return token_outputs


def load_config_from_file(config_fp: str):
    with open(config_fp, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides="")
    return DotDict(hparams)


def load_semantic_model(config_fp, ckpt_path):
    cfg = load_config_from_file(config_fp)
    pl_module = (
        cfg.pl_module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            semantic_token_model=cfg.semantic_token_model,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    return pl_module


def load_coarse_model(config_fp, ckpt_path):
    cfg = load_config_from_file(config_fp)
    encoder = cfg.encoder_transform.to(device).eval()
    pl_module = (
        cfg.pl_module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            acoustic_token_model=cfg.acoustic_token_model,
            cuda_transforms=encoder,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    return pl_module


def load_fine_model(config_fp, ckpt_path):
    cfg = load_config_from_file(config_fp)
    encoder = cfg.encoder_transform.to(device).eval()
    pl_module = (
        cfg.pl_module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            acoustic_token_model=cfg.acoustic_token_model,
            cuda_transforms=encoder,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    return pl_module, encoder


def load_diffusion_model(config_fp, ckpt_path):
    cfg = load_config_from_file(config_fp)
    encoder = cfg.encoder_transform.to(device).eval()
    pl_module = (
        cfg.pl_module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=cfg.model,
            cuda_transforms=encoder,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    return pl_module, encoder


def generate_semantic(text_emb, semantic_temp):
    sample_length = 250  # 10 seconds
    with torch.no_grad():
        print(f"Generating {sample_length} semantic tokens...")
        semantic_tokens = semantic_model.generate_samples(
            text_embeddings=text_emb,
            temperature=semantic_temp,
            num_outputs=1,
            sequence_length=sample_length,
        )
    return semantic_tokens.long()


def generate_coarse(text_emb, semantic_tokens, coarse_temp):
    sample_length = 800  # 10 seconds
    with torch.no_grad():
        print(f"Generating {sample_length} coarse tokens...")
        coarse_tokens = coarse_model.generate_samples(
            labels=None,
            semantic=semantic_tokens,
            cond_embeddings=text_emb,
            temperature=coarse_temp,
            num_outputs=1,
            sequence_length=sample_length,
        )
    return coarse_tokens.long()


def generate_fine(coarse_tokens, fine_temp, fine_step_size=80):
    window_size = 240
    assert fine_step_size <= window_size, "Step size can't be larger than window size!"
    assert (
        coarse_tokens.shape[2] - window_size
    ) % fine_step_size == 0, "For now, remaining length must be multiples of step size"
    prefix_size = window_size - fine_step_size

    with torch.no_grad():
        fine_tokens = None
        start_idx = 0
        samples_to_generate = window_size
        seed_sequence = None
        while fine_tokens is None or fine_tokens.shape[2] < coarse_tokens.shape[2]:
            print(f"Generating {samples_to_generate} fine tokens...")
            sub_labels = coarse_tokens[:, :, start_idx : start_idx + window_size]
            sub_fine_tokens = fine_model.generate_samples(
                cond_embeddings=None,
                labels=sub_labels,
                temperature=fine_temp,
                num_outputs=1,
                sequence_length=window_size,
                seed_sequence=seed_sequence,
            )
            selected_fine_tokens = sub_fine_tokens[:, :, -samples_to_generate:]
            if fine_tokens is None:
                fine_tokens = selected_fine_tokens
            else:
                fine_tokens = torch.cat([fine_tokens, selected_fine_tokens], dim=2)
            # Prepare for next round
            start_idx += fine_step_size
            samples_to_generate = fine_step_size
            if prefix_size > 0:
                dummy_coarse_tokens = coarse_tokens[:, :, -prefix_size:]
                prefix_fine_tokens = sub_fine_tokens[:, :, -prefix_size:]
                tmp_model = fine_model.acoustic_token_model
                seed_sequence = tmp_model.extract_channel_group_and_interleave(
                    labels=torch.cat(
                        [dummy_coarse_tokens, prefix_fine_tokens], dim=1
                    ).permute(0, 2, 1),
                    start_channel=2,
                    end_channel=6,
                    apply_offset=True,
                )
            else:
                seed_sequence = None
    return fine_tokens.long()


def generate_by_musiclm(raw_text_prompt, semantic_temp, coarse_temp, fine_temp):
    with torch.no_grad():
        token_outputs = tokenize(raw_text_prompt, bert_tokenizer)
        text_emb = mulan_model.model.text_encoder(**token_outputs)[0]

    generated_semantic_tokens = generate_semantic(text_emb[None, :], semantic_temp)
    generated_coarse_tokens = generate_coarse(
        text_emb[None, :], generated_semantic_tokens, coarse_temp
    )
    generated_fine_tokens = generate_fine(generated_coarse_tokens, fine_temp)
    generated_acoustic_tokens = torch.cat(
        [generated_coarse_tokens, generated_fine_tokens], dim=1
    )
    print("MusicLM is decoding acoustic tokens...")
    with torch.no_grad():
        demo = (
            musiclm_acoustic_encoder.decode(generated_acoustic_tokens).cpu().squeeze(0)
        )
    demo = (demo / demo.abs().max()).permute(1, 0).numpy()
    print("Outputting the generated audio...")
    return 24000, demo


def generate_by_diffusion(raw_text_prompt, guidance_scale, num_steps):
    with torch.no_grad():
        token_outputs = tokenize(raw_text_prompt, bert_tokenizer)
        text_emb = mulan_model.model.text_encoder(**token_outputs)[0]

        initial_latents = torch.randn(1, 256, 800, device=device)
        generated_acoustic_tokens = diffusion_model.generate_samples(
            initial_latents,
            text_emb[None, None, :],
            guidance_scale=guidance_scale,
            num_steps=int(num_steps),
        )
        demo = (
            diffusion_acoustic_encoder.decode(generated_acoustic_tokens)
            .cpu()
            .squeeze(0)
        )
    demo = (demo / demo.abs().max()).permute(1, 0).numpy()
    print("Outputting the generated audio...")
    return 24000, demo


######################
#    model init   #
######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
#    text model   #
######################
print("Loading BERT tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Loading MuLan model...")
mulan_model = Engine.load_from_checkpoint(configs["mulan_ckpt"]).eval().to(device)

######################
#     MusicLM     #
######################
print("Loading semantic model...")
semantic_model = load_semantic_model(configs["semantic_yaml"], configs["semantic_ckpt"])
print("Loading coarse model...")
coarse_model = load_coarse_model(configs["coarse_yaml"], configs["coarse_ckpt"])
print("Loading fine model...")
fine_model, musiclm_acoustic_encoder = load_fine_model(
    configs["fine_yaml"], configs["fine_ckpt"]
)

######################
#     Diffusion   #
######################
print("Loading diffusion model...")
diffusion_model, diffusion_acoustic_encoder = load_diffusion_model(
    configs["diffusion_yaml"], configs["diffusion_ckpt"]
)

# sr, gen_audio = generate_by_musiclm("jazz", 1.0, 1.0, 1.0)
# torchaudio.save("generate_by_musiclm.wav", torch.from_numpy(gen_audio), sr)
# sr, gen_audio = generate_by_diffusion("jazz", 7, 100)
# torchaudio.save("generate_by_diffusion.wav", torch.from_numpy(gen_audio), sr)

with gr.Blocks() as demo:
    gr.Markdown("Text to Music demo app")

    with gr.Tab("Generate samples"):
        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                text_prompt = gr.Textbox(
                    placeholder="Enter the text prompt here", label="Text prompt"
                )
                semantic_temp = gr.Slider(
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    label="Temperature of semantic tokens sampling",
                )
                coarse_temp = gr.Slider(
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    label="Temperature of coarse tokens sampling",
                )
                fine_temp = gr.Slider(
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    label="Temperature of fine tokens sampling",
                )
                guidance_scale = gr.Slider(
                    value=7.0,
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                    label="Guidance scale of diffusion sampling",
                )
                diffusion_steps = gr.Slider(
                    value=100.0,
                    minimum=1.0,
                    maximum=200.0,
                    step=1.0,
                    label="Number of steps of diffusion sampling",
                )
                with gr.Row(variant="panel"):
                    generate_by_diffusion_button = gr.Button("Generate by Diffusion")
                    generate_by_musiclm_button = gr.Button("Generate by MusicLM")

            with gr.Column(variant="panel"):
                generated_sample = gr.Audio(label="Generated audio")

    with gr.Tab("SingSong (WIP)"):
        with gr.Row(variant="panel"):
            with gr.Column(variant="panel"):
                mic_input_audio = gr.Audio(
                    label="Input audio from mic", source="microphone"
                )
            with gr.Column(variant="panel"):
                generated_singsong_sample_1 = gr.Audio(label="Generated audio 1")
                generated_singsong_sample_2 = gr.Audio(label="Generated audio 2")
                generated_singsong_sample_3 = gr.Audio(label="Generated audio 3")

    with gr.Tab("Reload models"):
        with gr.Column(variant="panel"):
            with gr.Accordion(label="MuLan", open=False):
                mulan_model_ckpt = gr.Textbox(
                    lines=1,
                    placeholder="The file path to the mulan model checkpoint.",
                    label="ckpt",
                )
                mulan_model_yaml = gr.Textbox(
                    lines=1,
                    placeholder="The file path to the mulan model yaml.",
                    label="yaml",
                )
                mulan_model_load_button = gr.Button("Load")
        with gr.Column(variant="panel"):
            with gr.Accordion(label="Diffusion", open=False):
                diffusion_model_ckpt = gr.Textbox(
                    lines=1,
                    placeholder="The file path to the diffusion model checkpoint.",
                    label="ckpt",
                )
                diffusion_model_yaml = gr.Textbox(
                    lines=1,
                    placeholder="The file path to the diffusion model yaml.",
                    label="yaml",
                )
                diffusion_model_load_button = gr.Button("Load")
        with gr.Column(variant="panel"):
            with gr.Accordion(label="MusicLM", open=False):
                with gr.Column(variant="panel"):
                    with gr.Accordion(label="Semantic", open=False):
                        semantic_model_ckpt = gr.Textbox(
                            lines=1,
                            placeholder=(
                                "The file path to the semantic model checkpoint."
                            ),
                            label="ckpt",
                        )
                        semantic_model_yaml = gr.Textbox(
                            lines=1,
                            placeholder="The file path to the semantic model yaml.",
                            label="yaml",
                        )
                        semantic_model_load_button = gr.Button("Load")
                with gr.Column(variant="panel"):
                    with gr.Accordion(label="Coarse", open=False):
                        coarse_model_ckpt = gr.Textbox(
                            lines=1,
                            placeholder="The file path to the coarse model checkpoint.",
                            label="ckpt",
                        )
                        coarse_model_yaml = gr.Textbox(
                            lines=1,
                            placeholder="The file path to the coarse model yaml.",
                            label="yaml",
                        )
                        coarse_model_load_button = gr.Button("Load")
                with gr.Column(variant="panel"):
                    with gr.Accordion(label="Fine", open=False):
                        fine_model_ckpt = gr.Textbox(
                            lines=1,
                            placeholder="The file path to the fine model checkpoint.",
                            label="ckpt",
                        )
                        fine_model_yaml = gr.Textbox(
                            lines=1,
                            placeholder="The file path to the fine model yaml.",
                            label="yaml",
                        )
                        fine_model_load_button = gr.Button("Load")

    generate_by_musiclm_button.click(
        fn=generate_by_musiclm,
        inputs=[text_prompt, semantic_temp, coarse_temp, fine_temp],
        outputs=generated_sample,
        show_progress=True,
    )
    generate_by_diffusion_button.click(
        fn=generate_by_diffusion,
        inputs=[text_prompt, guidance_scale, diffusion_steps],
        outputs=generated_sample,
        show_progress=True,
    )

demo.queue()
demo.launch(
    max_threads=10,
    show_error=True,
    server_name=os.getenv("ARNOLD_TENSORBOARD_CURRENT_HOST", "0.0.0.0"),
    server_port=os.getenv("ARNOLD_TENSORBOARD_CURRENT_PORT", 6006),
)
