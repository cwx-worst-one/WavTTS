import pandas as pd
import torch
from tqdm import tqdm
from webdataset import WebDataset


def gather_text_prompts(
    prompts_group: str, direct_prompt: str, batch_size: int
):
    items = []
    if prompts_group in ["google", "all"]:
        df = pd.read_csv(
            "/mnt/bn/audio-diffusion/data/google_prompts/google_prompts.csv"
        )
        for _, row in df.iterrows():
            items.append([row["category"], row["text"]])

    if prompts_group in ["musiccaps-asp", "all"]:
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["aspect_list"]])

    if prompts_group in ["musiccaps-cap", "all"]:
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["text"]])

    if prompts_group in ["sami", "all"]:
        tar_file_paths = [
            "/mnt/bn/audio-diffusion/data/mulan_prompts/sami_prompts.tar.mulan149",
            "/mnt/bn/audio-diffusion/data/mulan_prompts/genre_stats.tar.mulan149",
            "/mnt/bn/audio-diffusion/data/mulan_prompts/mood_stats.tar.mulan149",
            "/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_long_prompts.tar.mulan149",  # noqa
            "/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_short_prompts.tar.mulan149",  # noqa
            "/mnt/bn/audio-diffusion/data/mulan_prompts/painting_desc.tar.mulan149",
            "/mnt/bn/audio-diffusion/data/mulan_prompts/theme_stats.tar.mulan149",
        ]
        for tar_file_path in tar_file_paths:
            prefix = tar_file_path.split("/")[-1].split(".")[0]
            tar_file = WebDataset(tar_file_path).decode()
            for item in tar_file:
                items.append([prefix, item["text.txt"]])

    if prompts_group in ["gpt", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/gpt_text_prompts.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["gpt", line.strip()])

    if prompts_group in ["image", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/image_captions.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["image captions", line.strip()])

    if prompts_group in ["painting", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/image2text2music.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["image2text2music", line.strip()])

    if prompts_group == "direct_prompt":
        for i in range(batch_size):
            items.append(["direct_prompt", direct_prompt])

    if prompts_group == "audio_prompt":
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["audio prompt (musiccaps)", row["text"]])

    prompts = []
    categories = []
    pbar = tqdm(items)
    for item in pbar:
        pbar.set_description("Extracting mulan text embeds...")
        prompts.append(item[1])
        categories.append(item[0])
    return prompts, categories


def gather_prompts(
    mulan_model, prompts_group: str, direct_prompt: str, batch_size: int, device: str
):
    items = gather_text_prompts(prompts_group, direct_prompt, batch_size)
    
    text_embs = []
    prompts = []
    categories = []
    pbar = tqdm(items)
    for item in pbar:
        pbar.set_description("Extracting mulan text embeds...")
        text_token_id = mulan_model.text_to_token_ids(text=item[1], device=device)
        text_emb = mulan_model(text_token_id, data_type="text")
        text_embs.append(text_emb)
        prompts.append(item[1])
        categories.append(item[0])
    if prompts_group == "audio_prompt":
        print("Loading musiccaps_audio.pyt...")
        audio_embs = torch.load(
            "/mnt/bn/audio-diffusion/data/musiccaps/musiccaps_audio.pyt"
        ).to(device)
        print("Loaded")
    else:
        audio_embs = torch.zeros([len(items), 128]).to(device)
    text_embs = torch.cat(text_embs, dim=0)
    return text_embs, audio_embs, prompts, categories
