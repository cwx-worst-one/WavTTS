# run the codes in sami_ai_model:
import pdb
from curses import KEY_SLEFT
import pdb


def clean_concat(row):
    text = row['human_label_texts']
    text_lst = text.split('\n')
    new_lst = list()
    for tt in text_lst:
        if tt.startswith("Instrument:") or \
            tt.startswith("Genre:") or \
            tt.startswith("Theme:") or \
            tt.startswith("Mood:") or \
            tt.startswith("Places:") or \
            tt.startswith("Musician experience level:") or\
            tt.startswith("Epochs:"):
            cu_text = tt.split(":")[1].strip()
            new_lst.append(cu_text)
    result = "".join(new_lst)
    return result

COMBINATIONS = {1: ['instrument', 'genre', 'theme', 'mood'],
                2: ['genre', 'theme', 'mood'],
                3: ['instrument'],
                4: ['instrument', 'musician experience level'],
                5: ['genre', 'theme', 'mood', 'places'],
                6: ['instrument','genre','theme','mood','places','musician experience level','epochs']    
                }
COMB_NUM = 1
def pick(info, comb_num):
    text_lst = list()
    for kk in COMBINATIONS[comb_num]:
        text_lst.append(info[kk])
    return "".join(text_lst)


def clean(row):
    text = row['human_label_texts']
    text_lst = text.split('\n')
    info = dict()
    keys = ['instrument', 
            'genre', 
            'theme', 
            'mood', 
            'places', 
            'musician experience level',
            'epochs']
    for kk in keys:
        info[kk] = ""
    
    for tt in text_lst:
        if tt.startswith("Instrument:"):
            info['instrument'] = tt.split(":")[1].strip()
        elif tt.startswith("Genre:"):
            info['genre'] = tt.split(":")[1].strip()
        elif tt.startswith("Theme:"):
            info['theme'] = tt.split(":")[1].strip()
        elif tt.startswith("Mood:"):
            info['mood'] = tt.split(":")[1].strip()
        elif tt.startswith("Places:"):
            info['places'] = tt.split(":")[1].strip()
        elif tt.startswith("Musician experience level:"):
            info['musician experience level'] = tt.split(":")[1].strip()
        elif tt.startswith("Epochs:"):
            info['epochs'] = tt.split(":")[1].strip()
    result = pick(info, COMB_NUM)
    return result





def calculate_prompt_text_embeds():
    text_dset = [
        s.lower()
        for s in [
            "Healing",
            "Lonely",
            "Tense",
            "Excited",
            "Funny",
            "Memory",
            "Angry",
            "Chill",
            "Happy",
            "Sorrow",
            "Dynamic",
            "Romantic",
            "Inspirational",
        ]
    ]
    extra_tags = [
        "joyful",
        "sad",
        "angry",
        "calm",
        "energetic",
        "melancholic",
        "hopeful",
        "inspiring",
        "mysterious",
        "romantic",
        "nostalgic",
        "uplifting",
        "aggressive",
        "soothing",
        "sentimental",
        "cheerful",
        "relaxing",
        "dramatic",
        "playful",
        "scary",
        "dark",
        "intense",
        "peaceful",
        "gloomy",
        "dreamy",
        "exciting",
        "fun",
        "humorous",
        "passionate",
        "sorrowful",
        "tense",
        "tranquil",
        "warm",
        "whimsical",
        "anxious",
        "distant",
        "reflective",
        "empowering",
        "sensual",
        "heartwarming",
    ]

    for tag in extra_tags:
        if tag not in text_dset:
            text_dset.append(tag)

    import pandas as pd
    # to compute google prompts
    # f = "/mnt/bn/mm-data/user/dongguo/musiclm/test_cases/google_prompts.csv"
    # df = pd.read_csv(f)
    # text_dset = df["text"].values.tolist()

    f = "/mnt/bn/weituo-nas/music_edit/unify_repo/tmp/data/test_set/5k_non_vocal/non_vocal_half.csv"
    df = pd.read_csv(f)
    text_dset = df.apply(clean, axis=1).tolist()

    nsamples = len(text_dset)

    import argparse
    import pickle
    import json
    import torch
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--model_version", type=str, default="mulan_247")
    args = parser.parse_args()
    device = f"cuda:{args.device_id}"

    if args.model_version == "mulan_127":
        args.ckpt_path = "/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_1b_gpt_all/mulan-step=036000-median_rank_0=127-kaggle.ckpt"
    elif args.model_version == "mulan_170":
        assert args.model_version == "mulan_170"
        args.ckpt_path = "/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_g4/mulan-step=044800-median_rank_1=170-kaggle.ckpt"
    elif args.model_version == "mulan_247":
        args.ckpt_path = "/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_v1.0/mulan-step=033600-median_rank_0=247-kaggle.ckpt"
    else:
        args.ckpt_path = "/mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_mme_0701_supcon/checkpoints/mulan-step=006600-median_rank_0=228-kaggle.ckpt"

    def load_mulan_model(args):
        device_id = args.device_id
        from recipes.audio_lm.requires.model_initializer import init_mulan

        mulan_model = init_mulan(
            args.ckpt_path, device_id, cache_dir=None, version="g4"
        )
        mulan_model["mulan"].eval()
        assert mulan_model["mulan"].training is False
        return mulan_model

    # load model
    mulan_modules = load_mulan_model(args)

    # load csv to list using pandas
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    embeds = np.zeros(shape=[nsamples, 512], dtype=np.float32)

    with torch.no_grad():
        for xl in tqdm(range(0, len(text_dset), 512)):
            xr = min(len(text_dset), xl + 512)

            data = {}
            encodings = tokenizer(
                text_dset[xl:xr],
                padding="max_length",
                truncation=True,
                max_length=400,  # for most recent mulan
                return_tensors="pt",
            )
            data["input_ids"] = encodings["input_ids"].to(device)
            data["token_type_ids"] = encodings["token_type_ids"].to(device)
            data["attention_mask"] = encodings["attention_mask"].to(device)

            embeds[xl:xr, :] += (
                mulan_modules["mulan"].text_encoder(**data).cpu().data.numpy()
            )
    pickle.dump(
        [text_dset, embeds],
        open(
            f"/opt/tiger/mulan/text_embeds/text_embeds_5k_half_247_comb{COMB_NUM}.pkl",
            "wb",
        ),
    )

    with open(f"/opt/tiger/mulan/text_embeds/text_embeds_5k_half_247_comb{COMB_NUM}.npy", "wb") as ff:
        np.save(ff, embeds)

    print("so far so good?")



# def calculate_mulan_audio_embeds():
import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm

def load_mulan_model(args):
    device_id = args.device_id
    from recipes.audio_lm.requires.model_initializer import init_mulan

    mulan_model = init_mulan(
        args.ckpt_path, device_id, cache_dir=None, version="g4"
    )
    mulan_model["mulan"].eval()
    assert mulan_model["mulan"].training is False
    return mulan_model

@torch.no_grad()
def get_mulan_embed(mulan_modules, x):
    # [b, 1, d]
    return mulan_modules["mulan_infer_fn"](
        model=mulan_modules["mulan"],
        music=x.float()[:, 0 : 24000 * 10],
        device=x.device,
    ).unsqueeze(1)

def run_mulan_inference(args):
    # load model
    device = torch.device(f"cuda:{args.device_id}")
    # load model
    mulan_modules = load_mulan_model(args)

    audio_files = sorted(glob.glob(f"{args.input_folder}/*.npy"))
    print("\n\n calculating embeddings for {len(audio_files)} audios\n\n")
    for file in tqdm(audio_files):
        audio_clip = np.load(file)  # f"{input_folder}/{file}")
        audio_clip = torch.from_numpy((audio_clip / 32768.0).astype("float32")).to(
            device
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                audio_emb = get_mulan_embed(mulan_modules, audio_clip)

        stem = str(os.path.basename(file)).split(".")[0]
        torch.save(audio_emb, f"{args.target_folder}/{stem}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/opt/tiger/mulan/non_vocal_half_clips_20s",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="/opt/tiger/mulan/music_embeds",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        # default="/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_g4/mulan-step=044800-median_rank_1=170-kaggle.ckpt"
        # default="/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_1b_gpt_all/mulan-step=036000-median_rank_0=127-kaggle.ckpt",
        # default="/mnt/bn/mm-data/projects/mulan/ckpts/mme/mulan-step=002400-median_rank_0=232-kaggle.ckpt"
        default="/mnt/bn/mm-data/projects/mulan/ckpts/mulan_nonvocal/mulan_v1.0/mulan-step=033600-median_rank_0=247-kaggle.ckpt"
    )
    args = parser.parse_args()
    os.makedirs(args.target_folder, exist_ok=True)
    run_mulan_inference(args)

# calculate_prompt_text_embeds()
