import argparse
import os

from einops import rearrange
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from recipes.mulan.utils.calculate_film_audio_embeds import load_film_model
import soundfile as sf
import time


REGION = "US"
if REGION == "US":
    # NOTE US TOS CONFIG
    TOS_PATH = "  http://tosv.byted.org/obj/cb-bucket-us/"
    TOS_bucket = "cb-bucket-us"
    accessKey = "RJ40ZMJQBZ27UYZX48R1"
else:
    # NOTE CN TOS CONFIG
    TOS_PATH = "  http://tosv.byted.org/obj/cb-bucket/"
    TOS_bucket = "cb-bucket"
    accessKey = "XN651TDBI79VVQR4JON3"
# Init TOS
import bytedtos

tos = bytedtos.Client(TOS_bucket, accessKey, timeout=99999999999)

def upload_one_file_to_tos(key: str, data: str):
    """
    key: the expected tos url, e.g. the TOS url will be
        f"  http://tosv.byted.org/obj/cb-bucket-us/{key}"
    data: the file path of data to be uploaded
    """
    try:
        tos.head_object(key)
        print(f"File {key} already exist")
    except:
        # if we don't have this file on the cloud, upload it
        try:
            tos.put_object(key, open(data, "rb"))
            time.sleep(1)
        except:
            print(f"Bad {data}")


def fine_sim_matrix(text_vec, music_vec):
    r"""Prepare logits for fine loss calculation."""
    # Ref: https://arxiv.org/abs/2111.07783
    music_vec_t = rearrange(music_vec, "b s d -> b d s")

    # For loop, (we cannot use torch.matmul on full batch)
    score1 = []
    score2 = []
    for i in range(len(text_vec)):
        def mm(m_v):
            dot_product = torch.matmul(text_vec[i:i+1], m_v)
            s1 = torch.mean(torch.max(dot_product, dim=-1)[0], dim=-1)
            s2 = torch.mean(torch.max(dot_product, dim=-2)[0], dim=-1)
            return s1, s2
        s1, s2 = torch.vmap(mm, chunk_size=32)(music_vec_t)
        score1.append(s1.squeeze(1))
        score2.append(s2.squeeze(1))
    score1 = torch.stack(score1)
    score2 = torch.stack(score2)

    # Sum up the scores
    score = score1 + score2
    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--model_version",
        type=str,
        default="film_tiny_mix_all"
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device_id}")

    # # to compute google prompts
    # f = "/mnt/bn/mm-data/user/dongguo/musiclm/test_cases/google_prompts.csv"
    # df = pd.read_csv(f)
    # text_dset = df["text"].values.tolist()
    f = "/mnt/bn/mm-data/user/dongguo/musiclm/test_cases/long_text_prompts.txt"
    with open(f, "r") as fin:
        lines = fin.readlines()
    text_dset = [l.strip() for l in lines]

    audio_folder = "/mnt/bn/mm-data/projects/mulan/testing_embed/non_vocal_30k_clips_20s"
    audio_emb_folder = "/mnt/bn/mm-data/projects/mulan/testing_embed/film_max_rank_0_39/nonvocal_30k_emb"
    cache_folder = ".retrieval_cache"
    output_csv = "long_text_prompts_nonvocal_film_mix_39.csv"
    os.makedirs(cache_folder, exist_ok=True)

    nsamples = len(text_dset)

    # load model
    film_modules = load_film_model(args)

    # load csv to list using pandas
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    text_embeds = np.zeros(shape=[nsamples, 250, 32], dtype=np.float32)

    with torch.no_grad():
        for xl in tqdm(range(0, len(text_dset), 512)):
            xr = min(len(text_dset), xl + 512)
            data = {}
            encodings = tokenizer(
                text_dset[xl:xr],
                padding="max_length",
                truncation=True,
                max_length=250,  # for most recent film
                return_tensors="pt",
            )
            data["input_ids"] = encodings["input_ids"].to(device)
            data["token_type_ids"] = encodings["token_type_ids"].to(device)
            data["attention_mask"] = encodings["attention_mask"].to(device)

            text_embeds[xl:xr, :] += (
                film_modules["film"].text_encoder(**data).cpu().data.numpy()
            )
    
    film_modules["film"].to("cpu")

    audio_files = []
    audio_embeds = []
    for audio_file in tqdm(os.listdir(audio_emb_folder)):
        audio_id = audio_file.split(".")[0].split("_")[0]
        audio_embeds.append(torch.load(f"{audio_emb_folder}/{audio_file}").squeeze(1))
        audio_files.append(audio_file + ".clip_0.wav")
        audio_files.append(audio_file + ".clip_1.wav")
        audio_files.append(audio_file + ".clip_2.wav")
    audio_embeds = torch.cat(audio_embeds, dim=0)

    # Calculate sim score
    text_embeds = torch.tensor(text_embeds).to(device)
    score = fine_sim_matrix(text_embeds, audio_embeds)
    vs, xs = torch.topk(score, k=8, dim=1)
    xs = xs.cpu().data.numpy().astype(np.int32)
    vs = vs.cpu().data.numpy()
    summary = []
    all_audios = []
    for i in range(len(xs)):
        row = []
        row.append(text_dset[i])
        used = False
        for k in range(xs.shape[1]):
            for x in row:
                if audio_files[xs[i, k]].split(".clip_")[0] in repr(x):
                    used = True
                    continue
            if used == True:
                used = False
                continue
            if audio_files[xs[i, k]] not in row:
                row.extend([vs[i, k], audio_files[xs[i, k]]])
                all_audios.append(audio_files[xs[i, k]])
            if len(row) >= 7:
                break
        summary.append(row)
    all_audios = list(set(all_audios))
    all_audios = [f.replace(".npy", "").replace(".pt", "") for f in all_audios]

    audio_index = {}
    for f in os.listdir(audio_folder):
        audio_index[f.split(".")[0]] = f

    ## Convert selected audios from npy to wav, and save in a cache folder
    for audio_file in tqdm(all_audios):
        audio_name = audio_file.split(".clip_")[0]
        audio_id = int(audio_file.replace(".wav", "")[-1])
        audio = np.load(f"{audio_folder}/{audio_index[audio_name]}")
        audio = audio / 32768.0
        sf.write(f"{cache_folder}/{audio_file}", audio[audio_id, :], 24000)
    
    upload_audio_files = os.listdir(cache_folder)
    for audio_file in tqdm(upload_audio_files):
        audio_path = f"{cache_folder}/{audio_file}"
        audio_key = audio_file
        upload_one_file_to_tos(audio_key, audio_path)

    for i in range(len(summary)):
        for j in [2, 4, 6]:
            summary[i][j] = "http://tosv.byted.org/obj/cb-bucket-us/" + summary[i][
                j
            ].replace(".pt", "").replace(".npy", "")
    summary = pd.DataFrame(
        summary,
        columns=[
            "text",
            "score_1",
            "top_1",
            "score_2",
            "top_2",
            "score_3",
            "top_3",
        ],
    )
    summary.to_csv(output_csv)
