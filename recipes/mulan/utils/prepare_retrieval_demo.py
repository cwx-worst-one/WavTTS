# def experiment_230605_mulan_mcc30K_retrieval():
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pandas as pd
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

audio_folder = "/mnt/bn/mm-data/projects/mulan/testing_embed/clips_20s"
assert os.path.exists(audio_folder)

cache_folder = "/opt/tiger/mulan/cache_audios"
os.makedirs(cache_folder, exist_ok=True)

text_emb_folder = (
    "/mnt/bn/mm-data/projects/mulan/testing_embed/vocal_rank_0_228/text_embeds_30k_vocal_mulan-step=006600-median_rank_0=228-kaggle.npy"
)
audio_emb_folder = (
    "/mnt/bn/mm-data/projects/mulan/testing_embed/vocal_rank_0_228/vocal_30k_emb"
)

## Collect the texts
f = "/mnt/bn/mm-data/user/dongguo/musiclm/test_cases/google_prompts.csv"
df = pd.read_csv(f)
texts = df["text"].values.tolist()
nsamples = len(texts)

## collect long prompt texts
# with open("/mnt/bn/mm-data/user/dongguo/musiclm/test_cases/long_text_prompts.txt", "r") as ff:
#     lines = ff.readlines()
# texts = [line.strip() for line in lines]
# info = {"category": ["short text from producer"] * len(texts), "text": texts}
# df = pd.DataFrame.from_dict(info)
# nsamples = len(texts)


## Collect the text embeds
# text_embeds = []
# for i in range(nsamples):
#     text_embeds.append(np.load(f"{text_emb_folder}/{i}.npy"))
# text_embeds = np.concatenate(text_embeds, axis=0)
text_embeds = torch.tensor(np.load(text_emb_folder))


## Load audio embeds
audio_files = []
audio_embeds = []
for audio_file in tqdm(os.listdir(audio_emb_folder)):
    audio_id = audio_file.split(".")[0].split("_")[0]
    audio_embeds.append(torch.load(f"{audio_emb_folder}/{audio_file}").squeeze(1))
    audio_files.append(audio_file + ".clip_0.wav")
    audio_files.append(audio_file + ".clip_1.wav")
    audio_files.append(audio_file + ".clip_2.wav")
audio_embeds = torch.cat(audio_embeds, dim=0)
# audio_embeds = audio_embeds / np.sqrt(
#     np.sum(audio_embeds**2, axis=1, keepdims=True)
# )

## Calculate the cosine similarity and find top-K matching music
# cross_sims = text_embeds@audio_embeds.T
text_embeds = text_embeds.to(audio_embeds.device)
cross_sims = torch.einsum('md,nd->mn', text_embeds, audio_embeds)
vs, xs = torch.topk(torch.tensor(cross_sims), k=8, dim=1)
xs = xs.cpu().data.numpy().astype(np.int32)
vs = vs.cpu().data.numpy()
summary = []
all_audios = []
for i in range(len(xs)):
    row = []
    row.append(df.category[i])
    text = df.text[i]
    row.append(text)
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
        if len(row) >= 8:
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
    for j in [3, 5, 7]:
        summary[i][j] = "  http://tosv.byted.org/obj/cb-bucket-us/" + summary[i][
            j
        ].replace(".pt", "").replace(".npy", "")
summary = pd.DataFrame(
    summary,
    columns=[
        "category",
        "text",
        "score_1",
        "top_1",
        "score_2",
        "top_2",
        "score_3",
        "top_3",
        # "score_4",
        # "top_4",
        # "score_5",
        # "top_5",
    ],
)
summary.to_csv("vocal_mulan_30k_rank_0_228.csv")
