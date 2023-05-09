import glob
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from recipes.mulan.modules.pl_module import LitMuLanModule

# load model
device = torch.device("cuda:0")
ckpt_path = "mulan_exp/MuLan_large/g4/checkpoints/mulan-step=044800-median_rank_1=170-kaggle.ckpt"
litmodel = LitMuLanModule.load_from_checkpoint(ckpt_path)

# audio tower
litmodel.music_encoder.eval()
litmodel.music_encoder.to(device)
litmodel.music_encoder.mut.manually_to_device(device)
# text tower
litmodel.text_encoder.eval()
litmodel.text_encoder.to(device)

# load csv to list using pandas
text_path = "mcc_30k/google_prompts.csv"
index_list = pd.read_csv(text_path)["index"].to_list()
text_list = pd.read_csv(text_path)["text"].to_list()
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

for idx, text in zip(index_list, text_list):
    data = {}
    encodings = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=200,
        return_tensors="pt",
    )
    data["input_ids"] = encodings["input_ids"]
    data["token_type_ids"] = encodings["token_type_ids"]
    data["attention_mask"] = encodings["attention_mask"]

    text_emb = litmodel.text_encoder(
        data["input_ids"].to(device),
        data["attention_mask"].to(device),
        data["token_type_ids"].to(device),
    )
    torch.save(text_emb, f"google_prompts_text_embs/{idx}.pt")

input_path = "mcc_30k/mcc_30K_30s"
output_path = "mcc_30K_30s_emb"
files = glob.glob(f"{input_path}/*.npy")

for file in tqdm(files):
    audio_clip = np.load(file)
    audio_clip = torch.from_numpy((audio_clip / 32768.0).astype("float32"))

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            audio_emb = litmodel.music_encoder(
                audio_clip.unsqueeze(1).to(device), spec_aug=False
            )
    stem = str(os.path.basename(file)).split(".")[0]
    torch.save(audio_emb, f"{output_path}/{stem}.pt")
