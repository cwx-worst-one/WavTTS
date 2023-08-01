import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
from transformers import AutoTokenizer

from recipes.mulan.models.music_encoder import get_music_encoder
from recipes.mulan.models.text_encoder import get_text_encoder


class LitFiLMModule(pl.LightningModule):
    def __init__(
        self,
        music_encoder,
        text_encoder,
        emb_dim,
        spec_aug,
        text_seq_len,
        music_seq_len,
        lr,
        weight_decay,
        temperature,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        # music_encoder large vit, text_encoder bert
        self.music_encoder = get_music_encoder(music_encoder, emb_dim, "seq", music_seq_len)
        self.text_encoder = get_text_encoder(text_encoder, emb_dim, "seq")
        self.temperature = torch.log(torch.tensor(1 / temperature))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        # Validation outputs
        self.val_outputs = dict()

    def tokenize_text(self, text):
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=250,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        token_type_ids = tokenized_text["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def encode_text(self, text):
        input_ids, attention_mask, token_type_ids = self.tokenize_text(text)
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        text_embed = self.text_encoder(input_ids, attention_mask, token_type_ids)
        return text_embed

    def _shared_step(self, batch, spec_aug=False):
        music_embed = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=spec_aug)
        text_embed = self.text_encoder(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        return {
            "text_vec": text_embed,
            "music_vec": music_embed,
        }

def load_film_model(args):
    device_id = args.device_id
    device = torch.device(f"cuda:{device_id}")
    if args.model_version == "film_tiny_mix_all":
        ckpt_path = "/mnt/bn/mm-data/user/film_exp/best_ckpts/film_mix_all_25hz_16h_16l/mulan-step=022000-median_rank_0=39-kaggle.ckpt"
    else:
        raise NotImplementedError(f"Invalid model version: {args.model_version}")
    litmodel = LitFiLMModule.load_from_checkpoint(ckpt_path)
    # audio tower
    litmodel.music_encoder.eval()
    litmodel.music_encoder.to(device)
    litmodel.music_encoder.manually_to_device(device)
    # text tower
    litmodel.text_encoder.eval()
    litmodel.text_encoder.to(device)

    @torch.no_grad()
    def film_inference(
        model, text=None, music=None, avg=True, shift_seconds=1
    ):
        assert (text is not None) ^ (
            music is not None
        ), "text inputs and music input can only select one"

        if text is not None:
            emb = model.encode_text(text)

        if music is not None:
            music_encoder = model.music_encoder
            emb = music_encoder(music)
        return emb

    return {
        "film": litmodel,
        "film_infer_fn": film_inference,
    }

@torch.no_grad()
def get_film_embed(film_modules, x):
    # x.shape=(3, 2T)
    # batch.shape=(3, 1, T)
    batch = x.float()[:, 0 : 24000 * 10].unsqueeze(1)
    emb = film_modules["film_infer_fn"](
        model=film_modules["film"],
        music=batch,
    )
    return emb

def run_film_inference(args):
    # load model
    device = torch.device(f"cuda:{args.device_id}")
    # load model
    film_modules = load_film_model(args)

    audio_files = sorted(glob.glob(f"{args.input_folder}/*.npy"))
    audio_files = audio_files[args.shard_id :: args.n_shards]
    print(f"\n\n calculating embeddings for {len(audio_files)} audios\n\n")
    for file in tqdm(audio_files):
        audio_clip = np.load(file)
        audio_clip = torch.from_numpy((audio_clip / 32768.0).astype("float32")).to(
            device
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                audio_emb = get_film_embed(film_modules, audio_clip)

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
        default="/mnt/bn/mm-data/projects/mulan/testing_embed/clips_20s/",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="/mnt/bn/mm-data/projects/mulan/testing_embed/film_max_rank_0_39/vocal_30k_emb",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="film_tiny_mix_all"
    )
    args = parser.parse_args()
    os.makedirs(args.target_folder, exist_ok=True)
    run_film_inference(args)
