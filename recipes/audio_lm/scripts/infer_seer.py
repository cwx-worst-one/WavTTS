import argparse
import os
import random
import re
import time
import unicodedata

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from tqdm import tqdm

from samantha.utils.hparams import DotDict


def set_seed(seed=1996):
    # reproduction setting
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    return


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def save_wav(audio, output_file, sr=24000):
    from scipy.io.wavfile import write

    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


def load_wav(path):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=24000)
    else:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(24000)
        wav = np.asarray(audio.get_array_of_samples())
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav


def sample(predict_logits, temp, mode="naive"):
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=1)  # [b, d]
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample().unsqueeze(1).to(args.device)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thres=args.gt)
        samples = gumbel_sample(predict_logits, temp).unsqueeze(dim=1)
    else:
        raise NotImplementedError()
    return samples


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.95):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def text2semantic(semantic_model, mulan_tokens, mulan_token_sep):
    bs = mulan_tokens.size(0)
    eos_ids = (
        torch.zeros([bs, 1], dtype=torch.long, device=args.device) + 1024
    )  # offset: w2v-bert
    mulan_tokens = mulan_tokens + 1024 + 1  # [b, 12] # offset: w2-vert + EOS 1
    if mulan_token_sep:
        mulan_tokens = (
            mulan_tokens + torch.arange(mulan_tokens.size(1)).to(args.device) * 1024
        )

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * semantic_frame_rate + semantic_offset
        if end >= args.duration * semantic_frame_rate + semantic_offset:
            end = args.duration * semantic_frame_rate + semantic_offset
            beg = end - (10 * semantic_frame_rate + semantic_offset)
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += stride_in_sec * semantic_frame_rate
    prev_end = 0
    semantic_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        if cache_len == 0:
            input_tokens = torch.cat([mulan_tokens, eos_ids], dim=1)
        else:
            prefix_semantic_samples = semantic_samples[:, cur_beg : cur_beg + cache_len]
            input_tokens = torch.cat(
                [mulan_tokens, eos_ids, prefix_semantic_samples], dim=1
            )
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for _ in pbar:
            pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
            semantic_outputs = semantic_model(
                input_tokens, past_key_values=past_key_values, use_cache=True
            )
            logits = semantic_outputs["logits"]  # [b, t, d]
            predict_logits = logits[:, -1, 0:1024]
            samples = sample(predict_logits, temp=args.st, mode=args.sample_mode)
            past_key_values = semantic_outputs["past_key_values"]
            input_tokens = samples
            if semantic_samples is None:
                semantic_samples = samples
            else:
                semantic_samples = torch.cat([semantic_samples, samples], dim=1)
    return semantic_samples


def coarse2fine(fine_model, coarse_samples):
    bs = coarse_samples.size(0)
    eos_ids = (
        torch.zeros(size=[bs, 1], dtype=torch.long, device=args.device) + num_res * 1024
    )  # [b, 1]

    fine_samples = None
    input_tokens = torch.cat([coarse_samples, eos_ids], dim=1)
    past_key_values = None

    pbar = tqdm(range(4000))
    for i in pbar:
        pbar.set_description("Fine")
        fine_outputs = fine_model(
            input_tokens, past_key_values=past_key_values, use_cache=True
        )
        logits = fine_outputs["logits"]  # [b, t, d]
        layer_idx = i % num_fine + num_coarse
        predict_logits = logits[
            :, -1, layer_idx * 1024 : (layer_idx + 1) * 1024
        ]  # [b, d]
        samples = sample(predict_logits, temp=args.ft, mode=args.sample_mode)
        samples = samples + layer_idx * 1024
        past_key_values = fine_outputs["past_key_values"]
        input_tokens = samples
        if fine_samples is None:
            fine_samples = samples
        else:
            fine_samples = torch.cat([fine_samples, samples], dim=1)
    return fine_samples


def ss_decode(ss_dec, coarse_samples, fine_samples):
    bs = coarse_samples.size(0)
    coarse_samples = coarse_samples.view([bs, -1, num_coarse])
    fine_samples = fine_samples.view([bs, -1, num_fine])
    vqgan_inputs = (
        torch.cat([coarse_samples, fine_samples], dim=2)
        - torch.arange(num_res).to(args.device) * 1024
    )  # [b, t, n_codebook]
    vqgan_inputs = vqgan_inputs.transpose(
        1, 2
    )  # [b, t, n_codebook] -> [b, n_codebook, t]
    wavs = ss_dec(vqgan_inputs).squeeze(1)
    return wavs


def gather_prompts(mulan_model):
    items = []
    if args.prompts_group in ["google", "all"]:
        import pandas as pd

        df = pd.read_csv(
            "/mnt/bn/audio-diffusion/data/google_prompts/google_prompts.csv"
        )
        for _, row in df.iterrows():
            items.append([row["category"], row["text"]])

    if args.prompts_group in ["google_short"]:
        import pandas as pd

        df = pd.read_csv(
            "/mnt/bn/audio-diffusion/data/google_prompts/google_prompts.csv"
        )
        for _, row in df.iterrows():
            if row["category"] == "10s Audio Generation From Text":
                items.append([row["category"], row["text"]])

    if args.prompts_group in ["google_long"]:
        import pandas as pd

        df = pd.read_csv(
            "/mnt/bn/audio-diffusion/data/google_prompts/google_prompts.csv"
        )
        for _, row in df.iterrows():
            if row["category"] == "Audio Generation From Rich Captions":
                items.append([row["category"], row["text"]])

    if args.prompts_group in ["musiccaps-asp", "all"]:
        import pandas as pd

        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["aspect_list"]])

    if args.prompts_group in ["musiccaps-cap", "all"]:
        import pandas as pd

        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["text"]])

    if args.prompts_group in ["sami", "all"]:
        from webdataset import WebDataset

        basedir = "/mnt/bn/audio-diffusion/data/mulan_prompts"
        tar_file_paths = [
            f"{basedir}/sami_prompts.tar.mulan149",
            f"{basedir}/genre_stats.tar.mulan149",
            f"{basedir}/mood_stats.tar.mulan149",
            f"{basedir}/musiccaps_long_prompts.tar.mulan149",
            f"{basedir}/musiccaps_short_prompts.tar.mulan149",
            f"{basedir}/painting_desc.tar.mulan149",
            f"{basedir}/theme_stats.tar.mulan149",
        ]
        for tar_file_path in tar_file_paths:
            prefix = tar_file_path.split("/")[-1].split(".")[0]
            tar_file = WebDataset(tar_file_path).decode()
            for item in tar_file:
                items.append([prefix, item["text.txt"]])

    if args.prompts_group in ["gpt", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/gpt_text_prompts.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["gpt", line.strip()])

    if args.prompts_group in ["image", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/image_captions.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["image captions", line.strip()])

    if args.prompts_group in ["painting", "all"]:
        with open(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/image2text2music.txt", "r"
        ) as fp:
            for line in fp.readlines():
                items.append(["image2text2music", line.strip()])

    if args.prompts_group == "direct_prompt":
        for i in range(args.bs):
            items.append(["direct_prompt", args.direct_prompt])

    pbar = tqdm(range(len(items)))
    for i in pbar:
        pbar.set_description("Extracting mulan text embeds...")
        text_emb = mulan_inference(mulan_model, text=items[i][1], device=args.device)
        items[i].append(text_emb)
    return items


@torch.no_grad()
def main():
    set_seed(seed=args.seed)
    mulan_model = create_mulan_model(ckpts.mulan, args.device).eval()
    mulan_centers = (
        torch.from_numpy(np.load(ckpts.mulan_centers)).float().to(args.device)
    )
    items = gather_prompts(mulan_model)
    print("Loading semantic_model...")
    semantic_lit_module = (
        SemanticModule.load_from_checkpoint(ckpts.semantic, args.device)
        .eval()
        .to(args.device)
    )
    semantic_model = semantic_lit_module.model
    print("Loading coarse_model...")
    coarse_lit_module = (
        SeerCoarseModule.load_from_checkpoint(ckpts.coarse, args.device)
        .eval()
        .to(args.device)
    )
    print("Loading fine_model...")
    fine_model = (
        FineModule.load_from_checkpoint(ckpts.fine, args.device)
        .eval()
        .to(args.device)
        .model
    )

    ss_dec = torch.jit.load(ckpts.ss_dec).eval()

    batch_items = [
        items[i * args.bs : (i + 1) * args.bs]
        for i in range((len(items) + args.bs - 1) // args.bs)
    ]
    for rd in range(args.rounds):
        start_time = time.time()
        for batch_idx, batch in enumerate(batch_items):
            text_embs = torch.cat([b[2] for b in batch], dim=0)
            mulan_ids, ds = mulan_rvq_indexs(text_embs, mulan_centers)
            semantic_samples = text2semantic(
                semantic_model, mulan_ids, semantic_lit_module.mulan_token_sep
            )
            coarse_samples = coarse_lit_module.predict(
                mulan_ids, semantic_samples, sample_mode=args.sample_mode
            )
            fine_samples = coarse2fine(fine_model, coarse_samples)
            wavs = ss_decode(ss_dec, coarse_samples, fine_samples)
            for wav_idx, wav in enumerate(wavs):
                wav_dir = os.path.join(filepath_prefix, batch[wav_idx][0])
                os.makedirs(wav_dir, exist_ok=True)
                if args.prompts_group == "direct_prompt":
                    fp = os.path.join(
                        wav_dir, f"{slugify(batch[wav_idx][1])[:128]}.{wav_idx}"
                    )
                else:
                    fp = os.path.join(
                        wav_dir, f"{slugify(batch[wav_idx][1])[:128]}.{rd}"
                    )
                print(f"[Saving] {fp}")
                save_wav(wav.cpu().numpy(), fp + ".wav", sr=sample_rate)
        print(f"[Elapsed Time] {time.time() - start_time}")


if __name__ == "__main__":

    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--rounds", type=int, default=10, help="How many rounds to run"
    )
    parser.add_argument(
        "-b", "--bs", type=int, default=20, help="Batch size for each forward pass"
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=10, help="Duration in seconds"
    )
    parser.add_argument("--sample_mode", choices=["naive", "gumbel"], default="naive")
    parser.add_argument(
        "--st", type=float, default=1.0, help="Semantic decoder sampling temperature"
    )
    parser.add_argument(
        "--ct", type=float, default=0.9, help="Coarse decoder sampling temperature"
    )
    parser.add_argument(
        "--ft", type=float, default=0.8, help="Fine decoder sampling temperature"
    )
    parser.add_argument("--gt", type=float, default=0.9, help="Gumbel sample threshold")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--prompts_group",
        choices=[
            "all",
            "musiccaps-cap",
            "musiccaps-asp",
            "google",
            "sami",
            "painting",
            "gpt",
            "image",
            "direct_prompt",
            "google_short",
            "google_long",
        ],
        default="google_short",
    )
    default_prompt = ""
    parser.add_argument("--direct_prompt", type=str, default=default_prompt)

    args = parser.parse_args()

    # Hparams
    num_res = 12
    num_coarse = 4
    num_fine = num_res - 4
    sample_rate = 24000
    frequency = 24000 // 480
    semantic_frame_rate = 25
    semantic_offset = 0
    stride_in_sec = 5

    # Checkpoint paths
    ckpts = {}
    from recipes.audio_lm.lit_modules import SeerCoarseModule, SemanticModule
    from recipes.audio_lm.lit_modules.v4_1.lit_fine import FineModule
    from recipes.audio_lm.requires.mulan.mulan_infer_g4 import (
        create_mulan_model,
        mulan_inference,
        mulan_rvq_indexs,
    )

    ckpts[
        "semantic"
    ] = "/mnt/bn/zongyu-lq/logs/semantic_sparse/mulan_g4/checkpoints/step=035216-val_accu_0=35.57.ckpt"  # noqa
    ckpts[
        "coarse"
    ] = "/mnt/bn/zongyu-lq/logs/coarse_seer/version_1.0/checkpoints/step=036000-accu=27.55.ckpt"  # noqa
    ckpts[
        "fine"
    ] = "/mnt/bn/zongyu-lq/ckpts/musiclm/epoch=03-step=193000-accu=18.92.ckpt"
    ckpts["ss_dec"] = "/mnt/bn/zongyu-lq/ckpts/soundstream/190k/ss_decoder_0.pt"
    ckpts[
        "mulan"
    ] = "/mlx/users/zongyu.yin/playground/samantha/.module_cache/musiclm_3ar_1024x12_x480/mulan-step=044800-median_rank_1=170-kaggle.ckpt"  # noqa
    ckpts[
        "mulan_centers"
    ] = "/mlx/users/zongyu.yin/playground/samantha/.module_cache/musiclm_3ar_1024x12_x480/kmeans_minibatch_codebook-mulan1b_g4_170-1024x12.npy"  # noqa

    ckpts = DotDict(ckpts)
    filepath_prefix = "outputs/"
    filepath_prefix += "MusicLM_seer"
    filepath_prefix += f"_{args.prompts_group}"
    filepath_prefix += f"_Mode_{args.sample_mode}"
    filepath_prefix += f"_St_{args.st}"
    filepath_prefix += f"_Ct_{args.ct}"
    filepath_prefix += f"_Ft_{args.ft}"
    filepath_prefix += f"_Gt_{args.gt}"
    filepath_prefix += f"_Seed_{args.seed}"

    main()
