import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from samantha.utils.hparams import DotDict

from ..utils.utils import sample, save_wav, set_seed, slugify


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


def semantic2coarse(coarse_model, mulan_tokens, semantic_samples, mulan_token_sep):
    bs = mulan_tokens.size(0)
    eos_ids = (
        torch.zeros([bs, 1], dtype=torch.long, device=args.device)
        + 1024
        + num_coarse * 1024
    )  # offset: w2v-bert + coarse
    mulan_tokens = (
        mulan_tokens + 1024 + num_coarse * 1024 + 2
    )  # offset: w2v-bert + coarse + EOS 2
    if mulan_token_sep:
        mulan_tokens = (
            mulan_tokens + torch.arange(mulan_tokens.size(1)).to(args.device) * 1024
        )

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * frequency * num_coarse
        if end >= args.duration * frequency * num_coarse:
            end = args.duration * frequency * num_coarse
            beg = end - 10 * frequency * num_coarse
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += stride_in_sec * frequency * num_coarse

    prev_end = 0
    coarse_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        semantic_beg = int(cur_beg / frequency / num_coarse * semantic_frame_rate)
        semantic_end = semantic_beg + 10 * semantic_frame_rate + semantic_offset
        semantic_slice = semantic_samples[:, semantic_beg:semantic_end]
        if cache_len == 0:
            if args.mulan_free:
                input_tokens = torch.cat([semantic_slice, eos_ids], dim=1)
            else:
                input_tokens = torch.cat(
                    [mulan_tokens, eos_ids, semantic_slice, eos_ids + 1], dim=1
                )
        else:
            prefix_coarse_samples = coarse_samples[:, cur_beg : cur_beg + cache_len]
            if args.mulan_free:
                input_tokens = torch.cat(
                    [semantic_slice, eos_ids, prefix_coarse_samples], dim=1
                )
            else:
                input_tokens = torch.cat(
                    [
                        mulan_tokens,
                        eos_ids,
                        semantic_slice,
                        eos_ids + 1,
                        prefix_coarse_samples,
                    ],
                    dim=1,
                )
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for i in pbar:
            pbar.set_description(
                f"Coarse [{cur_beg} - {cur_end}] [{semantic_beg} - {semantic_end}]"
            )
            coarse_outputs = coarse_model(
                input_tokens, past_key_values=past_key_values, use_cache=True
            )
            logits = coarse_outputs["logits"]  # [b, t, d]
            layer_idx = i % num_coarse
            predict_logits = logits[
                :, -1, layer_idx * 1024 : (layer_idx + 1) * 1024
            ]  # [b,d]
            samples = sample(predict_logits, temp=args.ct, mode=args.sample_mode)
            samples = samples + 1024 + layer_idx * 1024  # [b, 1], # offset: w2v-bert
            past_key_values = coarse_outputs["past_key_values"]
            input_tokens = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
    coarse_samples = coarse_samples - 1024  # remove offset: w2v-bert
    return coarse_samples


def coarse2fine(fine_model, coarse_samples):
    bs = coarse_samples.size(0)
    eos_ids = (
        torch.zeros(size=[bs, 1], dtype=torch.long, device=args.device) + num_res * 1024
    )  # [b, 1]

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * frequency * num_fine
        if end >= args.duration * frequency * num_fine:
            end = args.duration * frequency * num_fine
            beg = end - 10 * frequency * num_fine
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += stride_in_sec * frequency * num_fine

    prev_end = 0
    fine_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        coarse_beg = int(cur_beg / num_fine * num_coarse)
        coarse_end = coarse_beg + 2000
        coarse_slice = coarse_samples[:, coarse_beg:coarse_end]  # [b, coarse*t]
        if cache_len == 0:
            input_tokens = torch.cat([coarse_slice, eos_ids], dim=1)
        else:
            prefix_fine_samples = fine_samples[:, cur_beg : cur_beg + cache_len]
            input_tokens = torch.cat(
                [coarse_slice, eos_ids, prefix_fine_samples], dim=1
            )
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for i in pbar:
            pbar.set_description(
                f"Fine [{cur_beg} - {cur_end}] [{coarse_beg} - {coarse_end}]"
            )
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

    # if args.prompts_group == "audio_prompt":
    #     import pandas as pd
    #     df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
    #     for _, row in df.iterrows():
    #         items.append(["audio prompt (musiccaps)", row["text"]])

    pbar = tqdm(range(len(items)))
    for i in pbar:
        pbar.set_description("Extracting mulan text embeds...")
        text_emb = mulan_inference(mulan_model, text=items[i][1], device=args.device)
        items[i].append(text_emb)
        # prompts.append(item[1])
        # categories.append(item[0])
    # if args.prompts_group == "audio_prompt":
    #     print("Loading musiccaps_audio.pyt...")
    #     audio_embs = torch.load(
    #         "/mnt/bn/audio-diffusion/data/musiccaps/musiccaps_audio.pyt"
    #     ).to(args.device)
    #     print("Loaded")
    # else:
    #     audio_embs = torch.zeros([len(items), 128]).to(args.device)
    # text_embs = torch.cat(text_embs, dim=0)
    return items


@torch.no_grad()
def main():
    set_seed(seed=args.seed)
    print("Loading semantic_model...")
    semantic_lit_module = (
        SemanticModule.load_from_checkpoint(ckpts.semantic, args.device)
        .eval()
        .to(args.device)
    )
    semantic_model = semantic_lit_module.model
    print("Loading coarse_model...")
    coarse_lit_module = (
        CoarseModule.load_from_checkpoint(ckpts.coarse, args.device)
        .eval()
        .to(args.device)
    )
    coarse_model = coarse_lit_module.model
    print("Loading fine_model...")
    fine_model = (
        FineModule.load_from_checkpoint(ckpts.fine, args.device)
        .eval()
        .to(args.device)
        .model
    )

    ss_dec = torch.jit.load(ckpts.ss_dec).eval()

    mulan_model = create_mulan_model(ckpts.mulan, args.device).eval()
    mulan_centers = (
        torch.from_numpy(np.load(ckpts.mulan_centers)).float().to(args.device)
    )
    items = gather_prompts(mulan_model)
    batch_items = [
        items[i * args.bs : (i + 1) * args.bs]
        for i in range((len(items) + args.bs - 1) // args.bs)
    ]
    for rd in range(args.rounds):
        start_time = time.time()
        for batch_idx, batch in enumerate(batch_items):
            mulan_embeds = torch.cat([b[2] for b in batch], dim=0)
            mulan_tokens, ds = mulan_rvq_indexs(mulan_embeds, mulan_centers)
            semantic_samples = text2semantic(
                semantic_model, mulan_tokens, semantic_lit_module.mulan_token_sep
            )
            print("Semantic samples: ", semantic_samples.size())
            if args.mulan_free:
                coarse_samples = semantic2coarse(
                    coarse_model, mulan_tokens, semantic_samples, False
                )
            else:
                coarse_samples = semantic2coarse(
                    coarse_model,
                    mulan_tokens,
                    semantic_samples,
                    coarse_lit_module.mulan_token_sep,
                )
            print("Coarse samples: ", coarse_samples.size())
            fine_samples = coarse2fine(fine_model, coarse_samples)
            print("Fine samples: ", fine_samples.size())
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
                # with open(".".join(fp.split(".")[:-1]) + ".txt", "w") as prompt_txt:
                #     prompt_txt.write(prompts[prompt_idx])
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
    parser.add_argument("--sample_mode", choices=["naive", "gumbel"], default="gumbel")
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
    parser.add_argument("--mert", action="store_true")
    parser.add_argument("--mulan_free", action="store_true")

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
    default_prompt = "acoustic guitar"
    parser.add_argument("--direct_prompt", type=str, default=default_prompt)

    # parser.add_argument("--save_tensor", action="store_true", help="Save logits and sampled tokens") # noqa
    args = parser.parse_args()

    # Hparams
    num_res = 12
    num_coarse = 4
    num_fine = num_res - 4
    sample_rate = 24000
    frequency = 24000 // 480

    stride_in_sec = 5

    # Checkpoint paths
    ckpts = {}
    from recipes.audio_lm.lit_modules.lit_models import SemanticModule

    if args.mulan_free:
        from recipes.audio_lm.lit_modules.v4_3.lit_coarse_3ar import CoarseModule
    else:
        from recipes.audio_lm.lit_modules.lit_models import CoarseModule
    from recipes.audio_lm.lit_modules.v4_1.lit_fine import FineModule

    # sparse 2.4B
    # ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_sparse/version_3.1/checkpoints/epoch=03-step=100000-accu=36.39.ckpt' # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse/version_3.1/checkpoints/epoch=00-step=46000-accu=27.13.ckpt' # noqa
    # sparse 0.3B
    # ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_sparse/version_3.2/checkpoints/epoch=18-step=39000-accu=35.62.ckpt' # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse/version_3.2/checkpoints/epoch=06-step=70000-accu=28.64.ckpt' # noqa
    # llama 0.4B - mulan191
    # ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_llama/406M/checkpoints/epoch=22-step=61000-accu=37.32.ckpt' # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_llama/415M/checkpoints/epoch=02-step=120000-accu=30.45.ckpt' # noqa
    # llama 0.4B - mulan115
    # ckpts['semantic'] = "/mnt/bn/zongyu-lq/logs/semantic_llama_mulan115/406M/checkpoints/epoch=08-step=22000-accu=36.29.ckpt" # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_llama_mulan115/415M/checkpoints/epoch=01-step=50000-accu=30.41.ckpt' # noqa
    # sparse 0.4B - mulan247
    # ckpts['semantic'] = "/mnt/bn/zongyu-lq/ckpts/musiclm/semantic_sparse_mulan247_429M_epoch=79-step=313000-accu=38.00.ckpt" # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse_mulan247/434M/checkpoints/epoch=09-step=156000-accu=30.36.ckpt' # noqa
    # sparse 0.4B - mulan247 gpt finetune
    # ckpts['semantic'] = "/mnt/bn/zongyu-lq/logs/semantic_sparse_gpt_finetune/429M/checkpoints/epoch=00-step=45000-accu=38.20.ckpt" # noqa
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse_gpt_finetune/434M/checkpoints/epoch=00-step=50000-accu=30.54.ckpt' # noqa
    # sparse - mulang4
    ckpts[
        "semantic"
    ] = "/mnt/bn/zongyu-lq/logs/semantic_sparse/mulan_g4/checkpoints/step=035216-val_accu_0=35.57.ckpt"  # noqa
    ckpts[
        "coarse"
    ] = "/mnt/bn/zongyu-lq/logs/coarse_sparse/mulan_g4/checkpoints/step=058010-val_accu_0=28.79.ckpt"  # noqa

    if args.mulan_free:
        # coarse mulan-free
        ckpts[
            "coarse"
        ] = "/mnt/bn/zongyu-lq/logs/coarse/311M/checkpoints/epoch=02-step=126000-accu=30.39.ckpt"  # noqa

    if args.mert:
        semantic_frame_rate = 75
        semantic_offset = -1
        # sparse 0.4B - MERT
        ckpts[
            "semantic"
        ] = "/mnt/bn/zongyu-lq/ckpts/musiclm/semantic_sparse_mert_429M_epoch=07-step=96000-accu=56.24.ckpt"  # noqa
        ckpts[
            "coarse"
        ] = "/mnt/bn/zongyu-lq/logs/coarse_sparse_mert/434M/checkpoints/epoch=05-step=132000-accu=30.60.ckpt"  # noqa
    else:
        semantic_frame_rate = 25
        semantic_offset = 0

    ckpts[
        "fine"
    ] = "/mnt/bn/zongyu-lq/ckpts/musiclm/epoch=03-step=193000-accu=18.92.ckpt"
    ckpts["ss_dec"] = "/mnt/bn/zongyu-lq/ckpts/soundstream/190k/ss_decoder_0.pt"

    from recipes.audio_lm.requires.mulan.mulan_infer_g4 import (
        create_mulan_model,
        mulan_inference,
        mulan_rvq_indexs,
    )

    # mulan 149
    # ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan_149.pt'
    # ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook-1024x12.npy' # noqa
    # mulan 191
    # ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan-step=042400-median_rank_0=191-kaggle.ckpt' # noqa
    # ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook_mulan_multi_dset_191.npy' # noqa
    # mulan 115
    # ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan-step=006400-median_rank_1=115-kaggle.ckpt' # noqa
    # ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook_mulan-1b-151_6M_ninit3.npy' # noqa
    # mulan 247
    # ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan-step=033600-median_rank_0=247-kaggle.ckpt' # noqa
    # ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook-mulan249-1024x12.npy' # noqa
    # mulan g4
    ckpts[
        "mulan"
    ] = "/mlx/users/zongyu.yin/playground/samantha/.module_cache/musiclm_3ar_1024x12_x480/mulan-step=044800-median_rank_1=170-kaggle.ckpt"  # noqa
    ckpts[
        "mulan_centers"
    ] = "/mlx/users/zongyu.yin/playground/samantha/.module_cache/musiclm_3ar_1024x12_x480/kmeans_minibatch_codebook-mulan1b_g4_170-1024x12.npy"  # noqa

    ckpts = DotDict(ckpts)
    filepath_prefix = "outputs/"
    filepath_prefix += (
        "MusicLM_sparse_mulang4_040523"
        + ("_free" if args.mulan_free else "")
        + ("_mert" if args.mert else "")
    )
    filepath_prefix += f"_Dur_{args.duration}"
    filepath_prefix += f"_{args.prompts_group}"
    filepath_prefix += f"_Mode_{args.sample_mode}"
    filepath_prefix += f"_St_{args.st}"
    filepath_prefix += f"_Ct_{args.ct}"
    filepath_prefix += f"_Ft_{args.ft}"
    filepath_prefix += f"_Gt_{args.gt}"
    filepath_prefix += f"_Seed_{args.seed}"

    main()
