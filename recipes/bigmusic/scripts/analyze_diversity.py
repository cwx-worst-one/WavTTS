import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import json

from pathlib import Path
from recipes.musiclm.requires.model_initializer import init_mulan
from recipes.musiclm.inference.utils import load_wav


def load_audio_tensor(audio_path, device):
    wav_tensor = torch.tensor(load_wav(str(audio_path), sr=24000)).to(device)
    return wav_tensor.unsqueeze(0)


def main(args):
    device = torch.device("cuda:0")
    print(f"Loading MuLan from {args.mulan_ckpt}")
    mulan = init_mulan(
        hpath=args.mulan_ckpt,
        local_rank=0,
        cache_dir=".module_cache/musiclm",
        version="g4",
    )
    mulan_model = mulan["mulan"]
    mulan_infer_fn = mulan["mulan_infer_fn"]

    for input_dir in args.input_dirs:
        print(f"--- {input_dir} ---")
        wav_groupings = {}
        wav_count = 0
        for wav_fp in list(Path(input_dir).glob("**/*.generated.wav")):
            name = os.path.basename(wav_fp).split(".")[0]
            if name not in wav_groupings:
                wav_groupings[name] = []
            wav_groupings[name].append(wav_fp)
            wav_count += 1
        print(f"Loaded {len(wav_groupings)} groups with {wav_count} wavs")

        diversity_scores = {}
        for group, wav_fps in wav_groupings.items():
            print(f"...{group}")
            mulan_embeds = []
            for wav_fp in wav_fps:
                audio = load_audio_tensor(wav_fp, device)
                to_pad = 10 * 24000 - audio.shape[-1]
                if to_pad > 0:
                    audio = F.pad(audio, (0, to_pad))
                with torch.no_grad():
                    mulan_emb = mulan_infer_fn(
                        model=mulan_model,
                        music=audio.float(),
                        device=device,
                        shift_seconds=5,
                    )
                mulan_embeds.append(mulan_emb)
            x = []
            y = []
            for i in range(len(mulan_embeds)):
                for j in range(i + 1, len(mulan_embeds)):
                    x.append(mulan_embeds[i])
                    y.append(mulan_embeds[j])
            sim = F.cosine_similarity(torch.vstack(x), torch.vstack(y))
            diversity_scores[group] = sim.mean().item()

        scores = list(diversity_scores.values())
        mean, std = np.mean(scores), np.std(scores)
        diversity_scores["mean"] = mean
        diversity_scores["std"] = std
        print(f"score: {mean} +/- {std}")
        with open(os.path.join(input_dir, "diversity_scores.json"), "w") as f:
            json.dump(diversity_scores, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+")
    parser.add_argument(
        "--mulan_ckpt",
        default="/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-median_rank_1=110-kaggle-minimal.ckpt",
    )
    args = parser.parse_args()
    main(args)