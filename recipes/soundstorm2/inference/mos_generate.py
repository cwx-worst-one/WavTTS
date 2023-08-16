import argparse
import os

import torch
import torchaudio

from recipes.soundstorm.inference.semantic2audio import generate, load_model


def main(args):
    sample_rate = 24000
    device = "cuda"
    iterations = list(map(int, args.iterations.split(",")))
    print(f"iterations: {iterations}")
    score_strategies = args.score_strategies.split(",")
    print(f"score_strategies: {score_strategies}")
    temperatures = list(map(float, args.temperatures.split(",")))
    print(f"temperatures: {temperatures}")
    soundstorm = load_model(args.soundstorm_ckpt, device)

    with open(args.prompts, "r") as f:
        for line in f:
            semantic_path = line.strip()
            ary = semantic_path.split("/")
            category = ary[-2]
            fname = ary[-1].replace(".pt", "")
            print(f"--- {category} / {fname} ---")
            semantic_ids = torch.load(semantic_path).reshape(1, -1).to(device)
            sampled_audio = generate(
                soundstorm=soundstorm,
                semantic_tokens=semantic_ids,
                max_seq_len=10 * soundstorm.audio_model.frame_rate,
                iterations=iterations,
                score_strategies=score_strategies,
                temperature=temperatures,
            )
            sampled_audio = sampled_audio[0].cpu()
            # Save
            output_dir = os.path.join(args.output_dir, category)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{fname}.wav")
            torchaudio.save(output_path, sampled_audio, sample_rate, format="wav")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument("output_dir")
    args = parser.add_argument(
        "--prompts",
        default="/mnt/bn/audio-diffusion/data/semantic_sequences/semantic_sequences_text_prompt_collection_20230615/prompts.txt",
    )
    args = parser.add_argument(
        "--soundstorm_ckpt",
        default="/mnt/bn/audio-diffusion/ducle/logs/soundstorm_mha/mcc40m_conservative_filtered/checkpoints/step=075000-loss=0.0000.ckpt",
    )
    args = parser.add_argument(
        "--semantic_ckpt",
        default="/mnt/bn/audio-diffusion/ducle/logs/semantic_flash_llama/mcc40m_filtered_705m/checkpoints/step=081000-tr_loss=2.4451-val_loss_0=2.8561.ckpt",
    )
    args = parser.add_argument(
        "--iterations",
        # default="16,16,12,12,4,4,2,2,1,1,1,1",  # RTF ~= 0.28
        # default="48,32,24,16,8,4,2,2,1,1,1,1", # RTF ~= 0.50
        default="128,64,32,16,8,8,4,4,2,2,2,2",  # RTF ~= 1.23
    )
    args = parser.add_argument(
        "--score_strategies",
        default="random,random,random,random,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit,maskgit",
    )
    args = parser.add_argument(
        "--temperatures",
        # default="1.0,1.0,0.95,0.95,0.9,0.9,0.8,0.8,0.4,0.4,0.4,0.4",
        default="0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95",
    )
    args = parser.parse_args()
    main(args)
