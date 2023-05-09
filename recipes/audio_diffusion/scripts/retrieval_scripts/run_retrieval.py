import argparse

import numpy as np
import torch
import torch.nn.functional as F
from webdataset import WebDataset
from webdataset.writer import TarWriter


def load_segments(segments_fname, device):
    segments = []
    wav_paths = []
    with open(segments_fname, "r") as f:
        for line in f:
            fname = line.strip()
            print(f"Loading segments from {fname}...")
            seg = np.load(fname)
            segments.append(seg)
            wav_path = fname.split(".")
            wav_path[2] = "wav"
            for i in range(seg.shape[0]):
                wav_paths.append((".".join(wav_path), i))
    segments = np.concatenate(segments, axis=0)
    print(f"segments: {segments.shape}")
    assert segments.shape[0] == len(wav_paths)
    return torch.from_numpy(segments).to(device), wav_paths


def do_retrieval_top_k(segments, wav_paths, text_emb, score_fn, top_k):
    text_emb = torch.from_numpy(text_emb).to(segments.device)
    text_emb = text_emb.unsqueeze(dim=0)
    text_emb = text_emb.expand(segments.shape[0], text_emb.shape[1])
    scores = score_fn(segments, text_emb)
    indices = torch.argsort(scores, descending=True)[:top_k]
    retrieved_emb = F.normalize(segments[indices].mean(dim=0), p=2, dim=0)
    retrieved_wav = []
    for idx in indices:
        fname, i = wav_paths[idx.item()]
        retrieved_wav.append(np.load(fname)[i])
    retrieved_wav = np.stack(retrieved_wav, axis=0)
    score = score_fn(text_emb[:1], retrieved_emb.unsqueeze(dim=0))
    return (
        retrieved_emb.cpu().float().numpy(),
        retrieved_wav,
        score.cpu().float().numpy(),
        len(indices),
    )


def do_retrieval_search(segments, text_emb, score_fn, start_size, max_size, step_size):
    emb_dim = len(text_emb)
    text_emb = torch.from_numpy(text_emb).to(segments.device)
    text_emb = text_emb.unsqueeze(dim=0)
    scores = score_fn(segments, text_emb.expand(segments.shape[0], emb_dim))
    indices = torch.argsort(scores, descending=True)
    # Sorry for the ugly code!
    sizes = []
    curr_size = start_size
    while curr_size < max_size:
        curr_retrieved_emb = F.normalize(
            segments[indices[:curr_size]].mean(dim=0), p=2, dim=0
        )
        curr_text_emb_score = score_fn(
            text_emb[:1], curr_retrieved_emb.unsqueeze(dim=0)
        )
        curr_audio_emb_score = score_fn(
            segments,
            curr_retrieved_emb.unsqueeze(dim=0).expand(segments.shape[0], emb_dim),
        ).mean()
        sizes.append(
            (curr_size, curr_text_emb_score[0].item(), curr_audio_emb_score.item())
        )
        # Prepare for next round
        curr_size += step_size
    # Sort in descending scores
    sizes.sort(key=lambda x: -(x[1] + x[2]))
    # print(f"...sizes[:3] = {sizes[:3]}")
    best_size = sizes[0][0]
    retrieved_emb = F.normalize(segments[indices[:best_size]].mean(dim=0), p=2, dim=0)
    score = score_fn(text_emb[:1], retrieved_emb.unsqueeze(dim=0))
    return (retrieved_emb.cpu().float().numpy(), score.cpu().float().numpy(), best_size)


def main(args):
    assert args.mode in {"top_k", "search"}, f"Invalid mode: {args.mode}"
    device = torch.device("cuda:0")
    segments, wav_paths = load_segments(args.segments_fname, device=device)
    score_fn = torch.nn.CosineSimilarity()
    with open(args.tar_list_fname, "r") as f:
        for line in f:
            input_fname = line.strip()
            output_fname = f"{input_fname}.retrieved_with_wav"
            if args.mode == "top_k":
                output_fname = f"{output_fname}.top_{args.top_k}"
            else:
                output_fname = (
                    f"{output_fname}"
                    f".search_{args.start_size}"
                    f"_{args.max_size}_{args.step_size}"
                )
            print(f"--- Reading from {input_fname}, outputting to {output_fname}")
            fout = TarWriter(open(output_fname, "wb"))
            for item in WebDataset(input_fname).decode():
                text = item["text.txt"]
                if args.mode == "top_k":
                    retrieved_emb, retrieved_wav, score, size = do_retrieval_top_k(
                        segments, wav_paths, item["emb.npy"], score_fn, args.top_k
                    )
                else:
                    retrieved_emb, score, size = do_retrieval_search(
                        segments,
                        item["emb.npy"],
                        score_fn,
                        args.start_size,
                        args.max_size,
                        args.step_size,
                    )
                print(f"text: {text}, size: {size}, score: {score}")
                item["emb.npy"] = retrieved_emb
                item["score.npy"] = score
                item["wav.npy"] = retrieved_wav

                fout.write(item)
            fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument(
        "tar_list_fname",
        default="/mnt/bn/audio-diffusion/data/mulan_prompts/tar_list.txt",
        help="List of input tar paths",
    )
    args = parser.add_argument(
        "--segments_fname",
        default="/mnt/bn/audio-diffusion/data/mulan_retrieval/emb_and_wav_list.txt",
        help="List of numpy segments",
    )
    args = parser.add_argument(
        "--mode", default="top_k", help="Valid modes: [top_k, search]"
    )
    args = parser.add_argument(
        "--top_k", type=int, default=5, help="(top_k) How many embeddings to retrieve"
    )
    args = parser.add_argument(
        "--start_size",
        type=int,
        default=200,
        help="(search) Size of initial embedding set",
    )
    args = parser.add_argument(
        "--max_size",
        type=int,
        default=2000,
        help="(search) Max size of final embedding set",
    )
    args = parser.add_argument(
        "--step_size",
        type=int,
        default=20,
        help="(search) How many segments to add per step",
    )
    args = parser.parse_args()
    main(args)
