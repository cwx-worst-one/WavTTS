import argparse
import os

import numpy as np
import torch
import webdataset


def main(args):
    segments_emb = None
    segments_wav = None
    output_split_id = 0
    with open(args.tar_list_fname, "r") as f:
        for line in f:
            tar_fname = line.strip()
            row_count = 0
            segment_count = 0
            print(f"--- Reading from {tar_fname} ---", flush=True)
            dataset = webdataset.WebDataset(f"pipe:hdfs dfs -cat {tar_fname}").decode()
            for item in dataset:
                # Get one embedding for every 10 seconds
                mulan_embs = item["acc_audio_emb.npy"]
                if item["acc.npy"].shape[0] < 240000:
                    continue
                wav = torch.from_numpy(item["acc.npy"]).unfold(0, 240000, 24000).numpy()
                if mulan_embs.shape[0] != wav.shape[0]:
                    length = min(mulan_embs.shape[0], wav.shape[0])
                    mulan_embs = mulan_embs[:length]

                if segments_emb is None:
                    segments_emb = mulan_embs
                else:
                    segments_emb = np.concatenate([segments_emb, mulan_embs], axis=0)

                if segments_wav is None:
                    segments_wav = wav
                else:
                    segments_wav = np.concatenate([segments_wav, wav], axis=0)

                row_count += 1
                segment_count += mulan_embs.shape[0]
                if row_count % 200 == 0:
                    print(f"...processed {row_count} rows", flush=True)
                if segment_count > 1000:
                    print(
                        (
                            f"Saving {segments_emb.shape[0]} segments_emb to"
                            f" {args.output_npy_fname} {output_split_id:06}"
                        ),
                        flush=True,
                    )
                    np.save(
                        os.path.join(
                            "/mnt/bn/audio-diffusion/data/mulan_retrieval/emb_and_wav",
                            args.output_npy_fname + f".emb.{output_split_id:06}",
                        ),
                        segments_emb,
                    )
                    np.save(
                        os.path.join(
                            "/mnt/bn/audio-diffusion/data/mulan_retrieval/emb_and_wav",
                            args.output_npy_fname + f".wav.{output_split_id:06}",
                        ),
                        segments_wav,
                    )
                    output_split_id += 1
                    segment_count = 0
                    del segments_emb
                    del segments_wav
                    segments_emb = None
                    segments_wav = None
            if segments_emb is not None and segments_wav is not None:
                print(
                    (
                        f"Saving {segments_emb.shape[0]} segments_emb to"
                        f" {args.output_npy_fname} {output_split_id:06}"
                    ),
                    flush=True,
                )
                np.save(
                    os.path.join(
                        "/mnt/bn/audio-diffusion/data/mulan_retrieval/emb_and_wav",
                        args.output_npy_fname + f".emb.{output_split_id:06}",
                    ),
                    segments_emb,
                )
                np.save(
                    os.path.join(
                        "/mnt/bn/audio-diffusion/data/mulan_retrieval/emb_and_wav",
                        args.output_npy_fname + f".wav.{output_split_id:06}",
                    ),
                    segments_wav,
                )
                output_split_id += 1
                segment_count = 0
                del segments_emb
                del segments_wav
                segments_emb = None
                segments_wav = None
            print(
                f"DONE! Processed {row_count} rows and {segment_count} segments",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument("tar_list_fname", help="List of input tar paths")
    args = parser.add_argument("output_npy_fname", help="Output numpy file path")
    args = parser.parse_args()
    main(args)
