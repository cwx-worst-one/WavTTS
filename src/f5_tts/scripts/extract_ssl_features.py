"""
Script for extracting SSL features (e.g., Hubert) using PyTorch DDP.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoFeatureExtractor, AutoModel


#################################################################################
#                         Helper Functions                                      #
#################################################################################


def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # Real logger for the main process
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m][Rank {}] %(message)s".format(rank),
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f"{logging_dir}/log_rank{rank}.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # Other processes can also log to their own files or be simpler
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m][Rank {}] %(message)s".format(rank),
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(f"{logging_dir}/log_rank{rank}.txt")],  # Log to file only
        )
        logger = logging.getLogger(__name__)
    return logger


# Custom Dataset for audio file paths
class AudioPathDataset(Dataset):
    def __init__(self, audio_root_dir, file_extension=".flac"):  # Added file_extension
        self.audio_root_dir = audio_root_dir
        self.file_lines = list(
            Path(self.audio_root_dir).rglob(f"*{file_extension}")
        )  # Use glob to find all files with the given extension
        self.file_extension = file_extension

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, idx):
        file_path = self.file_lines[idx]
        relative_path = file_path.relative_to(self.audio_root_dir)  # xxx/xxx/xxx.flac
        return str(file_path), str(relative_path)


#################################################################################
#                         Feature Extraction Logic                              #
#################################################################################


def main(args):
    """
    Extracts SSL features.
    """
    assert torch.cuda.is_available(), "Feature extraction currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    # world_size is useful for DistributedSampler, batch size calculations etc.
    world_size = dist.get_world_size()
    # Batch size per GPU. For feature extraction, often 1.
    # args.global_batch_size would be the total batch across all GPUs.
    # Here, we assume DataLoader batch_size is per-GPU.
    # assert args.global_batch_size % world_size == 0, f"Global batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an output directory (optional, for logs primarily if args.output_dir handles features)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # Central logging directory
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    dist.barrier()  # Ensure directory is created before other ranks proceed

    logger = create_logger(os.path.join(args.output_dir, "logs"), rank)
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}, device={device}.")
    logger.info(f"Script arguments: {args}")

    # Load SSL model and feature extractor:
    logger.info(f"Loading SSL model from {args.ssl_model_path}...")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.ssl_model_path, trust_remote_code=args.trust_remote_code
    )
    ssl_model = (
        AutoModel.from_pretrained(args.ssl_model_path, trust_remote_code=args.trust_remote_code).eval().to(device)
    )

    logger.info("SSL model and feature extractor loaded successfully.")

    # Setup data:
    dataset = AudioPathDataset(args.audio_root_dir, args.file_extension)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # Typically False for feature extraction unless order doesn't matter
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # Batch size per GPU
        shuffle=False,  # Shuffle is handled by sampler
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,  # Process all files
    )

    logger.info(f"Rank {rank}: Dataset size: {len(dataset)}, Sampler will process: {len(sampler)} files.")

    processed_count = 0
    total_files_on_rank = len(sampler)

    start_time = time()

    for batch_idx, (full_audio_paths, relative_path_stems) in enumerate(loader):
        # Assuming batch_size_per_gpu is 1 for simplicity in feature extraction context
        # If batch_size_per_gpu > 1, you'll need to loop through items in the batch
        for i in range(len(full_audio_paths)):
            full_audio_path = full_audio_paths[i]
            relative_path_stem = relative_path_stems[i]

            try:
                wav, sr = torchaudio.load(full_audio_path)
                if sr != args.sample_rate:
                    # logger.debug(f"Resampling {full_audio_path} from {sr} Hz to {args.sample_rate} Hz")
                    wav = torchaudio.functional.resample(wav, sr, args.sample_rate)

                # Ensure wav is mono and suitable for the feature_extractor
                if wav.shape[0] > 1:  # If stereo, take the first channel
                    wav = wav[0, :].unsqueeze(0)
                if wav.ndim == 1:  # Ensure it's (1, num_samples)
                    wav = wav.unsqueeze(0)

                input_values = feature_extractor(
                    wav.squeeze(0).numpy(),  # Squeeze to 1D array, convert to numpy as some extractors prefer
                    sampling_rate=args.sample_rate,
                    return_tensors="pt",
                ).input_values.to(device)

                with torch.no_grad():
                    ssl_output = ssl_model(input_values, output_hidden_states=True)

                # Choose the desired representation (e.g., last_hidden_state or a specific layer)
                # For Hubert, last_hidden_state is common.
                # If you need a specific layer, access ssl_output.hidden_states[layer_index]
                if args.layer_index == -1:  # last_hidden_state
                    representation = ssl_output.last_hidden_state
                elif args.layer_index >= 0 and args.layer_index < len(ssl_output.hidden_states):
                    representation = ssl_output.hidden_states[args.layer_index]
                else:
                    logger.error(
                        f"Invalid layer_index: {args.layer_index}. Available layers: 0 to {len(ssl_output.hidden_states) - 1}. Using last_hidden_state."
                    )
                    representation = ssl_output.last_hidden_state

                representation = representation.squeeze(0)  # (1, Seq, Dim) -> (Seq, Dim)
                assert len(representation.shape) == 2, (
                    f"Expected 2D tensor, got {representation.shape} for {full_audio_path}"
                )

                # Define output path for the feature
                # e.g., output_dir/speaker_id/chapter_id/filename_stem.npy
                output_feat_path = Path(args.output_dir) / relative_path_stem
                output_feat_path = output_feat_path.with_suffix(".npy")
                if not output_feat_path.parent.exists():
                    output_feat_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(output_feat_path, representation.detach().cpu().numpy())
                # logger.debug(f"Saved features for {relative_path_stem} to {output_feat_path}")

                processed_count += 1
                if processed_count % args.log_every == 0:
                    elapsed_time = time() - start_time
                    eta = (
                        (elapsed_time / processed_count) * (total_files_on_rank - processed_count)
                        if processed_count > 0
                        else 0
                    )
                    logger.info(
                        f"Processed {processed_count}/{total_files_on_rank} files. "
                        f"Last: {relative_path_stem}. ETA: {eta:.2f}s"
                    )

            except Exception as e:
                logger.error(f"Error processing {full_audio_path}: {e}")
                # Optionally, save a list of failed files
                with open(os.path.join(args.output_dir, "logs", f"failed_files_rank{rank}.txt"), "a") as f_err:
                    f_err.write(f"{full_audio_path}\t{e}\n")
                continue  # Skip to the next file

    logger.info(f"Rank {rank} finished processing {processed_count} files.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Feature Extractor using DDP")
    parser.add_argument(
        "--audio-root-dir",
        type=str,
        required=True,
        help="Root directory where audio files listed in data-list-path are located.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="extracted_features", help="Directory to save extracted features and logs."
    )
    parser.add_argument(
        "--ssl-model-path",
        type=str,
        default="facebook/hubert-large-ll60k",
        help="Path or HuggingFace identifier for the SSL model (e.g., Hubert, Wav2Vec2).",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Which layer's hidden state to extract. -1 for last_hidden_state. "
        "0 for the first layer of hidden_states, etc.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for audio processing.")
    parser.add_argument(
        "--file-extension", type=str, default=".flac", help="Audio file extension (e.g., .wav, .flac, .mp3)."
    )

    # DDP / Performance related arguments
    parser.add_argument("--global-seed", type=int, default=42, help="Global seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers per GPU.")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N files.")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow trusting remote code for HuggingFace models."
    )

    args = parser.parse_args()
    main(args)