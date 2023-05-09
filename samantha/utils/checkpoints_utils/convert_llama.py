"""This script converts the original PyTorch LLaMA model checkpoints published by
Meta into our own source code implementation of the model architecture. The LLaMA model
comes in fivedifferent sizes: 7, 13, 30 and 65 billion parameters, in addition to the
SentencePiece tokenizer that was initialised on their corpus.

The original LLaMA source code is published under a restrictive GPL license, hence it had to be
reimplemented so that further (commercially motivated) research can be conducted. 

@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},  # noqa
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
"""

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

shard_dims = {
    "lm_head.weight": 0,
    "wte.weight": 1,
    "attn.to_q.weight": 0,
    "attn.to_k.weight": 0,
    "attn.to_v.weight": 0,
    "attn.WO.weight": 1,
    "mlp.c_fc1.weight": 0,
    "mlp.c_fc2.weight": 0,
    "mlp.c_proj.weight": 1,
}

NUM_SHARDS = {"7B": 1, "13B": 2, "30B": 4, "65B": 8}


def calc_rotary_inv_freq(n_embd: int, n_head: int) -> float:
    # n_heads_per_shard = n_heads // params["n_shards"]
    dims_per_head = n_embd // n_head
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
    )
    return inv_freq


def convert_llama_state_dict(
    state_dict: Dict[str, torch.Tensor],
    inv_freq: float,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    converted = {}
    converted["transformer.wte.weight"] = state_dict["tok_embeddings.weight"].to(dtype)
    converted["lm_head.weight"] = state_dict["output.weight"].to(dtype)
    converted["transformer.ln_f.weight"] = state_dict["norm.weight"].to(dtype)

    layer_idxs = [k.split(".")[1] for k in state_dict if k.startswith("layers")]
    for layer_idx in tqdm(layer_idxs, desc="Transfering weights"):
        # attention
        converted[f"transformer.h.{layer_idx}.attn.to_q.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wq.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.attn.to_k.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wk.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.attn.to_v.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wv.weight"
        ].to(dtype)

        converted[f"transformer.h.{layer_idx}.attn.WO.weight"] = state_dict[
            f"layers.{layer_idx}.attention.wo.weight"
        ].to(dtype)

        # mlp
        converted[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w1.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w2.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"] = state_dict[
            f"layers.{layer_idx}.feed_forward.w3.weight"
        ].to(dtype)

        # rms norm
        converted[f"transformer.h.{layer_idx}.ln_1.weight"] = state_dict[
            f"layers.{layer_idx}.attention_norm.weight"
        ].to(dtype)
        converted[f"transformer.h.{layer_idx}.ln_2.weight"] = state_dict[
            f"layers.{layer_idx}.ffn_norm.weight"
        ].to(dtype)

        converted[
            f"transformer.h.{layer_idx}.attn.rotary_embeddings.inv_freq"
        ] = inv_freq
    return converted


def convert_meta_llama_weights(
    ckpt_dir: str, output_dir: str, model_size: str = "7B", dtype: str = torch.float32
) -> None:  # pragma: no cover
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)

    tokenizer_path = ckpt_dir / "tokenizer.model"

    ckpt_dir = ckpt_dir / model_size
    output_dir = output_dir / model_size
    os.makedirs(output_dir, exist_ok=True)

    params_fp = ckpt_dir / "params.json"
    with open(params_fp) as f:
        params = json.load(f)

    n_embd = params["dim"]
    n_head = params["n_heads"]
    # n_shards = NUM_SHARDS[model_size]
    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    if "tokenizer.model" not in os.listdir(output_dir.parent):
        shutil.copy(tokenizer_path, output_dir.parent)

    checkpoint_files = sorted(ckpt_dir.glob("*.pth"))
    checkpoint_files.sort()
    n_checkpoints = len(checkpoint_files)

    if n_checkpoints == 0:
        raise RuntimeError(f"No checkpoints were found at ckpt_dir {ckpt_dir}")

    # for the bigger models, there are multiple model-parallel checkpoints
    # and we combine them into one single file
    combined = None
    for file in tqdm(checkpoint_files, total=n_checkpoints):
        checkpoint = torch.load(file, map_location="cpu")

        inv_freq = calc_rotary_inv_freq(n_embd=n_embd, n_head=n_head)
        converted = convert_llama_state_dict(checkpoint, inv_freq, dtype=dtype)

        if combined is None:
            combined = converted
            continue
        for name, param in converted.items():
            dim = None
            for k, d in shard_dims.items():
                if k in name:
                    dim = d
                    break
            if dim is None:
                # Extra check: assert that tensors are the same if not sharded
                # assert torch.allclose(combined[name], param)
                continue
            combined[name] = torch.cat((combined[name], param), dim=dim)

        del checkpoint
        del converted
        gc.collect()
    return combined


if __name__ == "__main__":  # pragma: no cover
    output_dir = "output"
    converted_model = convert_meta_llama_weights("downloads/llama", "output")
    torch.save(converted_model, Path(output_dir, "llama.pt"))
