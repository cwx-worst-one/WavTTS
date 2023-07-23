import argparse
import sys
from typing import List

import torch

import recipes
import recipes.text2semantic
import recipes.text2semantic.modules
import recipes.text2semantic.lit_modules
import recipes.text2semantic.modules.continuous_vae_llama_ctiga
import recipes.text2semantic.modules.loss
import recipes.text2semantic.modules.valle_nar

def convert(legacy_ckpt_paths: List[str], new_ckpt_paths: List[str]):
    r"""Convert legacy checkpoints to new due to some BC changes.

    Args:
        legacy_ckpt_paths (List[str]): path list of legacy ckpts
        new_ckpt_paths (List[str]): path list of new ckpts

    """

    if len(legacy_ckpt_paths) != len(new_ckpt_paths):
        raise ValueError("length of legacy and new paths must be same.")
        
    sys.modules["recipes"] = recipes
    sys.modules["recipes.valle"] = recipes.text2semantic
    sys.modules["recipes.valle.lit_modules"] = recipes.text2semantic.lit_modules
    sys.modules["recipes.valle.modules"] = recipes.text2semantic.modules
    sys.modules["recipes.valle.modules.continuous_vae_llama_ctiga"] = recipes.text2semantic.modules.continuous_vae_llama_ctiga
    sys.modules["recipes.valle.modules.loss"] = recipes.text2semantic.modules.loss
    sys.modules["recipes.valle.modules.valle_nar"] = recipes.text2semantic.modules.valle_nar
    
    for legacy_ckpt, new_ckpt in zip(legacy_ckpt_paths, new_ckpt_paths):
        ckpt = torch.load(legacy_ckpt, map_location="cpu")
        torch.save(ckpt, new_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-ckpt-paths", type=str, nargs="+", help="path to legacy checkpoints"
    )
    parser.add_argument(
        "--new-ckpt-paths", type=str, nargs="+", help="path to new checkpoints"
    )
    args = parser.parse_args()

    convert(args.legacy_ckpt_paths, args.new_ckpt_paths)
