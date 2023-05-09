#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert zero checkpoint to FP32 checkpoint.
"""

import argparse
import os, re
import torch
from pytorch_lightning.utilities.deepspeed import (
        convert_zero_checkpoint_to_fp32_state_dict,
        get_model_state_file,
        get_optim_files,
        ds_checkpoint_dir
)

DS_PARAM_REGEX = r'_forward_module\.(.+)'

def convert_deepspeed_checkpoint(deepspeed_ckpt_path: str, pl_ckpt_path: str = None):
    '''
    Creates a PyTorch Lightning checkpoint from the DeepSpeed checkpoint directory, while patching
    in parameters which are improperly loaded by the DeepSpeed conversion utility.
    deepspeed_ckpt_path: Path to the DeepSpeed checkpoint folder.
    pl_ckpt_path: Path to the reconstructed PyTorch Lightning checkpoint. If not specified, will be
        placed in the same directory as the DeepSpeed checkpoint directory with the same name but
        a .pt extension.
    Returns: path to the converted checkpoint.
    '''
    if not (deepspeed_ckpt_path.endswith('.ckpt') and os.path.isdir(deepspeed_ckpt_path)):
        raise ValueError(
            'args.ckpt_dir should point to the checkpoint directory'
            ' output by DeepSpeed (e.g. "last.ckpt" or "epoch=4-step=39150.ckpt").'
        )

    # Convert state dict to PyTorch format
    if not pl_ckpt_path:
        pl_ckpt_path = f'{deepspeed_ckpt_path[:-4]}pt' # .ckpt --> .pt

    if not os.path.exists(pl_ckpt_path):
        convert_zero_checkpoint_to_fp32_state_dict(deepspeed_ckpt_path, pl_ckpt_path)

    # Patch in missing parameters that failed to be converted by DeepSpeed utility
    pl_ckpt = _merge_deepspeed_weights(deepspeed_ckpt_path, pl_ckpt_path)
    torch.save(pl_ckpt, pl_ckpt_path)

    return pl_ckpt_path

def _merge_deepspeed_weights(deepspeed_ckpt_path: str, fp32_ckpt_path: str):
    '''
    Merges tensors with keys in the DeepSpeed checkpoint but not in the fp32_checkpoint
    into the fp32 state dict.
    deepspeed_ckpt_path: Path to the DeepSpeed checkpoint folder.
    fp32_ckpt_path: Path to the reconstructed
    '''
    # This first part is based on pytorch_lightning.utilities.deepspeed.convert_zero_checkpoint_to_fp32_state_dict
    checkpoint_dir = ds_checkpoint_dir(deepspeed_ckpt_path)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location='cpu')
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    deepspeed_model_file = get_model_state_file(checkpoint_dir, zero_stage)

    # Start adding all parameters from DeepSpeed ckpt to generated PyTorch Lightning ckpt
    ds_ckpt = torch.load(deepspeed_model_file, map_location='cpu')
    ds_sd = ds_ckpt['module']

    fp32_ckpt = torch.load(fp32_ckpt_path, map_location='cpu')
    fp32_sd = fp32_ckpt['state_dict']

    for k, v in ds_sd.items():
        try:
            match = re.match(DS_PARAM_REGEX, k)
            param_name = match.group(1)
        except:
            print(f'Failed to extract parameter from DeepSpeed key {k}')
            continue

        v = v.to(torch.float32)
        if param_name not in fp32_sd:
            print(f'Adding parameter {param_name} from DeepSpeed state_dict to fp32_sd')
            fp32_sd[param_name] = v
        else:
            ### if the conversion fails, increase the atol value to allow larger difference between
            ### deepspeed weights and fp32 weights
            assert torch.allclose(v, fp32_sd[param_name], atol=1e-2)

    return fp32_ckpt

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_ckpt", type=str, required=True,
                        help="The path to the DeepSpeed ZERO checkpoint.")
    parser.add_argument("--fp32_ckpt", type=str, required=True,
                        help="The path to the FP32 checkpoint.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = setup_args()
    convert_deepspeed_checkpoint(args.zero_ckpt, args.fp32_ckpt)