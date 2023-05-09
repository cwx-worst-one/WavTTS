#!/bin/bash 

checkpoint_dir=$1

python3 recipes/llm_asr/utils/ckpt_dsfp32-pl.py --zero_ckpt $checkpoint_dir --fp32_ckpt $checkpoint_dir/checkpoint_fp32.pth