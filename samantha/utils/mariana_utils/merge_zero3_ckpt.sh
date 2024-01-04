#!/bin/bash

checkpoint_dir=${1}
dtype=${2:-fp16}

python3 merge_zero3_ckpt.py --dtype=$dtype --checkpoint_dir=$checkpoint_dir