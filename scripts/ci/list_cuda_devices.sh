#!/usr/bin/env bash

# Copy from bumi
# Copyright 2024 ByteDance Inc. All rights reserved.
# Author: Gao Lu (gaolu.e820@bytedance.com)
#
# List usable CUDA devices, and sort the result by usable memory capacity.
#
# For A800 machine, we only include the first 4 GPUs in the result. This is because our A800 machine
# is also bing used by inference guys, we don't want our CI to interfere with their daily jobs.

# Check if NVIDIA A800 is present
if nvidia-smi | grep -q A800; then
  # Specific command for when A800 is found. Filters to GPUs 0-3 after the listing and sorting.
  filter_command="grep -E '[0-3]'"
else
  # General command for all other situations.
  filter_command="cat" # 'cat' command acts as a pass-through.
fi

nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv |
  sed 's/\s*\(.\)iB/\1/g' |
  tail -n +2 |
  sort -R |
  sort -rt ',' -k 2 -h --stable |
  awk -F , '{print $1}' |
  eval $filter_command |
  head -n 4 |
  sort -R
