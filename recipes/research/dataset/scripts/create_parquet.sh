#!/bin/bash

devices=3  # 32 workers per node = 96 cpu cores
master_addr=${master_addr:=$ARNOLD_WORKER_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}

torchrun \
    --node_rank=$ARNOLD_ID \
    --nproc_per_node=$devices \
    --nnodes=$ARNOLD_WORKER_NUM \
    --rdzv_endpoint=${master_addr}:${master_port} \
    recipes/research/dataset/scripts/create_parquet.py 