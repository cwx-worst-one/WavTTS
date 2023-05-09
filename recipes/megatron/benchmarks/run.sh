#!/bin/bash

set -e
export BYTED_TORCH_FX=O0
export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

# HyperParameters
: ${MICRO_BATCH_SIZE:=1}
: ${GLOBAL_BATCH_SIZE:=128}

: ${TP_SIZE:=2}
: ${PP_SIZE:=2}
: ${SEQ_PARALLEL:=1}

: ${DIST_OPTIMIZER:=1}
: ${FLASH_ATTN:=1}
: ${ACTIVATION_CKPT:=1}

: ${NLAYERS:=30}
: ${NHIDDEN:=4096}
: ${NHEADS:=32}
: ${SEQ_LEN:=2048}

: ${EXP_NAME:="7b1-openwebtext"}

TENSORBOARD_DIR=hdfs://harunava/home/byte_speech_sv/jingsong.gao/sami_ai_llm/logs/$EXP_NAME
CURR_PATH=$(cd $(dirname $0); pwd)

CHECKPOINT_PATH=ckpt
TOKENIZER_PATH="bigscience/bloom"
DATA_PATH="$CURR_PATH/data/openwebtext_text_document"

DISTRIBUTED_ARGS="
    --nproc_per_node ${ARNOLD_WORKER_GPU} \
    --nnodes ${ARNOLD_WORKER_NUM} \
    --node_rank ${ARNOLD_ID} \
    --master_addr ${METIS_WORKER_0_HOST} \
    --master_port ${METIS_WORKER_0_PORT}
"

PARALLEL_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
"

if [ $SEQ_PARALLEL -eq 1 ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS --sequence-parallel"
fi

if [ $DIST_OPTIMIZER -eq 1 ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS --use-distributed-optimizer"
fi

if [ $ACTIVATION_CKPT -eq 1 ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS --recompute-activations"
fi

if [ $FLASH_ATTN -eq 1 ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS --use-flash-attn"
fi

GPT_ARGS="
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 1.0e-5 \
    --initial-loss-scale 8192 \
    --train-iters 1000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model $TOKENIZER_PATH \
    --pad-vocab-size-to 250880 \
    --data-impl mmap \
    --split 950,49,1
"

OUTPUT_ARGS="
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

CMD="torchrun $DISTRIBUTED_ARGS $CURR_PATH/pretrain_gpt.py \
    $PARALLEL_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"

echo $CMD
$CMD
