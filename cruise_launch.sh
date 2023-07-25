#!/bin/bash -ex

cd $(dirname $0)
WORK_DIR=$(pwd)
echo "work dir: $WORK_DIR"

export SAMANTHA_DISABLE_LOGGING_BASIC_CONFIG=1
export NCCL_DEBUG=WARN

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/opt/tiger/Megatron-LM/:$WORK_DIR/mariana:$PYTHONPATH

TORCHRUN "$@"