#!/bin/bash -ex

cd $(dirname $0)
WORK_DIR=$(pwd)
echo "work dir: $WORK_DIR"

export SAMANTHA_DISABLE_LOGGING_BASIC_CONFIG=1
export NCCL_DEBUG=WARN

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/opt/tiger/Megatron-LM/:$WORK_DIR/mariana:$PYTHONPATH
export MARIANA_SKIP_MASON_INSTALL=1

# adapt script file path in mariana/launch.sh
cp apps/mariana/setup_cruise.sh ./setup_cruise.sh
bash apps/mariana/launch.sh "$@"