#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

# Install dependencies
pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/megatron/requirements.txt

# Set Path
export MEGATRONPATH="$(pwd)/llm"
export PYTHONPATH="${PYTHONPATH}:$MEGATRONPATH"
echo "megatron dir: $MEGATRONPATH"

# Run bash script
bash $@
