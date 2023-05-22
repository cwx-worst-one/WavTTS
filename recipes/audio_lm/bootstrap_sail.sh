#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

pip3 install torch_complex -i https://bytedpypi.byted.org/simple
pip3 install --no-deps rotary_embedding_torch -i https://bytedpypi.byted.org/simple

sudo cp recipes/audio_lm/scripts/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/

bash launch.sh $@