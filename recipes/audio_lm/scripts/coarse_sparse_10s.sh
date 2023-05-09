#!/bin/bash

export MUSICLM_DIR=/opt/tiger/samantha
cd $MUSICLM_DIR

sudo apt update && sudo apt install tmux htop ffmpeg -y && pip3 install ipython pydub -i https://bytedpypi.byted.org/simple/

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple && \
pip3 install wheel Cython numpy && \
pip3 install -q -r requirements.txt && \
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple && \
cp $MUSICLM_DIR/recipes/audio_lm/scripts/matmul.py /home/tiger/.local/lib/python3.8/site-packages/triton/ops/blocksparse/

# fit --config recipes/audio_lm/conf/sparse_us/coarse_sparse_10s.yaml
bash recipes/audio_lm/bootstrap.sh $@