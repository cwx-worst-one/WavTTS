#!/bin/bash

export MUSICLM_DIR=/opt/tiger/samantha
cd $MUSICLM_DIR

sudo apt update && sudo apt install tmux htop ffmpeg -y && pip3 install ipython pydub -i https://bytedpypi.byted.org/simple/

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple && \
pip3 install wheel Cython numpy && \
pip3 install -q -r requirements.txt && \
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple && \
cp $MUSICLM_DIR/recipes/audio_lm/scripts/matmul.py /home/tiger/.local/lib/python3.8/site-packages/triton/ops/blocksparse/

if [ ! -e last.ckpt ]
then
    hdfs dfs -get $1/checkpoints/last.ckpt last.ckpt || echo "Can't get hdfs file"
fi

if [ ! -d inference_test ]
then
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/inference_test .
fi

if [ ! -e last.ckpt ]
then
    bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/semantic_v2_3ar_sparse.yaml --run_opts.hdfs_path=$1
else
    bash recipes/audio_lm/bootstrap.sh fit --config recipes/audio_lm/conf/semantic_v2_3ar_sparse.yaml --run_opts.hdfs_path=$1 --ckpt_path=./last.ckpt
fi
