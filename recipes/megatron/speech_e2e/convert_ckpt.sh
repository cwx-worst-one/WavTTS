#!/bin/bash

CURR_PATH=$(cd $(dirname $0); pwd)
mkdir -p $CURR_PATH/ckpt

# Download sample checkpoint
hdfs dfs -get /home/byte_speech_sv/jingsong.gao/sami_ai_llm/ckpt/6b4 $CURR_PATH/ckpt/

python3 $CURR_PATH/../tools/checkpoint/convert_gpt2.py \
    $CURR_PATH/ckpt/6b4 \
    $CURR_PATH/ckpt/6b4_hf
