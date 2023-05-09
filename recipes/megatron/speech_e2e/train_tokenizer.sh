#!/bin/bash

CURR_PATH=$(cd $(dirname $0); pwd)
INPUT_PATH="$CURR_PATH/data/dev-clean_mhubert-km1000.jsonl"
OUTPUT_PATH="$CURR_PATH/tokenizer"

python3 $CURR_PATH/tokenizer/train.py \
    $INPUT_PATH \
    $OUTPUT_PATH \
    --vocab_size 2000 \
