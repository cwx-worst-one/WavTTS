#!/bin/bash

CURR_PATH=$(cd $(dirname $0); pwd)
TOKENIZER_PATH="$CURR_PATH/tokenizer/"
INPUT_PATH="$CURR_PATH/data/dev-clean_mhubert-km1000.jsonl"
OUTPUT_PREFIX="$CURR_PATH/data/dev-clean_mhubert-km1000"

python3 $CURR_PATH/../tools/dataset/preprocess_data.py \
    --input $INPUT_PATH \
    --output-prefix $OUTPUT_PREFIX \
    --json-keys inputs targets \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model $TOKENIZER_PATH \
    --workers 40 \
    --chunk-size 10000
