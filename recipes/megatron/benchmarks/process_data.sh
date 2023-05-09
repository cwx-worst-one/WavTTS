#!/bin/bash

CURR_PATH=$(cd $(dirname $0); pwd)
TOKENIZER_PATH="bigscience/bloom"
INPUT_PATH="$CURR_PATH/data/openwebtext.jsonl"
OUTPUT_PREFIX="$CURR_PATH/data/openwebtext"

# Download dataset
hdfs dfs -get /home/byte_speech_sv/jingsong.gao/sami_ai_llm/data/openwebtext.jsonl $INPUT_PATH

python3 $CURR_PATH/../tools/dataset/preprocess_data.py \
    --input $INPUT_PATH \
    --output-prefix $OUTPUT_PREFIX \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model $TOKENIZER_PATH \
    --pad-vocab-size-to 250880 \
    --workers 40 \
    --chunk-size 10000
