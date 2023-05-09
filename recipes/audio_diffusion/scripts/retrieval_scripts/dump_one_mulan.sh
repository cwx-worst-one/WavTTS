#!/bin/bash
set -ex

if [ $# != 3 ]; then
    echo "Usage: $0 <data_index> <data_prefix> <output_prefix>"
    exit 0
fi
data_index=$1
data_prefix=$2
output_prefix=$3

DATA_DIR=`dirname $data_prefix`
mkdir -p $DATA_DIR/logs
python3 dump_mulan_emb.py \
    ${data_prefix}.${data_index} \
    ${output_prefix}.${data_index} \
    >$DATA_DIR/logs/dump.${data_index}.log 2>&1
