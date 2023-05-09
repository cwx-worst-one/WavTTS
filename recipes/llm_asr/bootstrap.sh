#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

if [ ! -z ${TOKENIZER} ]; then
  echo "Downloading tokenizer file from hdfs"
  hdfs dfs -get $TOKENIZER ./
fi

if [ $SPARSE_GPT == TRUE ]; then
  echo "Copying matmul to triton folder!"
  sudo cp recipes/llm_asr/utils/matmul.py /usr/local/lib/python3.8/site-packages/triton/ops/blocksparse/
fi

bash launch.sh $@
