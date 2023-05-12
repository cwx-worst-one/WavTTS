#!/bin/bash -ex

cd $(dirname $0)/../../
echo "Work dir: $(pwd)"

if [ $SPARSE_GPT == TRUE ]; then
  echo "Install modified triton"
  pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
  sudo cp recipes/audio_lm/scripts/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/
fi

if [ ! -d inference_test ]
then
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/inference_test .
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/.module_cache .
fi

bash launch.sh $@
