#!/bin/bash -ex

cd $(dirname $0)/../../
echo "Work dir: $(pwd)"

if [ $SPARSE_GPT == TRUE ]; then
  echo "Install modified triton"
  pip3 install sentencepiece -i https://bytedpypi.byted.org/simple
  pip3 install ffmpeg-python -i https://bytedpypi.byted.org/simple
fi

if [ ! -d inference_test ]
then
  if [ $US_ARNOLD == TRUE ]
  then
    echo "Download cache from US"
    hdfs dfs -get hdfs://harunava/home/byte_speech_sv/litang/musiclm/inference_test .
    hdfs dfs -get hdfs://harunava/home/byte_speech_sv/litang/musiclm/.module_cache .
  else
    echo "Download cache from CN"
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/inference_test .
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/.module_cache .
  fi
fi

bash launch.sh $@
