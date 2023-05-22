#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

pip3 install torch_complex bytedance-easycycle -i https://bytedpypi.byted.org/simple
pip3 install --no-deps rotary_embedding_torch -i https://bytedpypi.byted.org/simple

# Download gpt3 expansion
if [[ ! -d "assets/" &&  $ARNOLD_BASE_DIR == "hdfs://harunava" ]]; then
    hdfs dfs -get "hdfs://harunava/home/byte_speech_sv/mulan/assets"
fi

bash launch.sh $@