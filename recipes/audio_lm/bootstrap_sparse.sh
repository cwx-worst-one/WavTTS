#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

pip3 install torch_complex rotary_embedding_torch bytedance-easycycle -i https://bytedpypi.byted.org/simple

# Download gpt3 expansion
if [[ ! -d "assets/" &&  $ARNOLD_BASE_DIR == "hdfs://harunava" ]]; then
    hdfs dfs -get "hdfs://harunava/home/byte_speech_sv/mulan/assets"
fi

sudo cp recipes/audio_lm/scripts/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/
hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/coarse_10s_mask_v3.npy
hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/semantic_10s_mask.npy

bash launch.sh $@