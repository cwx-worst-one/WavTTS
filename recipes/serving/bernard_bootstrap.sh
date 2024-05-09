#!/usr/bin/env bash

export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"

cd /opt/tiger/samantha

sudo apt-get install -y espeak

pip3 install --upgrade pip
pip3 install -r recipes/serving/requirements.txt
pip3 install -U bytedance.easycycle -i https://bytedpypi.byted.org/simple

export PYTHONPATH=$PWD:$PWD/recipes/soundstream/torch-museval

if [ ! -d /opt/tiger/pypetrel ]; then
  cd /opt/tiger
  /opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_speech_sv/andrew.shaw/data/asr/pypetrel.1.0.0.129-dev.tar .
  tar -xvf pypetrel.1.0.0.129-dev.tar
  cd -
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH=${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel

function copy_from_hdfs() {
    local src=$1
    local dst=$2

/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get $src $dst
}

prompt_path="hdfs://haruna/home/byte_speech_sv/bigmusic/prompts/sstk_random_30_prompts.csv"
semantic_ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v8/semantic/v8_rc1-step=000250-minimal.ckpt"
mulan_ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v8/mulan/mulan-step=005000-median_rank_1=72-kaggle-minimal.ckpt"
diffusion_ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v8/diffusion/44.1k_stereo_v2/diffusion-step=600000-EMA-minimal.ckpt"
vocoder_ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v8/vocoder/44.1k_stereo_sa_v1/last-EMA-minimal.ckpt"

base_path="/opt/tiger/samantha/models"
mkdir -p "$base_path"
prompt_local="$base_path/$(basename "$prompt_path")"
semantic_local="$base_path/$(basename "$semantic_ckpt")"
mulan_local="$base_path/$(basename "$mulan_ckpt")"
diffusion_local="$base_path/$(basename "$diffusion_ckpt")"
vocoder_local="$base_path/$(basename "$vocoder_ckpt")"

copy_from_hdfs "$prompt_path" "$prompt_local"
echo "Finish copying prompt"
copy_from_hdfs "$semantic_ckpt" "$semantic_local"
echo "Finish copying semantic ckpt"
copy_from_hdfs "$mulan_ckpt" "$mulan_local"
echo "Finish copying mulan ckpt"
copy_from_hdfs "$diffusion_ckpt" "$diffusion_local"
echo "Finish copying diffusion ckpt"
copy_from_hdfs "$vocoder_ckpt" "$vocoder_local"
echo "Finish copying vocoder ckpt"

# start the euler server
worker_num=${SERVER_WORKER_NUM:-1}
thread_num=${SERVER_THREAD_NUM:-1}
python3 recipes/serving/server.py ${worker_num} ${thread_num}
