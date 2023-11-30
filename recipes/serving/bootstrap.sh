#!/usr/bin/env bash

#export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
#export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"

sudo apt-get install -y espeak

pip3 install --upgrade pip
pip3 install -r recipes/serving/requirements.txt
pip3 install -U bytedance.easycycle -i https://bytedpypi.byted.org/simple

export PYTHONPATH=$PWD:$PWD/recipes/soundstream/torch-museval

if [ ! -d /opt/tiger/pypetrel ]; then
  cd /opt/tiger
  /opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get /home/byte_speech_sv/andrew.shaw/data/asr/pypetrel.1.0.0.129-dev.tar .
  tar -xvf pypetrel.1.0.0.129-dev.tar
  cd -
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH=${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel

# For huggingface blocking our IP
if [ -d "/mnt/bn/audio-diffusion/.module_cache" ]; then
    echo "Found existing cache. Setting huggingface cache to /mnt/bn/audio-diffusion/.module_cache"
    export TRANSFORMERS_CACHE=/mnt/bn/audio-diffusion/.module_cache
else
    echo "Warning: Could not find existing huggingface cache. Set TRANSFORMERS_CACHE=/cache/path to avoid download errors."
fi


# start the euler server
worker_num=${SERVER_WORKER_NUM:-1}
thread_num=${SERVER_THREAD_NUM:-1}
python3 recipes/serving/server.py ${worker_num} ${thread_num}
