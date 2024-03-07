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


mkdir -p /opt/tiger/samantha/models/semantic
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/semantic/step=000400.ckpt /opt/tiger/samantha/models/semantic/step=000400.ckpt
mkdir -p /opt/tiger/samantha/models/mulan
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/mulan/mulan-step=005000-median_rank_1=72-kaggle.ckpt /opt/tiger/samantha/models/mulan/mulan-step=005000-median_rank_1=72-kaggle.ckpt
mkdir -p /opt/tiger/samantha/models/diffusion
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/diffusion/diffusion-step=250000-EMA.ckpt /opt/tiger/samantha/models/diffusion/diffusion-step=250000-EMA.ckpt
mkdir -p /opt/tiger/samantha/models/vocoder
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/vocoder/last-EMA.ckpt /opt/tiger/samantha/models/vocoder/last-EMA.ckpt
mkdir -p /opt/tiger/samantha/models/beat
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/beat/epoch=13-step=1400.pt /opt/tiger/samantha/models/beat/epoch=13-step=1400.pt
mkdir -p /opt/tiger/samantha/models/musicfm
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/musicfm/playlist_classic_stats.json /opt/tiger/samantha/models/musicfm/playlist_classic_stats.json
/opt/tiger/yarn_deploy/hadoop/bin/hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/musicfm/musicfm_25hz_playlist_330m_520k.pt /opt/tiger/samantha/models/musicfm/musicfm_25hz_playlist_330m_520k.pt


# start the euler server
worker_num=${SERVER_WORKER_NUM:-1}
thread_num=${SERVER_THREAD_NUM:-1}
python3 recipes/serving/server.py ${worker_num} ${thread_num}
