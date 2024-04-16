#!/bin/bash -ex

cd $(dirname $0)/../../

# set Arnold region when on Merlin instance
# it is done automatically on Arnold
if [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace-us.byted.org" ]; then
	echo "Setting ARNOLD_REGION=US"
	export ARNOLD_REGION="US"
elif [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace.byted.org" ]; then
	echo "Setting ARNOLD_REGION=CN"
	export ARNOLD_REGION="CN"
fi

# setting hdfs envs
export LD_LIBRARY_PATH=/opt/tiger/native_libhdfs/lib/native:$LD_LIBRARY_PATH
export ARNOLD_HDFS_NATIVE=1
export ARNOLD_HDFS_CELER=1
export INFSEC_HADOOP_ENABLED=1
export CPP_HDFS_CONF=/opt/tiger/arnold/hdfs_client/conf/celer_us/core-site.xml:/opt/tiger/arnold/hdfs_client/conf/celer_us/hdfs-site.xml

# Fix for broken mirrors "E: The repository 'http://mirrors.byted.org/debian bullseye-updates Release' no longer has a Release file."
if ! sudo apt update ; then
    echo "Updating apt failed. Removing broken mirrors"
    sudo sed -i "s/^deb http:\/\/mirrors.byted.org\/debian bullseye-updates main contrib non-free/#deb http:\/\/mirrors.byted.org\/debian bullseye-updates main contrib non-free/" /etc/apt/sources.list
    sudo sed -i "s/^deb-src http:\/\/mirrors.byted.org\/debian bullseye-updates main contrib non-free/#deb-src http:\/\/mirrors.byted.org\/debian bullseye-updates main contrib non-free/" /etc/apt/sources.list
fi

# For ASR wer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH="${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel"
# For sami_tts_api
export LD_LIBRARY_PATH=/opt/tiger/sami_engine_cleaned/libs:$LD_LIBRARY_PATH 

# Instructions on how to prepare library - https://bytedance.sg.larkoffice.com/docx/LRBCdwjcboV9eZxETqGlN7w9gMc
PYPETREL_VERSION="1.0.0.129-dev2"
if ! grep -q "version:$PYPETREL_VERSION" /opt/tiger/pypetrel/current_revision; then
    cd /opt/tiger
    rm -rf /opt/tiger/pypetrel
    echo "Downloading ASR model: pypetrel"
    hdfs dfs -get /home/byte_speech_sv/andrew.shaw/data/asr/pypetrel.$PYPETREL_VERSION.tar .
    tar -xvf pypetrel.$PYPETREL_VERSION.tar
    cd -
fi

if [ -d "/opt/tiger/sami_tts_api" ]; then
    echo "Downloading sami tts frontend engine"
    cd /opt/tiger
    hdfs dfs -get /home/byte_speech_sv/andrew.shaw/data/sami/sami_tts_api.tar .
    tar -xvf sami_tts_api.tar
    hdfs dfs -get /home/byte_speech_sv/andrew.shaw/data/sami/sami_engine_cleaned.tar .
    tar -xvf sami_engine_cleaned.tar
    cd -
fi

# For huggingface blocking our IP
if [ -d "/mnt/bn/audio-diffusion/.module_cache" ]; then
    echo "Found existing cache. Setting huggingface cache to /mnt/bn/audio-diffusion/.module_cache"
    export TRANSFORMERS_CACHE=/mnt/bn/audio-diffusion/.module_cache
    export HUGGINGFACE_HUB_CACHE=/mnt/bn/audio-diffusion/.module_cache
    # export TRANSFORMERS_OFFLINE=1 # Enable fully offline mode if you are running into huggingface errors on AWS
elif [ "$ARNOLD_REGION" == "CN" ]; then
    export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
    export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
else
    echo "Warning: Could not find existing huggingface cache. Set TRANSFORMERS_CACHE=/cache/path to avoid download errors."
fi
# To fix huggingface dataloading segmentation fault for phonemizer
TOKENIZERS_PARALLELISM=false

sudo apt update
sudo apt install espeak ffmpeg zip fonts-arphic-ukai -y
pip3 install -qr ./recipes/bigmusic/requirements.txt
pip3 install -qr ./recipes/diffusion/requirements.txt
# Upgrade easycycle to latest version
pip3 install -U bytedance.easycycle

# pip3 install recipes/soundstream/torch-museval # diffusion inference no longer depends on torch-museval
MAX_ORDER=10 pip3 install https://github.com/kpu/kenlm/archive/master.zip

sh launch.sh $@
