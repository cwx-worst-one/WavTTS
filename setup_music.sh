#!/bin/bash
set -x  # for better debug view

CUR_DIR=$(cd $(dirname $0); pwd)
cd $CUR_DIR

# set Arnold region when on Merlin instance
# it is done automatically on Arnold
if [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace-us.byted.org" ]; then
	echo "Setting ARNOLD_REGION=US"
	export ARNOLD_REGION="US"
elif [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace.byted.org" ]; then
	echo "Setting ARNOLD_REGION=CN"
	export ARNOLD_REGION="CN"
fi

# For ASR wer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH="${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel"
# For sami_tts_api
export LD_LIBRARY_PATH=/opt/tiger/sami_engine_cleaned/libs:$LD_LIBRARY_PATH 

if ! grep -q "version:1.0.0.129-dev" /opt/tiger/pypetrel/current_revision; then
    cd /opt/tiger
    rm -rf /opt/tiger/pypetrel
    if [ -d "/mnt/bn/audio-diffusion/ashaw/bvc/pypetrel.1.0.0.129-dev" ]; then
        echo "Copying ASR model pypetrel from audio-diffusion bytenas"
        cp -r /mnt/bn/audio-diffusion/ashaw/bvc/pypetrel.1.0.0.129-dev /opt/tiger/pypetrel
    else
        echo "Downloading ASR model: pypetrel"
        hdfs dfs -get /home/byte_speech_sv/andrew.shaw/data/asr/pypetrel.1.0.0.129-dev.tar .
        tar -xvf pypetrel.1.0.0.129-dev.tar
    fi
    cd -
fi

# For huggingface blocking our IP
if [ -d "/mnt/bn/audio-diffusion/.module_cache" ]; then
    echo "Found existing cache. Setting huggingface cache to /mnt/bn/audio-diffusion/.module_cache"
    export TRANSFORMERS_CACHE=/mnt/bn/audio-diffusion/.module_cache
else
    echo "Warning: Could not find existing huggingface cache. Set TRANSFORMERS_CACHE=/cache/path to avoid download errors."
fi

pip3 install recipes/soundstream/torch-museval
MAX_ORDER=10 pip3 install https://github.com/kpu/kenlm/archive/master.zip
