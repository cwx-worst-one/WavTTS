#!/bin/bash -ex

cd $(dirname $0)/../../

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

if ! grep -q "version:1.0.0.129" /opt/tiger/pypetrel/current_revision; then
    rm -rf /opt/tiger/pypetrel
    cp -r /mnt/bn/audio-diffusion/ashaw/bvc/pypetrel.1.0.0.129 /opt/tiger/pypetrel
fi

# For huggingface blocking our IP
if [ -d "/mnt/bn/audio-diffusion/.module_cache" ]; then
    echo "Found existing cache. Setting huggingface cache to /mnt/bn/audio-diffusion/.module_cache"
    export TRANSFORMERS_CACHE=/mnt/bn/audio-diffusion/.module_cache
else
    echo "Warning: Could not find existing huggingface cache. Set TRANSFORMERS_CACHE=/cache/path to avoid download errors."
fi

sudo apt update
sudo apt install espeak ffmpeg zip fonts-arphic-ukai -y
pip3 install -qr ./recipes/bigmusic/requirements.txt
pip3 install -qr ./recipes/diffusion/requirements.txt

# Diffusion
pip3 install ./recipes/soundstream/torch-museval


MAX_ORDER=10 pip3 install https://github.com/kpu/kenlm/archive/master.zip

sh launch.sh $@
