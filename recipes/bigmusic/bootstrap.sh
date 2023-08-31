#!/bin/bash -ex

cd $(dirname $0)/../../

sudo apt update
sudo apt install espeak -y
sudo apt install ffmpeg -y
pip3 install -qr ./recipes/bigmusic/requirements.txt


if [ -d "/mnt/bn/audio-diffusion/.module_cache" ]; then
    echo "Found existing cache. Setting huggingface cache to /mnt/bn/audio-diffusion/.module_cache"
    export TRANSFORMERS_CACHE=/mnt/bn/audio-diffusion/.module_cache
else
    echo "Warning: Could not find existing huggingface cache. Set TRANSFORMERS_CACHE=/cache/path to avoid download errors."
fi

sh launch.sh $@