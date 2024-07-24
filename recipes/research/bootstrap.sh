#!/bin/bash
CWD=$(pwd)

echo "Adding environment variables..."
if [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace-us.byted.org" ]; then
	echo "Setting ARNOLD_REGION=US"
	export ARNOLD_REGION="US"
elif [ "$ARNOLD_WORKSPACE_SERVER" == "https://workspace.byted.org" ]; then
	echo "Setting ARNOLD_REGION=CN"
	export ARNOLD_REGION="CN"
fi

if [ "$ARNOLD_REGION" == "CN" ]; then
	echo "Adding network proxy"
  export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 http_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy="byted.org"
fi


echo "Installing phonemizer..."
sudo apt-get install espeak-ng -y

echo "Installing Python packages..."
# transformers==4.39.1 \
pip3 install -q \
    av==11.0.0 \
    fsspec==2023.6.0 \
    pandas==2.1.1 \
    pyloudnorm==0.1.1 \
    phonemizer==3.2.1 \
    transformers==4.30.2 \
    more_itertools==10.1.0 \
    descript-audiotools \
    openai-whisper==20231117

# echo "Installing Flash Attention 2..."
# pip3 install -q flash-attn --no-build-isolation

echo "Overwriting protobuf..."
pip3 install protobuf==3.20

echo "Installing K-diffusion..."
pip3 install k-diffusion

echo "Installing byteformers..."
pip3 install -q ./recipes/research/byteformers/

echo "Installing Wandb patch..."
pip3 uninstall -q -y wandb
pip3 install -q byted-wandb --upgrade

echo "Updating pyarrow..."  # TODO: not sure if this is necessary
pip3 install pyarrow -q --upgrade

echo "Updating s3a..."
bash scripts/reinstall_s3a.sh 73

# export TRANSFORMERS_CACHE="/mnt/bn/janne-research-xl/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "WARNING, USING `TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}`"

# `save` is really just here to download required models for multi-node training...
# bash launch.sh save $@

export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=1000000000 

bash launch.sh $@
