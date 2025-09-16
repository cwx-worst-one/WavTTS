#!/bin/bash -ex

cd $(dirname $0)/../../

CWD=$(pwd)

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
if ! [[ ":$LD_LIBRARY_PATH:" == *":/opt/tiger/native_libhdfs/lib/native:"* ]]; then
    export LD_LIBRARY_PATH=/opt/tiger/native_libhdfs/lib/native:$LD_LIBRARY_PATH
fi
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

# local cache paths
LOCAL_CACHE_DIR="${CWD}/.cache"
echo "Creating local cache directories in: ${LOCAL_CACHE_DIR}"
mkdir -p \
    "${LOCAL_CACHE_DIR}" \
    "${LOCAL_CACHE_DIR}/torch/kernels"
export PYTORCH_KERNEL_CACHE_PATH="${LOCAL_CACHE_DIR}/torch/kernels"



## Attention library - 68 uses windowed attention
bash reinstall_s3a.sh 68;
# bash scripts/reinstall_s3a.sh 68;
export FLASHATTN_VERSION=2.3

# To fix huggingface dataloading segmentation fault for phonemizer
export TOKENIZERS_PARALLELISM=false


# torch-museval - SDR metrics
pip3 install recipes/soundstream/torch-museval
pip3 install -q -r recipes/sacodec/requirements_soundstream.txt

sudo apt update
# sudo apt install espeak -y # for phonemizer
pip3 install -qr ./recipes/sacodec/requirements.txt

sh launch.sh $@
