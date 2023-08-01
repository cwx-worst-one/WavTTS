#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

#sh .codebase/pipelines/install_dependencies.sh

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/mulan/requirements.txt
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
mkdir -p check
# Download gpt3 expansion
hdfs_base="hdfs://harunava/home/byte_speech_sv/mulan"
gpt3_expansion="assets"
if [ ! -d "assets/" ]
then
    echo "Download ecals gpt3 expansion"
    mkdir -p assets
    hdfs dfs -get $hdfs_base/$gpt3_expansion .
else
    echo "Ecals gpt3 expansion exists, skip download"
fi

# Download validation set
if [ ! -d "data/" ] 
then
    echo "Download validation set"
    mkdir -p data/sf
    mkdir -p data/kaggle
    hdfs dfs -get $hdfs_base/validation/kaggle_val.tar.gz data/kaggle
    hdfs dfs -get $hdfs_base/validation/sf_val.tar.gz data/sf
    echo "Unzip validation set"
    tar -xzf data/kaggle/kaggle_val.tar.gz -C data/kaggle
    tar -xzf data/sf/sf_val.tar.gz -C data/sf
    rm data/kaggle/kaggle_val.tar.gz
    rm data/sf/sf_val.tar.gz
else
    echo "Validation set exists, skip download"
fi

bash launch.sh $@
