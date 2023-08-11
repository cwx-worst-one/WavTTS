#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/musiclm/requirements.txt
pip3 install ./recipes/soundstream/torch-museval

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/mulan/requirements.txt
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
pip3 install -i https://bytedpypi.byted.org/simple http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.1260.tar.gz
pip3 install --no-deps fsspec==2023.6.0

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# Download mae ckpt
hdfs dfs -get "/home/byte_speech_sv/weitsung.lu/mut_mae/MuT_MAE/mut_large/mutmae-step=177600-loss_1=5-sf.pth"
hdfs dfs -get "/home/byte_speech_sv/xuchen.song/mae/mutmae-step=046400-loss_0=10-kaggle.pth"


# Download gpt3 expansion
hdfs_base="hdfs://harunava/home/byte_speech_sv/mulan"
gpt3_expansion="assets/ecals_gpt3_expansion.pkl"
g4_aed="assets/chatgpt_g4_aed.pkl"
g4_mcc="assets/chatgpt_g4_mcc.pkl"

if [ ! -d "assets/" ]
then
    echo "Download ecals gpt3 expansion"
    mkdir -p assets
    # hdfs dfs -get $hdfs_base/$gpt3_expansion $gpt3_expansion
    # hdfs dfs -get $hdfs_base/$g4_aed $g4_aed
    # hdfs dfs -get $hdfs_base/$g4_mcc $g4_mcc
    hdfs dfs -get $hdfs_base/assets
else
    echo "Ecals gpt3 expansion exists, skip download"
fi

# Download validation set
if [ ! -d "data/" ] 
then
    echo "Download kaggle validation set"
    mkdir -p data/kaggle
    hdfs dfs -get $hdfs_base/validation/kaggle_val_mul.tar.gz data/kaggle
    echo "Unzip validation set"
    tar -xzf data/kaggle/kaggle_val_mul.tar.gz -C data/kaggle
    rm data/kaggle/kaggle_val_mul.tar.gz

    echo "Download QQ validation set"
    mkdir -p data/qq
    hdfs dfs -get $hdfs_base/validation/qq_val.tar.gz data/qq
    echo "Unzip validation set"
    tar -xzf data/qq/qq_val.tar.gz -C data/qq
    rm data/qq/qq_val.tar.gz
else
    echo "Validation set exists, skip download"
fi

sh launch.sh $@