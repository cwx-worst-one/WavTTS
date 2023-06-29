#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"



# Download mae ckpt
hdfs dfs -get "/home/byte_speech_sv/weitsung.lu/mut_mae/MuT_MAE/mut_large/mutmae-step=177600-loss_1=5-sf.pth"

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

