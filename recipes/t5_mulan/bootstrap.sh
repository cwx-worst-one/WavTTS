#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/mulan/requirements.txt
pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
pip3 install peft bitsandbytes
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# Download mae ckpt
hdfs dfs -get "/home/byte_speech_sv/weitsung.lu/mut_mae/MuT_MAE/mut_large/mutmae-step=177600-loss_1=5-sf.pth"

bash launch.sh $@
