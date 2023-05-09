#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

sh .codebase/pipelines/install_dependencies.sh

if [[ $ARNOLD_DEVICE_TYPE =~ A100 && $USE_FLASH_ATTN == TRUE ]]; then
  echo "=============== install flash attention ================="
  if [ ! -d flash-attention ]; then
    git clone "https://wangxin.colin:7GYg3s6WRkovqxZrpKE5@code.byted.org/lab-audio/flash-attention.git"
  fi
  export PYTHONPATH=$(realpath flash-attention):$PYTHONPATH
  cd flash-attention || exit 1
  sudo pip3 install . || exit 1
  cd -
fi

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/audio_lm/requirements.txt
pip3 install -q -r requirements.txt

# if [ $TRANSFORMERS_DEV == TRUE ]; then
#   echo "=============== install transformers dev ================="
#   git clone "https://zongyu.yin:R91y4u2tJkJH1RQYF6p5@code.byted.org/zongyu.yin/transformers.git"
#   cd transformers || exit 1
#   pip3 install -e . || exit 1
#   cd -
# fi

# Download gpt3 expansion
if [[ ! -d "assets/" &&  $ARNOLD_BASE_DIR == "hdfs://harunava" ]]; then
    export TOKENIZERS_PARALLELISM=false
    hdfs dfs -get "hdfs://harunava/home/byte_speech_sv/mulan/assets"
fi

# pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple

# if [ $MULAN_115 == TRUE ]; then
#   pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
#   hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/coarse_10s_mask_v4_3.npy
#   hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/semantic_10s_mask.npy
# fi

# if [ $MERT == TRUE ]; then
#   hdfs dfs -get $ARNOLD_BASE_DIR/home/byte_speech_sv/zongyu.yin/assets/mert/inference_test .
# fi

if [ $SPARSE_GPT == TRUE ]; then
  \cp recipes/audio_lm/scripts/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/
  # \cp recipes/audio_lm/scripts/matmul.py /home/tiger/.local/lib/python3.9/site-packages/triton/ops/blocksparse/
  hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/coarse_10s_mask_v3.npy
  hdfs dfs -get hdfs://harunava/home/byte_speech_sv/zongyu.yin/assets/semantic_10s_mask.npy
fi

bash launch.sh $@