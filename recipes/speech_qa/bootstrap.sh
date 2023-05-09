#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

sh .codebase/pipelines/install_dependencies.sh

echo "=============== install flash attention ================="
if [ ! -d flash-attention ]; then
  git clone "https://wangxin.colin:7GYg3s6WRkovqxZrpKE5@code.byted.org/lab-audio/flash-attention.git"
fi

export PYTHONPATH=$(realpath flash-attention):$PYTHONPATH

if [[ $ARNOLD_DEVICE_TYPE =~ A100 && $USE_FLASH_ATTN == TRUE ]]; then
  cd flash-attention || exit 1
  sudo pip3 install . || exit 1
  cd -
fi

pip3 install -q --upgrade pip -i https://bytedpypi.byted.org/simple
pip3 install -q -r recipes/speech_qa/requirements.txt
pip3 install -q -r requirements.txt

echo "=============== install transformers dev ================="
git clone "https://zongyu.yin:R91y4u2tJkJH1RQYF6p5@code.byted.org/zongyu.yin/transformers.git"
cd transformers || exit 1
pip3 install -e . || exit 1
cd -

pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple

if [ $MULAN_115 == TRUE ]; then
  pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  if [ $ARNOLD_BASE_DIR == "hdfs://haruna" ]; then
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/zongyu.yin/assets/mulan_115/inference_test
  fi
fi

if [ $SPARSE_GPT == TRUE ]; then
  cp recipes/speech_qa/scripts/matmul.py /home/tiger/.local/lib/python3.8/site-packages/triton/ops/blocksparse/
fi

if [ ! -d inference_test ]
then
    hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/litang/musiclm/inference_test .
fi

bash launch.sh $@