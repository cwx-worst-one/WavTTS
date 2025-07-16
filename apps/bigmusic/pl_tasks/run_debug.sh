#!/bin/bash

##### old image need update this ####
# if [[ "x$RH2_DEBUG_MODE" == "x1" && -f .inited_debug_env0 ]]; then
#     echo "skip init debug env"
# else
#     pip3 install ninja
#     pip3 install flash-attn --no-build-isolation
#     scm_install data.aml.cruise 1.0.0.3971
#     touch .inited_debug_env0
# fi

#### torch 2.4 just refresh this ####
if [[ "x$RH2_DEBUG_MODE" == "x1" && -f .inited_debug_env ]]; then
    echo "skip init debug env"
else
    pip3 install -i https://bytedpypi.byted.org/simple -U byted-seed-models byted-hdfs-io
    pip3 install confusables
    pip3 install -U 'jsonargparse[signatures]>=4.27.7'
    pip3 install byted-seed_kernels
    pip3 install -U byted-bumi  # fix flash-ce amp bug
    pip3 uninstall triton -y || true
    touch .inited_debug_env
fi

bash apps/bigmusic/zh_vocal/env_huggingface.sh;

set -ex
export PYTHONPATH=/opt/tiger/samantha:$PYTHONPATH
export tokenizer_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/logs/umm_conformer_unified_tts_rope_rmpad/umm_stage3_unified_tts_rope_bert-base-multilingual-uncased/checkpoints/step=220000.ckpt
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org;
# ARNOLD_WORKER_GPU=1 \
TORCHRUN \
    apps/bigmusic/pl_tasks/seed_models_train.py \
    fit \
    -c apps/bigmusic/pl_tasks/config.yaml \
    -c apps/bigmusic/pl_tasks/config_data.yaml $@