#!/bin/bash -ex

cd /opt/tiger/samantha

if [ "$ARNOLD_REGION" = "CN" ]; then
    export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
    export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
fi

# pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
sudo apt -q update && sudo apt -q install -y libsndfile1 ffmpeg
pip3 install wheel Cython

# pip3 install -q -r recipes/mulan/requirements.txt
# pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
pip3 install peft bitsandbytes


# ruamel.yaml>=0.17.28
pip3 install einops
pip3 install pykeops
pip3 install frechet_audio_distance
pip3 install h5py
pip3 install a-unet
pip3 install music21


# pip3 install -r requirements.2.txt
# pip3 install -r recipes/audio_diffusion/requirements.txt
pip3 install numpy==1.23.4
pip3 install lion-pytorch -i https://pypi.python.org/simple
pip3 install pretty_midi
pip3 install pydantic==1.10.7
pip3 install deepspeed==0.8.3
pip3 install rotary_embedding_torch==0.6.2
pip3 install pytorch-lightning==2.0.0
pip3 install madmom==0.16.1
pip3 install triton==2.0.0

pip3 install hyperpyyaml
pip3 install command
pip3 install bytedance.easycycle~=1.1.0
pip3 install onnxruntime
# sed -i 's/np\.float/float/g' ~/.local/lib/python3.9/site-packages/madmom/io/__init__.py
# sed -i 's/np\.int/int/g' ~/.local/lib/python3.9/site-packages/madmom/evaluation/chords.py
# sed -i 's/np\.float/float/g' ~/.local/lib/python3.9/site-packages/madmom/evaluation/chords.py


# Compile cauchy_mult CUDA kernel for S4 layer-type
# cd recipes/audio_diffusion/modules/model_types/s4_layer/cauchy_cuda_kernel/
# sudo python3 setup.py install
# cd /opt/tiger/sami_ai_models/

if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT is undefined, setting to default"
  export PROJECT_ROOT=/mnt/bn/bigmusic-lf
fi

if [[ -z "${DATA_DIR}" ]]; then
  echo "DATA_DIR is undefined, setting to default"
  export DATA_DIR="${PROJECT_ROOT}"/data
fi

# export ARNOLD_OUTPUT="${PROJECT_ROOT}"/logs/ju-chiang.wang/$ARNOLD_TASK_ID/trials/$ARNOLD_TRIAL_ID

# pip3 install tensorboard
# if [ -z "$ARNOLD_TENSORBOARD_CURRENT_PORT" ]
# then
#     echo "TensorBoard port not defined, running on default port 6006"
#     tensorboard --logdir $ARNOLD_OUTPUT --bind_all &
# else
#     tensorboard --logdir $ARNOLD_OUTPUT --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
# fi

pip install ruamel.yaml==0.17.28
pip3 install mir_eval
pip3 install scipy==1.10.1
pip3 install nnAudio
pip3 uninstall -y wandb
pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
sudo pip3 uninstall -y soundfile pysoundfile
pip3 install soundfile
pip3 install librosa==0.10.2.post1

# tune model
# echo "Searching for the best batch size..."
# CUDA_VISIBLE_DEVICES=0 ARNOLD_WORKER_NUM=1 sh launch.sh tune $@ --trainer.strategy dp 2> .tune_log || true
# BATCH_SIZE=$(grep -oP '(?<=Finished batch size finder, will continue with full run using batch size )[0-9]+' .tune_log)
# fit model
# echo "Found best batch size: $((BATCH_SIZE-1))"
sh launch.sh $@ # --training_params.batch_size $((BATCH_SIZE-1))


