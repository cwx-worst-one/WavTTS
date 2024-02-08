#!/bin/bash -ex

cd /opt/tiger/samantha/

sudo apt -q update && sudo apt -q install -y libsndfile1 ffmpeg
pip3 install wheel Cython
pip3 install -r requirements.2.txt
pip3 install -r recipes/audio_diffusion/requirements.txt
pip3 install nnaudio
pip3 install numpy==1.23.5
pip install lion-pytorch -i https://pypi.python.org/simple
pip3 install pretty_midi
pip3 install pydantic==1.10.7
pip3 install soundfile
pip3 install deepspeed
pip3 install rotary_embedding_torch
pip3 install madmom

# Compile cauchy_mult CUDA kernel for S4 layer-type
# cd recipes/audio_diffusion/modules/model_types/s4_layer/cauchy_cuda_kernel/
# sudo python3 setup.py install
# cd /opt/tiger/samantha/

if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT is undefined, setting to default"
  export PROJECT_ROOT=/mnt/bn/audio-diffusion
fi

if [[ -z "${DATA_DIR}" ]]; then
  echo "DATA_DIR is undefined, setting to default"
  export DATA_DIR="${PROJECT_ROOT}"/data
fi

export ARNOLD_OUTPUT="${PROJECT_ROOT}"/logs/minz/$ARNOLD_TASK_ID/trials/$ARNOLD_TRIAL_ID

pip3 install tensorboard
if [ -z "$ARNOLD_TENSORBOARD_CURRENT_PORT" ]
then
    echo "TensorBoard port not defined, running on default port 6006"
    tensorboard --logdir $ARNOLD_OUTPUT --bind_all &
else
    tensorboard --logdir $ARNOLD_OUTPUT --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
fi

pip install ruamel.yaml==0.17.24
pip3 install mir_eval
pip3 install scipy==1.10.1
pip3 install nnAudio


# tune model
# echo "Searching for the best batch size..."
# CUDA_VISIBLE_DEVICES=0 ARNOLD_WORKER_NUM=1 sh launch.sh tune $@ --trainer.strategy dp 2> .tune_log || true
# BATCH_SIZE=$(grep -oP '(?<=Finished batch size finder, will continue with full run using batch size )[0-9]+' .tune_log)
# fit model
# echo "Found best batch size: $((BATCH_SIZE-1))"
sh launch.sh fit $@ # --training_params.batch_size $((BATCH_SIZE-1))
