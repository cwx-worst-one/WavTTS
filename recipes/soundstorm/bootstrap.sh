#!/bin/bash -ex

cd $(dirname $0)/../../

pip3 install -qr ./recipes/soundstorm/requirements.txt

if [[ -z "${ARNOLD_TRIAL_ID}" ]]; then
    export ARNOLD_OUTPUT="logs"
else
    export ARNOLD_OUTPUT=/mnt/bn/audio-diffusion/logs/soundstorm/${ARNOLD_TRIAL_ID}
fi

sh launch.sh $@