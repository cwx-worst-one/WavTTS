#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ batch_vc_decode.sh ]
#   Synopsis     [ Script to perform decoding for any-to-one voice conversion models in batch mode ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

upstream=offline_bn
task=task1
tag=umm_stage1_v1.1_L17
vocoder=/mnt/bn/cyz-lq-nas/s3prl_datasets/vcc2020/vocoder/hifigan_vctk+vcc2020

set -e

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# pip3 install -r downstream/a2o-vc-vcc2020/requirements.txt
# check arguments
# if [ $# != 4 ]; then
#     echo "Usage: $0 <upstream> <task> <tag> <vocoder_dir>"
#     exit 1
# fi

start_ep=9000
interval=1000
end_ep=10000

if [ ${task} == "task1" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${task} == "task2" ]; then
    trgspks=("TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1")
elif [ ${task} == "task1_all_task2_man" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2" "TMF1" "TMM1")
elif [ ${task} == "debug" ]; then
    trgspks=("TEF1")
fi

for trgspk in "${trgspks[@]}"; do
    for ep in $(seq ${start_ep} ${interval} ${end_ep}); do
        echo "Objective evaluation: Upstream ${upstream}, Ep ${ep}; trgspk ${trgspk}"
        expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
        expdir=result/vc/${expname}
        ./downstream/a2o-vc-vcc2020/decode.sh ${vocoder}/ ${expdir}/${ep} ${trgspk}
    done
done

voc_name=$(basename ${vocoder} | cut -d"_" -f 1)

python3 ./downstream/a2o-vc-vcc2020/find_best_epoch.py \
    --start_epoch ${start_ep} \
    --end_epoch ${end_ep} \
    --step_epoch ${interval} \
    --upstream ${upstream} --tag ${tag} --task ${task} --vocoder ${voc_name} \
    --expdir result/vc

unset http_proxy; unset https_proxy