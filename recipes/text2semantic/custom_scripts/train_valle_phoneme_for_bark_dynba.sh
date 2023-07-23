#!/bin/bash -ex

stage=$1
batch_total_tokens=$2

dataname=150h_podcast
log_dir=/opt/tiger/liuzhengxi/samantha/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/opt/tiger/liuzhengxi/podcast_data/all_training.txt
valid_meta_lst=/opt/tiger/liuzhengxi/podcast_data/podcast_metalist_dev.txt

sudo cp recipes/speartts/patch/data.py /usr/local/lib/python3.9/dist-packages/lightning_fabric/utilities/data.py
# ar
if [ ${stage} -le 1 ];then
    log_name=ar
    version=fp32_lr5e-5_batch${batch_total_tokens}
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    base_config="--config recipes/valle/conf/valle_phoneme_coarse_dynba_for_bark.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.00001 \
                --run_opts.return_full_seq False \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.batch_total_tokens $batch_total_tokens"

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
fi

if [ ${stage} -le 2 ];then
    log_name=nar
    version="fp32_lr5e-5_batch${batch_total_tokens}"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/valle/conf/valle_phoneme_fine_dynba_for_bark.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.00005 \
                --run_opts.return_full_seq True \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.batch_total_tokens $batch_total_tokens"
    
    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
    exit 0
fi

if [ ${stage} -le 3 ];then
    python3 recipes/valle/scripts/infer_valle.py \
        --ar_ckpt_path /opt/tiger/liuzhengxi/samantha/valle_phoneinput_150h_podcast--1-8/ar/fp32_lr5e-5_batch12000/checkpoints/epoch=00-step=50000-val_token_acc=0.24.ckpt \
        --nar_ckpt_path /opt/tiger/liuzhengxi/epoch=00-step=60000-val_loss=3.85.ckpt \
        --codec_ckpt_path hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export/ \
        --meta_file /opt/tiger/liuzhengxi/bark-data-process/podcast_metalist_dev.txt \
        --device cuda:0 \
        --out_dir output
    exit 0
fi