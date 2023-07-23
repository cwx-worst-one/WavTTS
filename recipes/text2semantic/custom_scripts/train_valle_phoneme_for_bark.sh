#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@

stage=$1
batch_size=$2

log_dir=/opt/tiger/liuzhengxi/samantha/bark_log
train_meta_lst=/opt/tiger/liuzhengxi/podcast_data/podcast_metalist_training.txt
valid_meta_lst=/opt/tiger/liuzhengxi/podcast_data/podcast_metalist_dev.txt

# ar
if [ ${stage} -le 1 ];then
    log_name=ar
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    bash launch.sh fit --config recipes/valle/conf/valle_phoneme_coarse_for_bark.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq False \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name "ar" \
        --run_opts.version "fp32-exp2"
    exit 0
fi

if [ ${stage} -le 2 ];then
    log_name=nar
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    bash launch.sh fit --config recipes/valle/conf/valle_phoneme_fine_for_bark.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq True \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name "nar" \
        --run_opts.version "fp32"
    exit 0
fi

if [ ${stage} -le 3 ];then
    python3 recipes/valle/scripts/infer_valle.py \
        --ar_ckpt_path /opt/tiger/liuzhengxi/ar.ckpt \
        --nar_ckpt_path /opt/tiger/liuzhengxi/nar.ckpt \
        --meta_file /opt/tiger/liuzhengxi/podcast_data/podcast_metalist_dev.txt \
        --device cuda \
        --out_dir inference_output
    exit 0
fi