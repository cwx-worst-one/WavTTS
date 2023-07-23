#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@

stage=$1
batch_size=$2

log_dir=/mnt/bn/jcong5/logs/samantha/valle_phoneinput_1400h
train_meta_lst=/mnt/bn/jcong5/data/llm_avearge_data/1400h/meta_list_all.txt.filter_1492.train.1400h
valid_meta_lst=/mnt/bn/jcong5/data/llm_avearge_data/1400h/meta_list_all.txt.filter_1492.dev.1400h

# ar
if [ ${stage} -le 1 ];then
    bash launch.sh fit --config recipes/valle/conf/valle_phoneme_coarse.yaml \
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
    bash launch.sh fit --config recipes/valle/conf/valle_phoneme_fine.yaml \
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
        --ar_ckpt_path /mnt/bn/jcong5/logs/samantha/valle_phoneinput_1400h/ar/fp32-lr5e-5-batch64/checkpoints/last.ckpt \
        --nar_ckpt_path /mnt/bn/jcong5/logs/samantha/valle_phoneinput_1400h/nar/fp32/checkpoints/last.ckpt \
        --meta_file /mnt/bn/jcong5/data/kat_test/4-10s_libri/thread-00.lst \
        --device cuda \
        --out_dir $log_dir/ar/fp32-lr5e-5-batch64/epoch=18-step=120000-val_token_acc
    exit 0
fi