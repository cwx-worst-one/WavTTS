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
        --run_opts.learning_rate 0.0001 \
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
        --ar_ckpt_path $log_dir/ar/fp32/checkpoints/epoch=00-step=10000-val_token_acc=0.22.ckpt \
        --nar_ckpt_path $log_dir/nar/fp32/checkpoints/epoch=07-step=100000-val_loss=3.96.ckpt \
        --codec_config /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/bytegen/valle/exp_libri_light-1400h_ar-A100-80GB-1-8/config.yaml \
        --codec_ckpt /mnt/bn/jcong5/workspace/models/soundstream/2023-01-17_causal_x300_1024_6book_doubleG/latest_ckpt.pyt \
        --meta_file /mnt/bn/jcong5/data/kat_test/4-10s_libri/thread-00.lst \
        --device cuda \
        --out_dir $log_dir/ar/fp32/test
    exit 0
fi