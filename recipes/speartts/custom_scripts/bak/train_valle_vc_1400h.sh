#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

stage=$1
batch_size=$2

mode=train  # train or debug

GPU_TYPE=`nvidia-smi -q | grep "Product Name" | head -n1 | awk '{print $NF}' | awk -F"-" '{print $1"-"$3}'`
log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle_vc/exp_1400h-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}
train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume2/data/bytegen/valle_vc/en_1400h/meta_list_train_unmerged.txt.filter_1492
valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume2/data/bytegen/valle_vc/en_1400h/meta_list_test_unmerged.txt.filter_1492

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

# ar
if [ ${stage} -le 1 ];then
    cd $work_dir
    tensorboard --logdir $log_dir/ar --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
    bash launch.sh fit --config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle_vc/conf/valle_vc_coarse_1400h.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq False \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name "ar" \
        --run_opts.version "fp32-exp2"
    cd -
    exit 0
fi

if [ ${stage} -le 2 ];then
    cd $work_dir
    tensorboard --logdir $log_dir/nar --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
    bash launch.sh fit --config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle_vc/conf/valle_vc_fine_1400h.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq True \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name "nar" \
        --run_opts.version "fp32"
    cd -
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
