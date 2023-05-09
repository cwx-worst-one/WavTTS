#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

stage=$1
batch_size=$2
mode=$3  # train or debug

GPU_TYPE=`nvidia-smi -q | grep "Product Name" | head -n1 | awk '{print $NF}' | awk -F"-" '{print $1"-"$3}'`

dataname=libri_light-WenetSpeech-1400h-1000h
log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/speartts/exp_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha
    export CUDA_VISIBLE_DEVICES=0
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
fi

# text2semantic_ar
if [ ${stage} -le 1 ];then
    train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume5/data/bytegen/text2semantic/${dataname}/meta_list_all.txt.filter_1492.train_unmerged
    valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume5/data/bytegen/text2semantic/${dataname}/meta_list_all.txt.filter_1492.test_unmerged
    cd $work_dir
    tensorboard --logdir $log_dir/text2semantic_ar --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
    log_name="text2semantic_ar"
    version="fp32"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/conf/text2semantic_coarse_${dataname}.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq False \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name $log_name \
        --run_opts.version $version"

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt $latest_ckpt_path 
    fi
    cd -
    exit 0
fi

# semantic2acoustic_ar
if [ ${stage} -le 2 ];then
    train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume1/data/bytegen/valle_vc/${dataname}/meta_list_train_unmerged.txt
    valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume1/data/bytegen/valle_vc/${dataname}/meta_list_test_unmerged.txt
    cd $work_dir
    tensorboard --logdir $log_dir/semantic2acoustic_ar --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
    log_name="semantic2acoustic_ar"
    version="fp32"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/conf/semantic2acoustic_coarse_${dataname}.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq False \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name $log_name \
        --run_opts.version $version"

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt $latest_ckpt_path 
    fi
    cd -
    exit 0
fi

# semantic2acoustic_nar
if [ ${stage} -le 3 ];then
    train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume1/data/bytegen/valle_vc/${dataname}/meta_list_train_unmerged.txt
    valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume1/data/bytegen/valle_vc/${dataname}/meta_list_test_unmerged.txt
    cd $work_dir
    tensorboard --logdir $log_dir/nar --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &
    log_name="semantic2acoustic_nar"
    version="fp32"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/conf/semantic2acoustic_fine_${dataname}.yaml \
        --run_opts.train_meta_lst $train_meta_lst \
        --run_opts.valid_meta_lst $valid_meta_lst \
        --run_opts.batch_size $batch_size \
        --run_opts.learning_rate 0.00005 \
        --run_opts.return_full_seq True \
        --run_opts.log_dir $log_dir \
        --run_opts.log_name $log_name \
        --run_opts.version $version"

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt $latest_ckpt_path 
    fi
    cd -
    exit 0
fi

if [ ${stage} -le 4 ];then
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
