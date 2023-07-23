#!/bin/bash -ex

stage=$1
batch_total_tokens=$2 # 15000 for A100

dataname=1400_librilight_90-1492
log_dir=/mnt/bn/jcong5/logs/samantha/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/mnt/bn/jcong5/data/llm_avearge_data/all/1400_librilight_90-1492/meta_list_all.txt.filter.train.add_metalen
valid_meta_lst=/mnt/bn/jcong5/data/llm_avearge_data/all/1400_librilight_90-1492/meta_list_all.txt.filter.dev.1024

sudo cp recipes/speartts/patch/data.py /usr/local/lib/python3.9/dist-packages/lightning_fabric/utilities/data.py
# ar
if [ ${stage} -le 1 ];then
    log_name=ar
    version=fp16_lr5e-5_batch${batch_total_tokens}
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    base_config="--config recipes/valle/conf/valle_phoneme_coarse_dynba.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.00005 \
                --run_opts.return_full_seq False \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.precision 16 \
                --run_opts.batch_total_tokens $batch_total_tokens"

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
fi

if [ ${stage} -le 2 ];then
    log_name=nar
    version="fp16_lr5e-5_batch${batch_total_tokens}"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/$log_name --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/valle/conf/valle_phoneme_fine_dynba.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.00005 \
                --run_opts.return_full_seq True \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.precision 16 \
                --run_opts.batch_total_tokens $batch_total_tokens"
    
    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
    exit 0
fi

if [ ${stage} -le 3 ];then
    ar_version=fp16_lr5e-5_batch${batch_total_tokens}
    nar_version=fp32_lr5e-5_batch${batch_total_tokens}
    
    log_dir=/mnt/bn/jcong5/logs/samantha/valle_phoneinput_1400_librilight_90-1492--1-8/
    
    for rank in 0; # 1 2 3 4 5 6 7;
    do
        python3 recipes/valle/scripts/infer_valle.py \
            --ar_ckpt_path $log_dir/ar/$ar_version/checkpoints/last.ckpt \
            --nar_ckpt_path $log_dir/nar/$nar_version/checkpoints/last.ckpt \
            --device cuda:$rank \
            --meta_file /mnt/bn/jcong5/data/librispeech_test_clean_fix/4-10s_metas/thread-0$rank.lst \
            --out_dir $log_dir/ar/$ar_version/wav_infer
    done
    out_dir=$log_dir/ar/$ar_version/
    wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
    sed 's#WAV_PRED_DIR#'$out_dir/wav_infer'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
    bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    cd -
    exit 0
fi
