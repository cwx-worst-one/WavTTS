#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE

sudo mv /usr/local/lib/python3.9/dist-packages/triton/compiler /usr/local/lib/python3.9/dist-packages/triton/compiler.bak

if [ $SPARSE_GPT == TRUE ]; then
  echo "Install modified triton"
  pip3 install sentencepiece
  pip3 install -U --pre triton -i https://bytedpypi.byted.org/simple
  sudo cp recipes/audio_lm/scripts/matmul.py /usr/local/lib/python3.9/dist-packages/triton/ops/blocksparse/
fi

if [ $DYN_BATCH_SIZE == TRUE ]; then
    sudo cp /opt/tiger/samantha/recipes/speartts/patch/data.py /usr/local/lib/python3.9/dist-packages/lightning_fabric/utilities/data.py
fi

stage=$1
batch_total_tokens=$2 # 15000 for A100
mode=$3  # train or debug

GPU_TYPE=`nvidia-smi -q | grep "Product Name" | head -n1 | awk '{print $NF}' | awk -F"-" '{print $1"-"$3}'`
time_ms=`echo $[$(date +%s%N)/1000000]`

dataname=1400_librilight_90-1492_mosnet2.8
log_dir_ar=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}
log_dir_nar=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume7/data/bytegen/valle/en_TTS_trim/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.train
valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume7/data/bytegen/valle/en_TTS_trim/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.valid

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha
    # export CUDA_VISIBLE_DEVICES=0
fi

if [ $mode == "train" ]; then
    work_dir=/opt/tiger/samantha
    cd $work_dir/scripts
    pip3 install -r requirements.txt
    cd -
fi

# ar
if [ ${stage} -eq 1 ];then
    cd $work_dir
    log_name="ar"
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS_trim"
    latest_ckpt_path=$log_dir_ar/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir_ar/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/bark/conf/llama/valle_coarse_llama.yaml \
                    --run_opts.train_meta_lst $train_meta_lst \
                    --run_opts.return_full_seq False \
                    --run_opts.log_dir $log_dir_ar \
                    --run_opts.log_name $log_name \
                    --run_opts.version $version \
                    --run_opts.batch_total_tokens $batch_total_tokens \
                    --distributed_batch_sampler_similar_length.seed $time_ms \
                    --llama_config.sparse=True \
                    --run_opts.num_epochs 500"
    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
    cd -
fi

if [ ${stage} -eq 2 ];then
    cd $work_dir
    log_name="nar"
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS_trim"
    latest_ckpt_path=$log_dir_nar/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir_nar/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/conf/valle_phoneme_fine_dynba.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.0004 \
                --run_opts.return_full_seq True \
                --run_opts.log_dir $log_dir_nar \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.precision 16 \
                --run_opts.batch_total_tokens $batch_total_tokens \
                --distributed_batch_sampler_similar_length.seed $time_ms \
                --trainer.val_check_interval 1000 \
                --run_opts.num_epochs 500"
    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
    cd -
fi


if [ ${stage} -eq 3 ];then
    cd $work_dir
    log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8
    ar_version=fp16_lr4e-4_batch${batch_total_tokens}
    nar_version=fp16_lr4e-4_batch${batch_total_tokens}

    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $log_dir/ar/$ar_version/checkpoints/{last.ckpt,temp}
    ar_model_step=`cat $log_dir/ar/$ar_version/checkpoints/temp | cut -d " " -f 1`
    ar_model_epoch=`cat $log_dir/ar/$ar_version/checkpoints/temp | cut -d " " -f 2`
    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $log_dir/nar/$nar_version/checkpoints/{last.ckpt,temp}
    nar_model_step=`cat $log_dir/nar/$nar_version/checkpoints/temp | cut -d " " -f 1`
    nar_model_epoch=`cat $log_dir/nar/$nar_version/checkpoints/temp | cut -d " " -f 2`
    out_dir=$log_dir/ar/$ar_version/wav_infer/ar_step${ar_model_step}_epoch${ar_model_epoch}-nar_step${nar_model_step}_epoch${nar_model_epoch}
    [ ! -d $out_dir ] && mkdir -p $out_dir
    cp $log_dir/ar/$ar_version/checkpoints/last.ckpt $out_dir/last_ar.ckpt
    cp $log_dir/nar/$nar_version/checkpoints/last.ckpt $out_dir/last_nar.ckpt
    for rank in {0..7};do
        {
        python3 recipes/bark/scripts/infer_valle.py \
            --ar_ckpt_path $log_dir/ar/$ar_version/checkpoints/last.ckpt \
            --nar_ckpt_path $log_dir/nar/$nar_version/checkpoints/last.ckpt \
            --device cuda:$rank \
            --meta_file /mnt/bn/jcong5/data/librispeech_test_clean_fix/4-10s_metas/thread-0$rank.lst \
            --out_dir ${out_dir}
        }&
    done
    wait
    out_dir=$log_dir/ar/$ar_version/
    wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
    sed 's#WAV_PRED_DIR#'$out_dir/gen/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
    bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    cd -
    exit 0
fi

# pre-nar
if [ ${stage} -eq 4 ];then
    cd $work_dir
    log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/
    ar_version=fp16_lr4e-4_batch${batch_total_tokens}
    nar_ckpt_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/nar/fp16_lr5e-5_batch15000_pretrain/checkpoints/last.ckpt
    nar_ckpt_dir=`dirname $nar_ckpt_path`

    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $log_dir/ar/$ar_version/checkpoints/{last.ckpt,temp}
    ar_model_step=`cat $log_dir/ar/$ar_version/checkpoints/temp | cut -d " " -f 1`
    ar_model_epoch=`cat $log_dir/ar/$ar_version/checkpoints/temp | cut -d " " -f 2`
    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $nar_ckpt_path $nar_ckpt_dir/temp
    nar_model_step=`cat $nar_ckpt_dir/temp | cut -d " " -f 1`
    nar_model_epoch=`cat $nar_ckpt_dir/temp | cut -d " " -f 2`
    out_dir=$log_dir/ar/$ar_version/wav_infer_pre-nar/ar_step${ar_model_step}_epoch${ar_model_epoch}-nar_step${nar_model_step}_epoch${nar_model_epoch}/
    [ ! -d $out_dir ] && mkdir -p $out_dir
    cp $log_dir/ar/$ar_version/checkpoints/last.ckpt $out_dir/last_ar.ckpt
    cp $nar_ckpt_path $out_dir/last_nar.ckpt
    for rank in {0..7};do
        {
        python3 recipes/bark/scripts/infer_valle_pre-nar.py \
            --ar_ckpt_path $log_dir/ar/$ar_version/checkpoints/last.ckpt \
            --nar_ckpt_path $nar_ckpt_path \
            --device cuda:$rank \
            --meta_file /mnt/bn/jcong5/data/librispeech_test_clean_fix/4-10s_metas/thread-0$rank.lst \
            --out_dir $out_dir
        }&
    done
    wait
    wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
    sed 's#WAV_PRED_DIR#'$out_dir/gen/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
    bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    cd -
    exit 0
fi

# pre-nar + gpu24
if [ ${stage} -eq 5 ];then
    cd $work_dir
    ar_ckpt_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000_sft_en_TTS/checkpoints/last-v1.ckpt
    ar_ckpt_dir=`dirname $ar_ckpt_path`
    nar_ckpt_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000
    nar_ckpt_dir=`dirname $nar_ckpt_path`

    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $ar_ckpt_path $ar_ckpt_dir/temp
    ar_model_step=`cat $ar_ckpt_dir/temp | cut -d " " -f 1`
    ar_model_epoch=`cat $ar_ckpt_dir/temp | cut -d " " -f 2`
    python3 recipes/bark/utils/get_step_epoch_from_ckpt.py $nar_ckpt_path $nar_ckpt_dir/temp
    nar_model_step=`cat $nar_ckpt_dir/temp | cut -d " " -f 1`
    nar_model_epoch=`cat $nar_ckpt_dir/temp | cut -d " " -f 2`
    out_dir=$ar_ckpt_dir/../wav_infer_pre-nar/ar_step${ar_model_step}_epoch${ar_model_epoch}-nar_step${nar_model_step}_epoch${nar_model_epoch}/
    [ ! -d $out_dir ] && mkdir -p $out_dir
    cp $ar_ckpt_path $out_dir/last_ar.ckpt
    cp $nar_ckpt_path $out_dir/last_nar.ckpt
    for rank in {0..7};do
        {
        python3 recipes/bark/scripts/infer_valle_pre-nar.py \
            --ar_ckpt_path $ar_ckpt_path \
            --nar_ckpt_path $nar_ckpt_path \
            --device cuda:$rank \
            --meta_file /mnt/bn/jcong5/data/librispeech_test_clean_fix/4-10s_metas/thread-0$rank.lst \
            --out_dir $out_dir
        }&
    done
    wait
    # wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
    # sed 's#WAV_PRED_DIR#'$out_dir/gen/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
    # bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    cd -
    exit 0
fi
