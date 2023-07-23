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

dataname=en_zh_TTS
log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/en_zh_TTS/meta_list_all.txt.add_metalen.shuf.train.fiter_90_2040.rm_libritts.add_spkid
valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/en_zh_TTS/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_2040.rm_libritts.add_spkid

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha
    export CUDA_VISIBLE_DEVICES=0
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
    version="llama_fp16_lr3e-4_batch${batch_total_tokens}_90_2040_rm_libritts_add_spkid"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/bark/conf/llama/valle_coarse_llama_add_spkid.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.return_full_seq False \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.batch_total_tokens $batch_total_tokens \
                --distributed_batch_sampler_similar_length.seed $time_ms \
                --llama_config.sparse=True \
                --run_opts.num_epochs 200 \
                --hp.phone_tokens_num 8000 \
                --hp.spk_num 1440 \
                --hp.spk_dict recipes/valle/datasets/dict/spk_en_zh_TTS.json \
                --llama_config.vocab_size 10467 \
                --llama_config.out_dim 10467 \
                --pl_module.tokenizer_len 8000 "
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
    version="gpt2_fp16_lr4e-4_batch${batch_total_tokens}_90_2040_rm_libritts_add_spkid"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/valle/conf/valle_phoneme_fine_dynba_add_spkid.yaml \
                --run_opts.train_meta_lst $train_meta_lst \
                --run_opts.valid_meta_lst $valid_meta_lst \
                --run_opts.learning_rate 0.0004 \
                --run_opts.return_full_seq True \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.precision 16 \
                --run_opts.batch_total_tokens $batch_total_tokens \
                --distributed_batch_sampler_similar_length.seed $time_ms \
                --run_opts.num_epochs 200 \
                --hp.phone_tokens_num 8000 \
                --hp.spk_num 1440 \
                --hp.spk_dict recipes/valle/datasets/dict/spk_en_zh_TTS.json \
                --nar_config.vocab_size 10467 \
                --nar_config.n_positions 4096 \
                --nar_config.n_ctx 4096 \
                --trainer.val_check_interval 1000"
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
    ar_ckpt_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000/checkpoints/last-v1.ckpt
    ar_ckpt_dir=`dirname $ar_ckpt_path`
    nar_ckpt_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/nar/fp16_lr5e-5_batch15000_pretrain/checkpoints/last.ckpt
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
    wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
    sed 's#WAV_PRED_DIR#'$out_dir/gen/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
    bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    cd -
    exit 0
fi
