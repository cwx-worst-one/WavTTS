#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE

sudo mv /usr/local/lib/python3.9/dist-packages/triton/compiler /usr/local/lib/python3.9/dist-packages/triton/compiler.bak
pip3 install bytedeuler --index-url=https://bytedpypi.byted.org/simple/

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

train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/11labs_Adam_075/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.train
valid_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/11labs_Adam_075/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.valid

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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_11labs_Adam_075_retry"
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
                    --run_opts.num_epochs 100 \
                    --run_opts.n_step_save 10 \
                    --run_opts.save_top_k 100"
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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_11labs_Adam_075_retry"
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
                --trainer.val_check_interval 50 \
                --run_opts.num_epochs 100 \
                --run_opts.n_step_save 50 \
                --run_opts.save_top_k 100"
    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path 
    fi
    cd -
fi


### testset1
if [ ${stage} -eq 3 ];then
    cd $work_dir
    spk_id=11labs_Adam_075
    testset=hard_prompt2_v3
 
    # for i in {0..10};do
    # for x in 'epoch=26-step=107050-accu=41.31.ckpt' 'epoch=96-step=107610-accu=97.76.ckpt';do
    for x in 'epoch=96-step=107610-accu=97.76.ckpt';do
        coarse_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/ar/fp16_lr4e-4_batch110000_sft_11labs_Adam_075_retry/checkpoints/$x
        fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
        coarse_dir=`dirname $coarse_path`

        coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${spk_id}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_retry${i}_minlen
        [ ! -d $out_dir ] && mkdir -p $out_dir

        # cp $coarse_path $out_dir/last_ar.ckpt
        # cp $fine_path $out_dir/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${spk_id}/meta.lst.${testset}
        samantha_dir=$work_dir
        timestamp=$(date +%s)
        thread_dir=/tmp/thread_metas_$timestamp/
        sudo mkdir $thread_dir
        num_job=8
        num=`wc -l $metalst | awk -F' ' '{print $1}'`
        num_per_thread=`expr $num / $num_job + 1`
        sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst $thread_dir/thread-

        for rank in {0..7};do
            {
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen.py \
                --ar_ckpt_path $coarse_path \
                --nar_ckpt_path $fine_path \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json
            }&
        done
        wait
        # wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
        # sed 's#WAV_PRED_DIR#'$out_dir/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
        # bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0

        # sox $out_dir/001-00{1..8}.wav $out_dir/001.wav
        # sox $out_dir/002-0{01..10}.wav $out_dir/002.wav
        # sox $out_dir/003-0{01..09}.wav $out_dir/003.wav
        # sox $out_dir/004-0{01..04}.wav $out_dir/004.wav
        # sox $out_dir/005-0{01..04}.wav $out_dir/005.wav
    done
    # done
    cd -
fi

### librispeech_test_clean
if [ ${stage} -eq 4 ];then
    cd $work_dir
    spk_id=11labs_Adam_075
    testset=librispeech_test_clean_prompt2
 
    for x in 'epoch=21-step=107010-accu=31.79.ckpt' 'epoch=26-step=107050-accu=41.31.ckpt' 'epoch=31-step=107090-accu=45.73.ckpt' 'epoch=36-step=107130-accu=50.57.ckpt' \
            'epoch=41-step=107170-accu=54.85.ckpt' 'epoch=46-step=107210-accu=59.46.ckpt' 'epoch=51-step=107250-accu=62.57.ckpt' 'epoch=56-step=107290-accu=75.40.ckpt' \
            'epoch=61-step=107330-accu=84.04.ckpt' 'epoch=66-step=107370-accu=89.16.ckpt' 'epoch=71-step=107410-accu=94.06.ckpt' 'epoch=76-step=107450-accu=95.77.ckpt' \
            'epoch=81-step=107490-accu=96.94.ckpt' 'epoch=86-step=107530-accu=97.76.ckpt' 'epoch=91-step=107570-accu=97.65.ckpt' 'epoch=96-step=107610-accu=97.76.ckpt';do
        coarse_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/ar/fp16_lr4e-4_batch110000_sft_11labs_Adam_075_retry/checkpoints/$x
        fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
        coarse_dir=`dirname $coarse_path`

        coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${spk_id}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}
        [ ! -d $out_dir ] && mkdir -p $out_dir

        cp $coarse_path $out_dir/last_ar.ckpt
        cp $fine_path $out_dir/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${spk_id}/meta.lst.${testset}
        samantha_dir=$work_dir
        timestamp=$(date +%s)
        thread_dir=/tmp/thread_metas_$timestamp/
        sudo mkdir $thread_dir
        num_job=8
        num=`wc -l $metalst | awk -F' ' '{print $1}'`
        num_per_thread=`expr $num / $num_job + 1`
        sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst $thread_dir/thread-

        for rank in {0..7};do
            {
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw.py \
                --ar_ckpt_path $coarse_path \
                --nar_ckpt_path $fine_path \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json
            }&
        done
        wait
        wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
        sed 's#WAV_PRED_DIR#'$out_dir/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
        bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    done
    cd -
fi


### librispeech_test_clean + minlen
if [ ${stage} -eq 5 ];then
    cd $work_dir
    spk_id=11labs_Adam_075
    testset=librispeech_test_clean_prompt2

    for x in 'epoch=96-step=107610-accu=97.76.ckpt';do
        coarse_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-1-8/ar/fp16_lr4e-4_batch110000_sft_11labs_Adam_075_retry/checkpoints/$x
        fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
        coarse_dir=`dirname $coarse_path`

        coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${spk_id}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen
        [ ! -d $out_dir ] && mkdir -p $out_dir

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${spk_id}/meta.lst.${testset}
        samantha_dir=$work_dir
        timestamp=$(date +%s)
        thread_dir=/tmp/thread_metas_$timestamp/
        sudo mkdir $thread_dir
        num_job=8
        num=`wc -l $metalst | awk -F' ' '{print $1}'`
        num_per_thread=`expr $num / $num_job + 1`
        sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst $thread_dir/thread-

        for rank in {0..7};do
            {
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen.py \
                --ar_ckpt_path $coarse_path \
                --nar_ckpt_path $fine_path \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json
            }&
        done
        wait
        wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/testset/LibriSpeech_test_clean/wav_res_ref_text.proto
        sed 's#WAV_PRED_DIR#'$out_dir/'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
        bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    done
    cd -
fi