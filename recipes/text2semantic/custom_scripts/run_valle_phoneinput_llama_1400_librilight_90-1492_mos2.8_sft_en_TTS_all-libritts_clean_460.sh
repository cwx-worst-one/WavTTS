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
log_dir_ar=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}
log_dir_nar=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/en_TTS_all-libritts_clean_460/meta_list_all.txt.add_metalen.shuf.train.fiter_90_1492.wer0.1
valid_meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/en_TTS_all-libritts_clean_460/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_1492.wer0.1

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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS_all-libritts_clean_460"
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
                    --run_opts.n_step_save 200 \
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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS"
    latest_ckpt_path=$log_dir_nar/$log_name/$version/checkpoints/last-v1.ckpt
    tensorboard --logdir $log_dir_nar/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/valle/conf/valle_phoneme_fine_dynba.yaml \
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


### x-librispeech_test_clean (old_v1)
if [ ${stage} -eq 3 ];then
    cd $work_dir
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/checkpoints
    coarse_path=$coarse_dir/'epoch=22-step=107200-accu=32.31.ckpt'
    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
    coarse_dir=`dirname $coarse_path`

    # for pair in env_daily-Amy_prompt1_betterstudio_librispeech_test_clean env_daily-Chrissy_prompt1_betterstudio_librispeech_test_clean \
    #             env_daily-Corey_prompt1_betterstudio_librispeech_test_clean env_daily-Herrick_prompt1_betterstudio_librispeech_test_clean \
    #             env_daily-F01_headset_prompt1_betterstudio_librispeech_test_clean env_daily-F01_phone_prompt1_betterstudio_librispeech_test_clean \
    #             env_daily-M01_headset_prompt1_betterstudio_librispeech_test_clean env_daily-M01_phone_prompt1_betterstudio_librispeech_test_clean;do
    for pair in youtube-librispeech_test_clean_prompt1_betterstudio;do
    # for pair in dina-librispeech_test_clean_prompt1 dina-librispeech_test_clean_prompt1_rmsil;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        coarse_stat=107k_epoch22
        fine_stat=1505k_epoch14
        # coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        # fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${testspk}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen_v2
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        ln -s $coarse_path $out_dir_model/last_ar.ckpt
        ln -s $fine_path $out_dir_model/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/meta.lst.${testset}
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
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen_v2.py \
                --ar_ckpt_path $out_dir_model/last_ar.ckpt \
                --nar_ckpt_path $out_dir_model/last_nar.ckpt \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json
            }&
        done
        wait
        if [ "$testspk" == 'librispeech_test_clean' ];then
            wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/wav_res_ref_text.proto.$testset
            cp $wav_res_ref_text_proto_path $out_dir/wav_res_ref_text
        else
            wav_res_ref_text_proto_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/wav_res_ref_text.proto
            prompt_wav_path=`head -n 1 $metalst | cut -d "|" -f 3`
            sed 's#PROMPT_WAV_PATH#'$prompt_wav_path'#g' $wav_res_ref_text_proto_path > $out_dir/wav_res_ref_text
        fi
        sed -i 's#WAV_PRED_DIR#'$out_dir/'#g' $out_dir/wav_res_ref_text
        bash recipes/valle/utils/run_evaluation.sh $out_dir/wav_res_ref_text 0
    done
    cd -
fi


### hqtts
if [ ${stage} -eq 4 ];then
    cd $work_dir
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/checkpoints
    coarse_path=$coarse_dir/'epoch=22-step=107200-accu=32.31.ckpt'
    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
    coarse_dir=`dirname $coarse_path`

#11labs_Adam_075-hq_tts_prompt2
    for pair in will-hq_tts;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        coarse_stat=107k_epoch22
        fine_stat=1505k_epoch14
        out_dir=$coarse_dir/../wav_infer/${testspk}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        ln -s $coarse_path $out_dir_model/last_ar.ckpt
        ln -s $fine_path $out_dir_model/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/meta.lst.${testset}
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
                --ar_ckpt_path $out_dir_model/last_ar.ckpt \
                --nar_ckpt_path $out_dir_model/last_nar.ckpt \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json
            }&
        done
        wait
    done
    cd -
fi


### x-librispeech_test_clean (new_v3) debug
if [ ${stage} -eq 5 ];then
    cd $work_dir
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/checkpoints
    coarse_paths=`ls -l $coarse_dir/*.ckpt | awk '{print $9}' | grep -v 'last' | awk 'NR%4==1'`
    coarse_path=$coarse_dir/'epoch=22-step=107200-accu=32.31.ckpt'
    coarse_dir=`dirname $coarse_path`
    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt

    for pair in librispeech_test_clean-norm;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        coarse_stat=107k_epoch22
        fine_stat=1505k_epoch14
        # coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        # fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${testspk}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen_newlab
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        ln -s $coarse_path $out_dir_model/last_ar.ckpt
        ln -s $fine_path $out_dir_model/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/meta.lst.${testset}
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
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen_newlab.py \
                --ar_ckpt_path $out_dir_model/last_ar.ckpt \
                --nar_ckpt_path $out_dir_model/last_nar.ckpt \
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


### x-librispeech_test_clean (noprompt)
if [ ${stage} -eq 6 ];then
    cd $work_dir
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/checkpoints
    coarse_path=`ls -l $coarse_dir/*.ckpt | awk '{print $9}' | grep -v 'last' | tail -n 1`
    coarse_path=$coarse_dir/'epoch=22-step=107200-accu=32.31.ckpt'
    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
    coarse_dir=`dirname $coarse_path`

    for pair in will-librispeech_test_clean;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        # coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        # fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        coarse_stat=107k_epoch22
        fine_stat=1505k_epoch14
        out_dir=$coarse_dir/../wav_infer/${testspk}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen_noprompt
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        ln -s $coarse_path $out_dir_model/last_ar.ckpt
        ln -s $fine_path $out_dir_model/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/meta.lst.${testset}
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
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen_noprompt.py \
                --ar_ckpt_path $out_dir_model/last_ar.ckpt \
                --nar_ckpt_path $out_dir_model/last_nar.ckpt \
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

### for_litang
if [ ${stage} -eq 7 ];then
    cd $work_dir
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/checkpoints
    coarse_path=$coarse_dir/'epoch=22-step=107200-accu=32.31.ckpt'
    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt
    coarse_dir=`dirname $coarse_path`

    for pair in librispeech_test_clean-norm;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        coarse_stat=107k_epoch22
        fine_stat=1505k_epoch14
        out_dir=$coarse_dir/../wav_infer/to_litang/${testspk}/${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen_for_litang
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        ln -s $coarse_path $out_dir_model/last_ar.ckpt
        ln -s $fine_path $out_dir_model/last_nar.ckpt

        metalst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/${testspk}/meta.lst.${testset}
        samantha_dir=$work_dir
        timestamp=$(date +%s)
        thread_dir=/tmp/thread_metas_$timestamp/
        sudo mkdir $thread_dir
        num_job=8
        num=`wc -l $metalst | awk -F' ' '{print $1}'`
        num_per_thread=`expr $num / $num_job + 1`
        sudo split -l $num_per_thread --additional-suffix=.lst -d $metalst $thread_dir/thread-

        for rank in {0..7};do
            python3 recipes/valle/scripts/infer_valle_pre-nar_raw_minlen.py \
                --ar_ckpt_path $out_dir_model/last_ar.ckpt \
                --nar_ckpt_path $out_dir_model/last_nar.ckpt \
                --device cuda:$rank \
                --meta_file $thread_dir/thread-0$rank.lst \
                --out_dir ${out_dir} \
                --metaid_to_textid_path $samantha_dir/recipes/valle/datasets/dict/metaid_to_textid.json \
                --save_full_seq True
        done
        wait
    done
    cd -
fi