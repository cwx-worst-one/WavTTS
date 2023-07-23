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

dataname=libri_light_norm_trim_add_sp-923h
log_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_${dataname}-${GPU_TYPE}-${ARNOLD_WORKER_NUM}-${ARNOLD_WORKER_GPU}

train_meta_lst=/mnt/bd/huangzhiying-lq-valle-volume11/data/bytegen/valle/${dataname}/meta_list_all.txt.add_metalen.shuf.train.fiter_90_1492
# valid_meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/libri_light-1400h/meta_list_all.txt.filter.dev.1024

if [ $mode == "debug" ]; then
    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha
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
    version="fp16_lr4e-4_batch${batch_total_tokens}"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/bark/conf/llama/valle_coarse_llama.yaml \
                    --run_opts.train_meta_lst $train_meta_lst \
                    --run_opts.return_full_seq False \
                    --run_opts.log_dir $log_dir \
                    --run_opts.log_name $log_name \
                    --run_opts.version $version \
                    --run_opts.batch_total_tokens $batch_total_tokens \
                    --distributed_batch_sampler_similar_length.seed $time_ms \
                    --llama_config.sparse=True \
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
    version="fp16_lr4e-4_batch${batch_total_tokens}"
    latest_ckpt_path=$log_dir/$log_name/$version/checkpoints/last.ckpt
    tensorboard --logdir $log_dir/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/bark/conf/llama/valle_fine_llama.yaml \
                    --run_opts.train_meta_lst $train_meta_lst \
                    --run_opts.return_full_seq True \
                    --run_opts.log_dir $log_dir \
                    --run_opts.log_name $log_name \
                    --run_opts.version $version \
                    --run_opts.batch_total_tokens $batch_total_tokens \
                    --distributed_batch_sampler_similar_length.seed $time_ms \
                    --llama_config.sparse=True"
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
    coarse_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_libri_light_norm_trim_add_sp-923h-A800-80GB-1-8/ar/fp16_lr4e-4_batch140000/checkpoints/
    coarse_name="last.ckpt"
    coarse_path=$coarse_dir/$coarse_name
    coarse_dir=`dirname $coarse_path`

    fine_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000/checkpoints/last.ckpt

    for pair in librispeech_test_clean-norm will-librispeech_test_clean;do
        testspk=`echo $pair | cut -d "-" -f 1`
        testset=`echo $pair | cut -d "-" -f 2`

        coarse_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $coarse_path)
        fine_stat=$(python3 recipes/valle/utils/get_step_epoch_from_ckpt.py $fine_path)
        out_dir=$coarse_dir/../wav_infer/${testspk}_${testset}_coarse_${coarse_stat}_fine_${fine_stat}_minlen_v2
        [ ! -d $out_dir ] && mkdir -p $out_dir

        out_dir_model=${out_dir}_model_backup
        [ ! -d $out_dir_model ] && mkdir -p $out_dir_model
        cp $coarse_path $out_dir_model/last_ar.ckpt
        cp $fine_path $out_dir_model/last_nar.ckpt

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
        if [ $testspk == "librispeech_test_clean" ]; then
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
