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

train_meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/en_TTS_trim/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.train.rm_libritts
valid_meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/en_TTS_trim/meta_list_all.txt.add_metalen.fiter_90_1492.shuf.valid.rm_libritts

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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS_trim_rm_libritts"
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
                    --run_opts.n_step_save 30 \
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
    version="fp16_lr4e-4_batch${batch_total_tokens}_sft_en_TTS_trim_rm_libritts"
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

### librispeech_test_clean + minlen
if [ ${stage} -eq 3 ];then
    cd $work_dir
    spk_id=librispeech_test_clean
    testset=norm

    for x in 'epoch=25-step=107190-accu=36.50.ckpt' 'epoch=30-step=107370-accu=33.77.ckpt' 'epoch=35-step=107520-accu=41.97.ckpt' 'epoch=40-step=107700-accu=46.42.ckpt' \
                'epoch=45-step=107880-accu=48.82.ckpt' 'epoch=50-step=108030-accu=44.56.ckpt' 'epoch=55-step=108210-accu=48.44.ckpt' 'epoch=60-step=108390-accu=57.44.ckpt' \
                'epoch=65-step=108540-accu=55.95.ckpt';do
        coarse_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch80000_sft_en_TTS_trim_rm_libritts/checkpoints/$x
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