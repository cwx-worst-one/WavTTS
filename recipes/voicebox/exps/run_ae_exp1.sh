#!/bin/bash -ex

set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE

mode=$1
batch_total_tokens=30000

log_name="voicebox"
version="0.0.1"
data_lst="hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/BigTTS/librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_5_10s/package/wav_1.0_web_dataset_1/data/*/*.tar"
log_dir="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/congjian/logs/${log_name}/${version}"

if [[ ${mode} == "train" ]];then
    latest_ckpt_path=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/ckpt/last.ckpt

    echo "restore ckpt: ${latest_ckpt_path}"
    base_config="--config recipes/voicebox/conf/voicebox.yaml \
        --run_opts.urls ${data_lst} \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path ${log_dir} \
        --run_opts.log_name ${log_name} \
        --run_opts.version ${version} \
        --run_opts.batch_total_tokens ${batch_total_tokens} \
        --run_opts.precision bf16 \
        --run_opts.checkpointing False \
        --run_opts.n_step_save 5000 \
        --trainer.accumulate_grad_batches 2 \
        --run_opts.learning_rate 0.0003 \
        --scheduler_cls.cycle_steps 400000 \
        --run_opts.num_epochs 2000 \
        --voicebox_config.attn_pdrop 0.1 \
        --voicebox_config.resid_pdrop 0.1 \
        --run_opts.weight_decay 0.1 \
        --run_opts.num_workers 1 \
        --trainer.log_every_n_steps 100"

    bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path
fi

# 训练集内编辑测试
if [[ ${mode} == "infer_librilight" ]];then
    ckpt_path=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/ckpt/last.ckpt
    output_dir=/mnt/bn/jcong5/logs/voicebox/$version/librilight-editing/
    mkdir -p $output_dir
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/librilight/meta.out
    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types content_editing

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi

# 集外 fake-zero-shot 测试
if [[ ${mode} == "infer_fake_zero_shot" ]];then
    ckpt_path=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/ckpt/last.ckpt
    output_dir=/mnt/bn/jcong5/logs/voicebox/$version/icl-fighting-zero-shot-last/
    mkdir -p $output_dir
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi