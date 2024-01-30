#!/bin/bash
set -e

function hdfs_checkpoint_grep_step() {
    # usage: hdfs_checkpoint_grep_step hdfs_log_dir step
    # e.g. hdfs_checkpoint_grep_step  hdfs://xxxxxx/ 25000
    folder=$1
    filter="step=$2-"
    ret=$(hdfs dfs -ls $folder/checkpoints | grep -o '\bhdfs\S*$' | grep $filter )
    echo $ret
}

export log_dir=${log_dir:-hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/xieshuangyi/voicebox/logdir/PrefixLDM4_block32_300M_8A100_setting0_40hzWVAE_UMMv062}
export out_dir=${out_dir:-/opt/tiger/samantha/output}
export token_chunk=${token_chunk:-8000}
export chunk_overlap=${chunk_overlap:-0}


export step=${step:-200000}

export diffusion_ckpt_path=$(hdfs_checkpoint_grep_step $log_dir $step)
export ckpt=$(basename $diffusion_ckpt_path)
export exp=$(basename $log_dir)
# export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/xieshuangyi/voicebox/logdir/PrefixLDM4_block32_300M_8A100_setting0_40hzWVAE_UMMv062/checkpoints/epoch=00-step=205000-loss=0.53.ckpt
# export ckpt=epoch=00-step=205000-loss=0.53.ckpt
# export exp=PrefixLDM4_block32_300M_8A100_setting0_40hzWVAE_UMMv062



# export lang=zh # zh, en

languages=("zh" "en")

for lang in "${languages[@]}"; do
    export lang=$lang
    log_file=infer_$step_$lang.log
    bash -x apps/bigtts/umm/diffusion/scripts/recons_umm_wvae_streaming.sh 2>&1 | tee $log_file
done

for lang in "${languages[@]}"; do
    echo "------- result of $lang ------------"
    log_file=infer_$step_$lang.log
    cat $log_file | grep "WER: " -B 1 -A 2
    cat $log_file | grep "ASV: " -B 1 -A 3
done