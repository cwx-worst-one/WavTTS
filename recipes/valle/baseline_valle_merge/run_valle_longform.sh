#!/bin/bash -ex

set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE
stage=$1
batch_total_tokens=$2 # 140000 for A100 for llama AR
mode=$3  # train or debug
dataname=$4
ckpt_name=$5
restore_ckpt=$6
spk_id=$7 #only used for infer mode 5


pip3 install -q bytedeuler peft sentencepiece ffmpeg --index-url=https://bytedpypi.byted.org/simple/
pip3 install --no-deps torchdata --index-url=https://bytedpypi.byted.org/simple/
if [ $mode == "infer" ]; then
    if [ $restore_ckpt != "" ]; then
        nar_ckpt_name=$restore_ckpt
    else
        nar_ckpt_name=$ckpt_name
    fi
fi


if [[ $DYN_BATCH_SIZE == TRUE ]]; then
    sudo cp /opt/tiger/samantha/recipes/speartts/patch/data.py /usr/local/lib/python3.9/dist-packages/lightning_fabric/utilities/data.py
fi

GPU_TYPE=`nvidia-smi -q | grep "Product Name" | head -n1 | awk '{print $NF}' | awk -F"-" '{print $1"-"$3}'`

log_dir=/mnt/bn/colin-workspace/exp/logs
# nar_log_dir only used in infer mode
nar_log_dir=/mnt/bn/colin-workspace/exp/logs

train_wds_lst=/mnt/bn/jeffus/data/metas/valle/${dataname}/wds.lst

work_dir=/opt/tiger/samantha


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# ar
if [ ${stage} -eq 1 ];then
    cd $work_dir
    log_name="ar"
    version="ctiga_llama_20k_no_ckpt"
    latest_ckpt_path=${restore_ckpt:-$log_dir/$log_name/$version/checkpoints/last.ckpt}
    # latest_ckpt_path=/mnt/bn/jeffus/pretrain/valle_baseline_v1/last_ar.ckpt
    echo "restore ckpt: ${latest_ckpt_path}"
    # tensorboard --logdir $log_dir/${log_name} --port $ARNOLD_TENSORBOARD_CURRENT_PORT --bind_all &

    base_config="--config recipes/valle/baseline_valle_merge/config/valle_llama_ar.yaml \
                --run_opts.batch_total_tokens $batch_total_tokens \
                --run_opts.train_wds_lst $train_wds_lst \
                --run_opts.log_dir $log_dir \
                --run_opts.log_name $log_name \
                --run_opts.version $version \
                --run_opts.return_full_seq False \
                --run_opts.n_step_save 1000 \
                --run_opts.max_len 8000 \
                --run_opts.learning_rate 0.0003 \
                --run_opts.precision 16 \
                --run_opts.buffer_size 2000 \
                --run_opts.enable_buffer_length True  \
                --run_opts.quality_check False \
                --run_opts.checkpointing False \
                --pl_module.provider ctiga
                "

    if [ ! -f "$latest_ckpt_path" ];then
        bash launch.sh fit $base_config
    else
        bash launch.sh fit $base_config --ckpt_path $latest_ckpt_path
    fi
    cd -
fi
