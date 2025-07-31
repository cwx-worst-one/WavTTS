bash scripts/setup_cruise.sh 3897
export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
export LITE_USE_PL_MODULE=1
export PYTHONPATH=apps/mariana:$PYTHONPATH
config="recipes/umm2/conf/conformer_unified_data/dataloader/umm_conformer_stage1_unified_VQ_baseline_re_lite.yaml"
EXP_NAME=$(date  "+%Y%m%d%H")
EXP_NAME=${EXP_NAME}_UMM-stage1_64GPU_28mins_causal_model
echo ${EXP_NAME}

bash /opt/tiger/samantha/launch.sh fit \
    --config $config \
    --extra_params.num_seg_per_track 10 \
    --run_opts.max_steps 800000 \
    --run_opts.version ${EXP_NAME} \
    --run_opts.log_name "umm_stage1" \
    --run_opts.hdfs_log_dir "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/debug" \
    --run_opts.precision "16-mixed" \
    --data.train_batcher.maximum_bucket_size 33840000 \
    --run_opts.learning_rate 3.0e-4 \
    --run_opts.cycle_steps 200000 \
    --run_opts.val_check_interval 10000 \
    --run_opts.use_fused_kernel True \
    --run_opts.use_causal_conformer True \
    --run_opts.conv_depthwise_kernel_size 5 \
