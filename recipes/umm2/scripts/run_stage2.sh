bash scripts/setup_cruise.sh 3897
export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
export LITE_USE_PL_MODULE=1

EXP_NAME=$(date  "+%Y%m%d%H")
EXP_NAME=${EXP_NAME}_UMM-stage2_Dur5-60_causal_GB26X64H20
echo ${EXP_NAME}

hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/token_files/tokenizer /opt/tiger/tokenizer
lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}

bash /opt/tiger/samantha/launch.sh fit \
    --config ./recipes/umm2/conf/conformer_unified_data/dataloader/umm_conformer_stage2_unified_VQ_baseline_re_lite_short.yaml \
    --run_opts.max_steps 800000 \
    --run_opts.version ${EXP_NAME} \
    --run_opts.log_name "umm_stage2_rmpad_v4" \
    --run_opts.hdfs_log_dir "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/debug/" \
    --run_opts.precision "16-mixed" \
    --run_opts.val_check_interval 800000 \
    --data.train_dataloader_config.num_workers 10 \
    --run_opts.learning_rate 2.0e-4 \
    --run_opts.cycle_steps 200000 \
    --run_opts.warmup_steps 30000 \
    --extra_params.min_duration 5 \
    --extra_params.max_duration 60 \
    --extra_params.ctc_downsample true \
    --extra_params.ctc_downsample_rate 2 \
    --extra_params.ctc_ignore_empty True \
    --data.train_batcher.maximum_bucket_size 20880000 \
    --trainer.limit_val_batches 0 \
    --run_opts.use_fused_kernel True \
    --run_opts.use_causal_conformer True \
    --config.conv_depthwise_kernel_size 5 \
    --config.enable_dyna_chunk False \
    --config.conformer_decoder_idx null \
    --run_opts.pretrained_ckpt "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage1/2025080106_UMM-stage1_64GPU_24mins_non_causal_model/checkpoints/step=0150000.ckpt"