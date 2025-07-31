bash scripts/setup_cruise.sh 3955
export PYTHONPATH=/opt/tiger/cruise:$PYTHONPATH
export LITE_USE_PL_MODULE=1
export PYTHONPATH=apps/mariana:$PYTHONPATH

EXP_NAME=$(date  "+%Y%m%d%H")
EXP_NAME=${EXP_NAME}_UMM-stage3_64GPU_causal_conformer_model
echo ${EXP_NAME}

pip3 install confusables thop ipdb --no-deps
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/ --no-dep

hdfs dfs -get hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/bigmusic/tokenizer /opt/tiger/tokenizer


bash /opt/tiger/samantha/launch.sh fit \
    --config recipes/umm2/conf/conformer_unified_data/dataloader/umm_conformer_stage3_unified_VQ_baseline_re_lite.yaml \
    --run_opts.max_steps 800000 \
    --run_opts.version ${EXP_NAME} \
    --run_opts.log_name "umm_stage3_rmpad_v2" \
    --run_opts.hdfs_log_dir "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/debug/" \
    --run_opts.precision "16-mixed" \
    --run_opts.val_check_interval 800000 \
    --data.train_dataloader_config.num_workers 10 \
    --run_opts.learning_rate 1.0e-5 \
    --run_opts.cycle_steps 200000 \
    --run_opts.warmup_steps 4000 \
    --extra_params.min_duration 5 \
    --extra_params.max_duration 60 \
    --extra_params.ctc_downsample true \
    --extra_params.ctc_downsample_rate 2 \
    --extra_params.ctc_ignore_empty True \
    --extra_params.umm_pretrained "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage2_rmpad_v4/2025081712_UMM-stage2_Dur5-60_causal_GB15X64H20/checkpoints/step=0040000.ckpt" \
    --trainer.limit_val_batches 0 \
    --run_opts.use_fused_kernel True \
    --run_opts.use_causal_conformer True \
    --config.conv_depthwise_kernel_size 5 \
    --config.enable_dyna_chunk False \
    --config.conformer_decoder_idx null \
    --data.train_batcher.maximum_bucket_size 17280000 \
    --ckpt_path "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025082216_UMM-stage3_64GPU_12mins_causal_conformer_model/checkpoints/step=0015000.ckpt"



