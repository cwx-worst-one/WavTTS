mode=$1

exp=DualCondNet2_250M_bf16_16A100_18wEN_10wCN_40hzWvae_UMMv02
logdir=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/congjian/voicebox/logdir/
logdir=$logdir/$exp/

#  for debug. ARNOLD_WORKER_GPU=1
if [[ ${mode} == "train" ]];then
    bash launch.sh fit --config recipes/voicebox/conf/PrefixLDM3a_DualCondUNet2_small_40hzUMM_40hzMel_parquet.yaml \
       --run_opts.data_id 470 \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path $logdir \
        --run_opts.log_name umm_diffusion_wvae \
        --run_opts.version $exp \
        --run_opts.batch_total_tokens 30000 \
        --run_opts.checkpointing False \
        --run_opts.n_step_save 5000 \
        --trainer.accumulate_grad_batches 1 \
        --scheduler_cls.cycle_steps 500000 \
        --run_opts.num_workers 6 \
        --trainer.log_every_n_steps 100 \
        --pl_module.umm_dropout 0.2 \
       	--train_dataset.mask_use_alignment False \
		--run_opts.target bn \
		--run_opts.prompt_feature bn \
		--run_opts.ctx_feature bn \
		--model_config.in_channels 64 \
		--model_config.out_channels 64 \
		--model_config.prompt_mel_dim 64
fi