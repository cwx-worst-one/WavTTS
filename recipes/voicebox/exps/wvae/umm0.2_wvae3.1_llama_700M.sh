mode=$1

exp=PrefixLDM4_Wvae31_UMM02_700M_Llama

logdir=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/congjian/voicebox/logdir/
logdir=$logdir/$exp/

if [[ ${mode} == "train" ]];then
    bash launch.sh fit --config recipes/voicebox/conf/PrefixLDM4.yaml \
            --run_opts.data_id 470 \
            --run_opts.log_dir ./logs \
            --run_opts.hdfs_path $logdir \
            --run_opts.log_name umm_diffusion_wvae \
            --run_opts.version $exp \
            --run_opts.batch_total_tokens 20000 \
            --run_opts.checkpointing False \
            --run_opts.n_step_save 5000 \
            --trainer.accumulate_grad_batches 1 \
            --scheduler_cls.cycle_steps 500000 \
            --run_opts.num_workers 8 \
            --trainer.log_every_n_steps 100 \
            --pl_module.umm_dropout 0.2 \
            --run_opts.target bn \
            --run_opts.prompt_feature bn \
            --run_opts.ctx_feature bn \
            --model_config.in_channels 64 \
            --model_config.out_channels 64 \
            --model_config.prompt_mel_dim 64 \
            --model_config.local_cond_dim 256 \
            --model_config.time_embed_dim 256 \
            --model_config.encoder_dim 1536 \
            --model_config.encoder_n_layers 24 \
            --model_config.encoder_n_heads 24
fi