bash launch.sh fit --config recipes/voicebox/conf/PrefixLDM4_40hzUMM_40hzMel_parquet.yaml  \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/test' \
		--run_opts.precision bf16 \
        --run_opts.log_name test \
        --run_opts.version debug \
        --run_opts.batch_total_tokens 30000 \
        --run_opts.checkpointing False \
        --run_opts.n_step_save 5000 \
		--run_opts.data_id 606 \
		--train_dataset.mask_use_alignment False \
		--train_dataset.wav_divide 600 \
		--pl_module.umm_dropout 0.2 \
		--model_config.local_cond_dim 512 \
		--model_config.time_embed_dim 512 \
		--model_config.encoder_dim 1024 \
		--model_config.encoder_n_layers 24 \
		--model_config.encoder_n_heads 16 \
		--train_dataset.use_bn False \
		#--model_config.use_token_vector True \
		#--pl_module.required_modules.umm_codebook.ckpt_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/pretrain_model/UMM/V0.2.codebook 
		#--model_config.diffusion_use_ctiga False \
		#--model_config.llama_provider default \
		#--run_opts.precision 32 \
		#--model_config.net_name DualCondUNet2 \
		#--model_config.diffusion_use_ctiga True\
		#--model_config.attention_features 64 \
		#--model_config.attention_heads 16 \
		#--model_config.use_positional_embedding False
		#--run_opts.precision 32 \
