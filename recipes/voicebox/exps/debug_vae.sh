bash launch.sh fit --config recipes/voicebox/conf/PrefixLDM4_25hzUMM_40hzMel_parquet.yaml  \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/test' \
		--run_opts.precision bf16 \
        --run_opts.log_name test \
        --run_opts.version debug \
        --run_opts.batch_total_tokens 30000 \
        --run_opts.checkpointing False \
        --run_opts.n_step_save 5000 \
		--run_opts.data_id 652 \
		--train_dataset.mask_use_alignment False \
		--pl_module.umm_dropout 0.2 \
		--run_opts.target bn \
		--run_opts.prompt_feature bn \
		--run_opts.ctx_feature bn \
		--model_config.in_channels 64 \
		--model_config.out_channels 64 \
		--model_config.prompt_mel_dim 64 \
		--model_config.local_cond_dim 512 \
		--model_config.time_embed_dim 256 \
		--model_config.encoder_dim 1024 \
		--model_config.encoder_n_layers 24 \
		--model_config.encoder_n_heads 16 \
		--bn_config.pad_trim False \
		--bn_config.bn_padding -5 \
		--bn_config.bn_norm_std 2 \
		--train_dataset.text_drop_rate 0.25 \
		#--model_config.use_textprefix False \
		#--model_config.use_prompt False \
		#--ckpt_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/PrefixLDM4_300M_32H800_setting3a_40hzWVAE_UMMv062_textDrop0.25_Norm2/checkpoints/epoch=00-step=320000-loss=0.52.ckpt
		#--pl_module.criterions [l2] 
		#--model_config.use_phone_lang True \
		#--train_dataset.use_phone_lang True \
		#--model_config.target_type x0 \
		#--model_config.min_t 0.001 \
		#--model_config.max_t 0.999 \
		#--pl_module.criterions [l1,ssim] \
		#--model_config.use_token_vector True \
		#--pl_module.required_modules.umm_codebook.ckpt_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/pretrain_model/UMM/V0.2.codebook 
		#--model_config.diffusion_use_ctiga False \
		#--model_config.llama_provider default \
		#--run_opts.precision 32 \
