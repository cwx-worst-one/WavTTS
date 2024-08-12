bash launch.sh fit  \
	--config apps/bigtts/umm/diffusion/conf/PrefixLDM4_streaming_25hzUMM_40hzMel_parquet_winmask.yaml \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/test' \
        --run_opts.log_name test \
        --run_opts.version debug \
        --run_opts.batch_total_tokens 30000 \
        --run_opts.n_step_save 5000 \
		--run_opts.data_id 1060 \
		--bn_config.pad_trim False \
        --item_transform.cfg_drop_rate 0.1 \
		--model_config.mask_token True \
		--item_transform.p_drop_bn_ctx 0.2
		#--train_dataset.cfg_drop_rate 0.1 \
        #--model_config.mask_token True
        #--train_dataloader.prefetch_factor 32
