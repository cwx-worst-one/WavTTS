export OVERRIDE_CRUISE_VERSION=2696
export CUDA_VISIBLE_DEVICES=1
export ARNOLD_WORKER_GPU=1

bash launch.sh fit --config apps/bigtts/speechdit/conf/DiT_v1.yaml \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/test' \
        --run_opts.log_name test \
        --run_opts.version debug \
        --run_opts.batch_total_tokens 25000 \
        --run_opts.n_step_save 5000 \
		--run_opts.data_id 1060 \
		--run_opts.valid_data_id 2184 \
		--run_opts.learning_rate 0.0001 \
		--model_config.encoder_dim 1536 \
		--model_config.encoder_n_layers 24 \
		--model_config.encoder_n_heads 24  \
		--model_config.use_seg_embed True \
		--model_config.use_bn_eos_bos True \
		--model_config.target_type  velocity \
		--model_config.flashattn_version 2 \
		--trainer.val_check_interval 1000 \
		--model_config.use_phone_lang True \
		--item_transform.use_phone_lang True \
		#--criterions.pseudohuber.c 0.1
		#--pl_module.criterions "["pseudohuber"]" \
		#--model_config.t_sampling_type logitnormal \
		#--model_config.encoder_n_kv_heads 4 \
		#--model_config.mlp_extend 1.333 \
		#--run_opts.fast_dev_run 5
		#--run_opts.learning_rate 0.00005 \
		#--run_opts.accumulate_grad_batches 1 \
		#--model_config.use_ctc_loss True \
		#--pl_module.use_ctc_loss True \
		#--pl_module.lambda_ctc_loss 0.1
