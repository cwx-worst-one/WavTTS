#export OVERRIDE_CRUISE_VERSION=2696
#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

bash launch.sh fit --config apps/bigtts/speechdit/conf/CD.yaml \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/debug' \
		--run_opts.log_name test \
		--run_opts.version debug \
        --run_opts.batch_total_tokens 30000 \
        --run_opts.n_step_save 10000 \
		--run_opts.data_id 1060 \
		--run_opts.precision "bf16-mixed" \
		--model_config.encoder_dim 1024 \
		--model_config.encoder_n_layers 24 \
		--model_config.encoder_n_heads 16  \
		--model_config.use_seg_embed True \
		--model_config.use_bn_eos_bos True \
		--trainer.log_every_n_steps 10 \
		--pl_module.resume_ckpt_path epoch=00-step=820000-loss=0.522.ckpt 
       # --run_opts.log_name test_distill \
		#--run_opts.data_id 1060 \
		#--distill_config.cfg_drop_rate 0.1 \
		#--distill_config.ode_solver_use_cfg False \
		#--distill_config.lambda_diffusion_loss 0.1 \
		#--run_opts.fast_dev_run 5
		#--run_opts.learning_rate 0.00005 \
		#--run_opts.accumulate_grad_batches 1 \
		#--model_config.use_ctc_loss True \
		#--pl_module.use_ctc_loss True \
		#--pl_module.lambda_ctc_loss 0.1
