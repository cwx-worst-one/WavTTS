


bash launch.sh fit --config recipes/valle/conf/llama/vae_llama_ctiga_spkid.yaml \
--run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/valle_len_metalist_SFT_WaveformVAE_noadam_2000_spkid.txt \
--run_opts.return_full_seq False \
--run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT \
--run_opts.log_name SFT_spkid_16A100_fix_2 \
--run_opts.version 0.0.1 \
--run_opts.batch_total_tokens 12000 \
--run_opts.precision bf16 \
--run_opts.checkpointing False \
--run_opts.n_step_save 5000 \
--trainer.accumulate_grad_batches 1 \
--run_opts.learning_rate 0.0003 \
--scheduler_cls.cycle_steps 500000 \
--run_opts.num_epochs 3000 \
--ckpt_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_spkid_16A100_fix/0.0.1/checkpoints/last.ckpt



# --model_cls.state_dict_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_ctiga_55k/ctiga_only_weights.ckpt
