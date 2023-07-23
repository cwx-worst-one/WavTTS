#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE
# CUDA_LAUNCH_BLOCKING=1 # only for debug



# test LT
# bash launch.sh fit --config recipes/valle/conf/llama/vae_llama_ctiga_WFVAE_LT.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas-2/data_dir/en_TTS_all_meta_dir/valle_len_metalist_WFVAE_LTV2.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/WFVAE_LT \
# --run_opts.log_name debug_V2 \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 15000 \
# --run_opts.precision bf16 \
# --run_opts.checkpointing False \
# --run_opts.n_step_save 5000 \
# --trainer.accumulate_grad_batches 1 \
# --run_opts.learning_rate 0.0003 \
# --scheduler_cls.cycle_steps 400000 \
# --run_opts.num_epochs 1000


bash launch.sh fit --config recipes/valle/conf/llama/vae_llama_ctiga.yaml \
--run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/avg_valle_len_metalist_WaveformVAE.txt \
--run_opts.return_full_seq False \
--run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae \
--run_opts.log_name avg_ctiga_32A100_bt15000_accu4 \
--run_opts.version 0.0.1 \
--run_opts.batch_total_tokens 15000 \
--run_opts.precision bf16 \
--run_opts.checkpointing False \
--run_opts.n_step_save 5000 \
--trainer.accumulate_grad_batches 4 \
--run_opts.learning_rate 0.0003 \
--scheduler_cls.cycle_steps 400000 \
--run_opts.num_epochs 2000 \
--ckpt_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_ctiga_32A100_bt15000_accu1/0.0.1/checkpoints/epoch=06-step=100000-kl_loss=0.61.ckpt



# --- A100 ---
# bash launch.sh fit --config recipes/valle/conf/llama/vae_llama_ctiga.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/valle_1400h_meta_dir/valle_len_metalist_WaveformVAE.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae \
# --run_opts.log_name baseline_citiga \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 16000 \
# --run_opts.precision bf16 \
# --run_opts.checkpointing False \
# --trainer.accumulate_grad_batches 4 \
# --run_opts.learning_rate 0.0003 \
# --distributed_batch_sampler_similar_length.seed `echo $[$(date +%s%N)/1000000]` \
# --run_opts.num_epochs 1000


# load ckpt
# bash launch.sh fit --config recipes/valle/conf/llama/vae_llama_ctiga.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/valle_len_metalist_SFT_WaveformVAE_noadam_2000.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT \
# --run_opts.log_name SFT_16A100_fix \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 12000 \
# --run_opts.precision bf16 \
# --run_opts.checkpointing False \
# --run_opts.n_step_save 5000 \
# --trainer.accumulate_grad_batches 1 \
# --run_opts.learning_rate 0.0003 \
# --scheduler_cls.cycle_steps 500000 \
# --run_opts.num_epochs 3000 \
# --ckpt_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_from_51k_16A00/0.0.1/checkpoints/last.ckpt


# --ckpt_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT/SFT_from_51k_16A00/0.0.1/checkpoints/last.ckpt



# --model_cls.state_dict_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_ctiga_51k/ctiga_only_weights.ckpt


# bash launch.sh fit --config recipes/valle/conf/llama/vae_llama.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/valle_1400h_meta_dir/valle_len_metalist_WaveformVAE.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae \
# --run_opts.log_name baseline \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 10000 \
# --distributed_batch_sampler_similar_length.seed `echo $[$(date +%s%N)/1000000]` \
# --run_opts.num_epochs 500


# export CUDA_VISIBLE_DEVICES=1
# bash launch.sh fit --config recipes/valle/conf/llama/vae_llama.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/11labs_meta_dir/valle_len_metalist_WaveformVAE_DaceyNew.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae_SFT \
# --run_opts.log_name DaceyNew \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 5000 \
# --run_opts.learning_rate 0.0001 \
# --ckpt_path /mnt/bn/cyz-lq-nas/work_dir/text2semantic/vae/avg_32A100_0712/0.0.1/checkpoints/epoch=08-step=41000-kl_loss=0.61.ckpt \
# --distributed_batch_sampler_similar_length.seed `echo $[$(date +%s%N)/1000000]` \
# --run_opts.num_epochs 1000




