#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE
# CUDA_LAUNCH_BLOCKING=1 # only for debug


# test LT MAE
bash launch.sh fit --config recipes/valle/conf/llama/continuous_llama_ctiga_WFVAE_LT.yaml \
--run_opts.train_meta_lst /mnt/bn/cyz-lq-nas-2/data_dir/en_TTS_all_meta_dir/valle_len_metalist_WFVAE_LT.txt \
--run_opts.return_full_seq False \
--run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/WFVAE_LT \
--run_opts.log_name debug_mae \
--run_opts.version 0.0.1 \
--run_opts.batch_total_tokens 15000 \
--run_opts.precision bf16 \
--run_opts.checkpointing False \
--run_opts.n_step_save 5000 \
--trainer.accumulate_grad_batches 1 \
--run_opts.learning_rate 0.0003 \
--scheduler_cls.cycle_steps 400000 \
--run_opts.num_epochs 10000


# bash launch.sh fit --config recipes/valle/conf/llama/continuous_llama.yaml \
# --run_opts.train_meta_lst /mnt/bn/cyz-lq-nas/data_dir/en_TTS_all_meta_dir/valle_len_metalist.txt \
# --run_opts.return_full_seq False \
# --run_opts.log_dir /mnt/bn/cyz-lq-nas/work_dir/text2semantic/baseline \
# --run_opts.log_name delta \
# --run_opts.version 0.0.1 \
# --run_opts.batch_total_tokens 10000 \
# --distributed_batch_sampler_similar_length.seed `echo $[$(date +%s%N)/1000000]` \
# --run_opts.num_epochs 500


