#!/bin/bash -ex

# home_dir=/home/tiger/workdir/
# bash launch.sh $@
set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE
# CUDA_LAUNCH_BLOCKING=1 # only for debug


bash launch.sh fit --config recipes/text2semantic/conf/llama/vae_llama_ctiga_wds.yaml \
--run_opts.urls 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/chenyuanzhe/WFVAE_v2_fixtrim/*/chunk*/*.tar' \
--run_opts.return_full_seq False \
--run_opts.log_dir ./logs \
--run_opts.log_name debug \
--run_opts.version 0.0.1 \
--run_opts.batch_total_tokens 45000 \
--run_opts.precision bf16 \
--run_opts.checkpointing False \
--run_opts.n_step_save 5000 \
--trainer.accumulate_grad_batches 1 \
--run_opts.learning_rate 0.0003 \
--scheduler_cls.cycle_steps 300000 \
--run_opts.simulated_cycle_rate 0.05 \
--run_opts.num_epochs 2000 "$@" 



