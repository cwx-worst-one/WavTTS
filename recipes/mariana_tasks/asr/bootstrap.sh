#!/bin/bash -ex

cd $(dirname $0)/../../../

WORK_DIR=$(pwd)

# for debug
# --trainer.logger=['console'] \
# --model.network.n_layer=4 \
# --model.network.n_embed=256 \
# --model.network.n_inner=1024 \
# --model.partial_pretrain=/mnt/bn/by-nas/Speech/Codes/LanguageModels/ckpts/Seed-13B-SFT-P2.0.0_D4.0.0_T600B-SFT3.8.0/checkpoints/global_step_2860/zero3_merge_states_with_bestrq_codes.pt \

hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/experiments/mariana/training_miscs
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/experiments/mariana/bbpe64k-0303
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/experiments/mariana/global_step_2860_zero3_merge_states_with_bestrq_codes.pt

bash cruise_launch.sh -m recipes.mariana_tasks.asr.train \
--model=recipes/mariana_tasks/configs/13b_v120_bbpe_64k.yaml \
--model.network.vocab_size=65026 \
--model.network.use_rmpad=true \
--model.network.use_ft_flash_attn=true \
--model.network.use_ft_linear=true \
--model.network.use_ft_layernorm=true \
--model.network.pad_idx=1 \
--model.network.gradient_checkpointing=true \
--model.network.resid_pdrop=0.1 \
--model.network.embd_pdrop=0.0 \
--model.network.attn_pdrop=0.1 \
--model.partial_pretrain=${WORK_DIR}/global_step_2860_zero3_merge_states_with_bestrq_codes.pt \
--trainer=recipes/mariana_tasks/configs/zero3-no-cg.yaml \
--trainer.max_epochs=20 \
--data.train_size=113308140 \
--data.train_batch_size=4 \
--trainer.precision=bf16 \
--trainer.optimizer_kwargs.scheduler.params.warmup_step_rate=0.0001 \
--trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
--trainer.accumulate_grad_batches=4 \
--trainer.gradient_clip_val=1.0 \
--trainer.default_hdfs_dir=hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/experiments/mariana/test_spokenllm_new_icm \
--data.train_path=hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/test_by_asr/train_sub*_rank*_shard*.parquet \
--data.val_path=hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/test_by_asr/cv/rank*_shard*.parquet \
--data.source_types=['parquet'] \
--data.tokenizer_type=bbpe \
--data.tokenizer=${WORK_DIR}/bbpe64k-0303 \
--data.template_fn=${WORK_DIR}/training_miscs/templates_v0.1.txt \
--data.val_template_fn=${WORK_DIR}/training_miscs/templates.txt \
--merge_zero3_states=True "$@"
