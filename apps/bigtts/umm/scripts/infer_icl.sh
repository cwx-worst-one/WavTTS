#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:/mnt/bn/music-ai-unified-repo-lq/yangbing/workspace/samantha
# export ARNOLD_WORKER_GPU=1
# export CUDA_VISIBLE_DEVICES=0

seed=1997
src_lang=zh
tgt_lang=zh

meta_lst="/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_0420_raw.denoise/meta_zh_cmos_filter.txt"
ar_ckpt_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenkuan/pretrain/did3259_icl_gqa_96A100_bf16_240806.avg.ckpt"
diffusion_ckpt_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/consistency_distill_fixed/CD_block32_8H800_setting0_UMM062_UMMnoDrop_LR8e-6_guid1to6_tDiff0.02_X0Prob0.1_L2SSIM/checkpoints/epoch=00-step=300000-loss=0.06.ckpt"

output_path="/mnt/bn/chenkuan-nas-lq-001/outputs/temp/icl_output"

bash apps/bigtts/umm/scripts/infer_ar_diffusion_two_stage_icl.sh \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --ar_ckpt_path $ar_ckpt_path \
    --meta_lst $meta_lst \
    --output_path $output_path 
    