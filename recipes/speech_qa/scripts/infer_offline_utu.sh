#!/bin/bash
export MUSICLM_DIR=/mnt/bn/cyz-lq-nas/project/samantha
cd $MUSICLM_DIR

python3 $MUSICLM_DIR/recipes/speech_qa/scripts/infer_audiogpt_utu.py \
    --input_dir=/mnt/bn/cyz-lq-nas/input_dir/test_offline \
    --output_dir=/mnt/bn/cyz-lq-nas/output_dir/test_offline_out \
    --ckpt=/mnt/bn/cyz-lq-nas/work_dir/audiogpt/test0420_utucd/version_0.4/checkpoints/epoch=00-step=28000-accu=54.81.ckpt \
    --token_size=2077 \
    --st=1.0 \
    --sample_mode=gumbel
