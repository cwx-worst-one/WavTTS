#!/bin/bash
export MUSICLM_DIR=/mnt/bn/cyz-lq-nas/project/samantha
cd $MUSICLM_DIR

python3 $MUSICLM_DIR/recipes/speech_qa/scripts/infer_audiogpt.py \
    --input_dir=/mnt/bn/cyz-lq-nas/input_dir/test_prompt \
    --output_dir=/mnt/bn/cyz-lq-nas/output_dir/test_prompt_out \
    --ckpt=/mnt/bn/cyz-lq-nas/work_dir/audiogpt/test0415_8gpu/version_0.4/checkpoints/last.ckpt \
    --token_size=1024 \
    --sample_mode=gumbel
