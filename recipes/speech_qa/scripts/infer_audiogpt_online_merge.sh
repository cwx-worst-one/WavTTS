#!/bin/bash
export MUSICLM_DIR=/mnt/bn/cyz-lq-nas/project/samantha
cd $MUSICLM_DIR

python3 $MUSICLM_DIR/recipes/speech_qa/scripts/infer_audiogpt_online_merge.py \
    --input_dir=/mnt/bn/cyz-lq-nas/input_dir/test_offline/wav_24k \
    --output_dir=/mnt/bn/cyz-lq-nas/output_dir/test_offline_merge_out \
    --ckpt=/mnt/bn/cyz-lq-nas/work_dir/audiogpt/0428_v0.2_0.3B/version_0.2/checkpoints/epoch=15-step=54000-accu=62.51.ckpt \
    --token_size=1024 \
    --st=0.6 \
    --sample_mode=gumbel
