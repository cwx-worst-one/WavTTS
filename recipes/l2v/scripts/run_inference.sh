#!/bin/bash
OUTPUT_DIR=$1
# OUTPUT_DIR=/mnt/bn/lyrics-to-song/ashaw/results/l2s/generated_samples
mkdir -p $OUTPUT_DIR


# Vocal Mulan rank180.
# Default.json prompts
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_combinations \
    --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/default.json \
    --extra_params.max_items 64 \
    --extra_params.inference_type text_prompt_combinations \
    --extra_params.conditions text_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 2

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
