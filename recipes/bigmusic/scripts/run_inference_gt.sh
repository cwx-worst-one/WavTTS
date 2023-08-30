#!/bin/bash
# OUTPUT_DIR=$1 # enable 

OUTPUT_DIR=results/output_gt
PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/rich-10s.csv 
FOLDER_NAME=$(basename $PROMPT_PATH .csv)

mkdir -p $OUTPUT_DIR

# MOS comparison

# BestRQ - Vocal Chroma
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_gt.yaml --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_vocal_chroma_tokens_coarse_gt_60k \
    --extra_params.inference_conditions style_audio --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 54 \
    --extra_params.coarse_ckpt /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/coarse_flash_llama_wav2vec/bestrq_chroma_coarse_0.7b/checkpoints/step=195000-tr_loss=3.6975.ckpt

# BestRQ - Embeds
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_gt.yaml --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_embeds_coarse gt \
    --extra_params.inference_conditions style_audio --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 54 \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse_unified/coarse_flash_llama_wav2vec/bestrq_50k_embeds_v1/checkpoints/step=045000-val_accu_0=31.46.ckpt

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
