#!/bin/bash
# OUTPUT_DIR=$1 # enable 

OUTPUT_DIR=results/output_gt_verify_diffusion
# PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/rich-10s.csv 
PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/karaoke-30s.csv
# PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/multi-tag-30s.csv
FOLDER_NAME=$(basename $PROMPT_PATH .csv)

mkdir -p $OUTPUT_DIR

# MOS comparison

# BestRQ - Vocal Chroma - diffusion - 10s
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_gt_10s.yaml --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/diffusion_125k \
    --extra_params.inference_conditions style_audio --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 16 \
    --extra_params.token2wav_type diffusion


# BestRQ - Vocal Chroma - diffusion - 30s
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_gt_30s.yaml --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/diffusion_125k_30s \
    --extra_params.inference_conditions style_audio --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 16 \
    --extra_params.token2wav_type diffusion


# BestRQ - Vocal Chroma - 2ar - 30s
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_gt_30s.yaml --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/coarse_700m_295k \
    --extra_params.inference_conditions style_audio --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 8 \
    --extra_params.token2wav_type ar \
    --pl_module.required_modules.ar_modules.coarse.hpath /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/coarse_flash_llama_wav2vec/bestrq_chroma_coarse_0.7b/checkpoints/step=295000-val_accu_0=17.92.ckpt

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
