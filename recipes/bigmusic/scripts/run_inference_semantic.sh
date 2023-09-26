#!/bin/bash
# OUTPUT_DIR=$1 # enable 

# MOS comparison
OUTPUT_DIR=results/generated_outputs
PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv # latest rewrite
FOLDER_NAME=$(basename $PROMPT_PATH .csv)
mkdir -p $OUTPUT_DIR

# Diffusion example
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_chroma_03B_30s_b20w6_140k \
    --extra_params.prompt_path $PROMPT_PATH  \
    --extra_params.token2wav_type diffusion

# 3AR example
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_1.9b_3tag_diffusion \
    --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.token2wav_type ar

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd
